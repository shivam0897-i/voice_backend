"""
Voice Analysis Engine.
Combines Wav2Vec2 deepfake detection with signal forensics.
"""
import logging
import os
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import warnings

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Heuristic thresholds (M7 fix: centralised for easy tuning) ──────
HEURISTIC_THRESHOLDS = {
    # Pitch scoring
    "pitch_optimal_stability": float(os.getenv("PITCH_OPTIMAL_STABILITY", "0.20")),
    "pitch_stability_range": float(os.getenv("PITCH_STABILITY_RANGE", "0.20")),
    "pitch_optimal_jitter": float(os.getenv("PITCH_OPTIMAL_JITTER", "0.04")),
    "pitch_jitter_range": float(os.getenv("PITCH_JITTER_RANGE", "0.05")),
    # Spectral scoring
    "spectral_optimal_entropy": float(os.getenv("SPECTRAL_OPTIMAL_ENTROPY", "5.8")),
    "spectral_entropy_range": float(os.getenv("SPECTRAL_ENTROPY_RANGE", "2.5")),
    "spectral_optimal_flatness": float(os.getenv("SPECTRAL_OPTIMAL_FLATNESS", "0.06")),
    "spectral_flatness_range": float(os.getenv("SPECTRAL_FLATNESS_RANGE", "0.08")),
    # Acoustic anomaly
    "anomaly_flatness_threshold": float(os.getenv("ANOMALY_FLATNESS_THRESHOLD", "0.13")),
    "anomaly_voiced_low": float(os.getenv("ANOMALY_VOICED_LOW", "0.35")),
    "anomaly_voiced_high": float(os.getenv("ANOMALY_VOICED_HIGH", "0.95")),
    "anomaly_hnr_low": float(os.getenv("ANOMALY_HNR_LOW", "6.0")),
    "anomaly_hnr_high": float(os.getenv("ANOMALY_HNR_HIGH", "35.0")),
}

# Global model cache
_model = None
_processor = None
_device = None


@dataclass
class AnalysisResult:
    """Result of voice analysis."""
    classification: str  # "AI_GENERATED" or "HUMAN"
    confidence_score: float  # 0.0 to 1.0
    explanation: str
    features: Dict[str, float]  # Individual feature scores for debugging


def get_device():
    """Get the best available device (GPU or CPU)."""
    global _device
    if _device is None:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
        logger.info(f"Using device: {_device}")
    return _device


_invert_labels: bool = False


def _detect_label_inversion(model):
    """Check once at load time whether this model needs label flipping."""
    global _invert_labels
    name = getattr(model.config, '_name_or_path', '').lower()
    if 'shivam-2211' in name or 'voice-detection-model' in name:
        _invert_labels = True
        logger.info("Model has inverted training labels — label flip enabled (logged once)")
    else:
        _invert_labels = False


def load_model():
    """
    Load the Wav2Vec2 deepfake detection model.
    Prioritizes HuggingFace Hub model, with local fallback.
    """
    global _model, _processor, _invert_labels
    
    if _model is None:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        
        # Model priority:
        # 1. Local fine-tuned model (for development)
        # 2. HuggingFace Hub model (for production/deployment)
        # 3. Fallback to public model
        
        local_path = settings.VOICE_MODEL_LOCAL_PATH
        hf_model = settings.VOICE_MODEL_ID
        backup_model = settings.VOICE_MODEL_BACKUP_ID
        
        if os.path.exists(local_path):
            logger.info(f"Loading local fine-tuned model from: {local_path}")
            model_name = local_path
        else:
            logger.info(f"Loading model from HuggingFace Hub: {hf_model}")
            model_name = hf_model
        
        try:
            _processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            _model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            _model.to(get_device())
            _model.eval()
            logger.info(f"Model loaded successfully: {model_name}")
            _detect_label_inversion(_model)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            if model_name != backup_model:
                logger.warning("Trying backup model...")
                model_name = backup_model
                try:
                    _processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                    _model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                    _model.to(get_device())
                    _model.eval()
                    logger.info(f"Backup model loaded: {model_name}")
                    _detect_label_inversion(_model)
                except Exception as e2:
                    raise RuntimeError(f"Could not load any model: {e2}")
            else:
                raise e
    
    return _model, _processor



def extract_signal_features(audio: np.ndarray, sr: int, fast_mode: bool = False) -> Dict[str, float]:
    """Extract signal-based features (pitch, entropy, silence)."""
    import librosa
    from scipy.stats import entropy
    
    features = {}
    
    try:
        # Use smaller FFT in fast mode for realtime throughput.
        n_fft = 512 if fast_mode else 2048
        hop_length = 256 if fast_mode else 512
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))

        # Pitch analysis.
        if fast_mode:
            # Approximate pitch variability from centroid dynamics to avoid expensive pYIN on realtime path.
            spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
            centroid_mean = float(np.mean(spec_centroid) + 1e-8)
            features["pitch_stability"] = float(np.clip(np.var(spec_centroid) / (centroid_mean ** 2), 0.0, 1.5))
            features["jitter"] = float(np.clip(np.mean(np.abs(np.diff(spec_centroid))) / centroid_mean, 0.0, 0.2))
            voiced_flag = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0] > 0.02
        else:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_voiced = f0[~np.isnan(f0)]
            if len(f0_voiced) > 10:
                pitch_mean = np.mean(f0_voiced)
                pitch_std = np.std(f0_voiced)
                features["pitch_stability"] = pitch_std / pitch_mean if pitch_mean > 0 else 0
                features["jitter"] = np.mean(np.abs(np.diff(f0_voiced))) / pitch_mean if pitch_mean > 0 else 0
            else:
                features["pitch_stability"] = 0.5
                features["jitter"] = 0.05

        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        features["spectral_centroid_var"] = float(np.var(spec_centroid))
        
        spec_flatness = librosa.feature.spectral_flatness(S=S)[0]
        features["spectral_flatness"] = float(np.mean(spec_flatness))
        
        # Entropy
        S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
        frame_entropies = [entropy(frame + 1e-10) for frame in S_norm.T]
        features["spectral_entropy"] = float(np.mean(frame_entropies))
        
        # Silence detection
        silence_threshold = 1e-5
        features["silence_ratio"] = float(np.sum(np.abs(audio) < silence_threshold) / len(audio))
        features["perfect_silence"] = float(np.sum(audio == 0) / len(audio))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features["zcr_variance"] = float(np.var(zcr))

        # Additional acoustic heuristics for suspicious audio artifacts.
        spec_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        features["spectral_rolloff_var"] = float(np.var(spec_rolloff))
        features["voiced_ratio"] = float(np.mean(voiced_flag.astype(np.float32))) if voiced_flag is not None else 0.0

        rms = librosa.feature.rms(y=audio)[0]
        features["rms_var"] = float(np.var(rms))

        if fast_mode:
            # Cheap HNR approximation from flatness and entropy for realtime throughput.
            hnr_db = float(max(0.0, 30.0 - (features["spectral_flatness"] * 120.0)))
        else:
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_rms = float(np.sqrt(np.mean(np.square(harmonic))) + 1e-8)
            percussive_rms = float(np.sqrt(np.mean(np.square(percussive))) + 1e-8)
            hnr_db = float(20.0 * np.log10(harmonic_rms / percussive_rms))
        features["harmonic_noise_ratio_db"] = hnr_db
        
    except Exception as e:
        logger.warning(f"Feature extraction error: {e}")
        features = {
            "pitch_stability": 0.5,
            "jitter": 0.05,
            "spectral_centroid_var": 1000,
            "spectral_flatness": 0.1,
            "spectral_entropy": 5.0,
            "silence_ratio": 0.0,
            "perfect_silence": 0.0,
            "zcr_variance": 0.01,
            "spectral_rolloff_var": 50000.0,
            "voiced_ratio": 0.65,
            "rms_var": 0.005,
            "harmonic_noise_ratio_db": 14.0,
        }
    
    return features


def generate_explanation(
    classification: str,
    ml_confidence: float,
    features: Dict[str, float]
) -> str:
    """Generate a data-driven forensic explanation for the classification."""
    
    # Calculate acoustic anomaly scores (0-100 scale)
    pitch_score = _calculate_pitch_score(features)
    spectral_score = _calculate_spectral_score(features)
    temporal_score = _calculate_temporal_score(features)
    
    # Overall authenticity score (inverted for AI detection)
    authenticity_score = (pitch_score + spectral_score + temporal_score) / 3
    
    # Confidence tier affects explanation style
    if ml_confidence >= 0.95:
        confidence_tier = "high"
    elif ml_confidence >= 0.75:
        confidence_tier = "moderate"
    else:
        confidence_tier = "low"
    
    if classification == "AI_GENERATED":
        return _explain_ai_detection(
            confidence_tier, ml_confidence, authenticity_score,
            pitch_score, spectral_score, temporal_score, features
        )
    else:
        return _explain_human_detection(
            confidence_tier, ml_confidence, authenticity_score,
            pitch_score, spectral_score, temporal_score, features
        )


def _calculate_pitch_score(features: Dict[str, float]) -> float:
    """Calculate pitch naturalness score (0-100). Higher = more human-like.
    
    Uses peaked scoring centred on the human sweet-spot so that both
    extremes (too perfect = AI, too erratic = glitch) are penalised.
    """
    pitch_stability = features.get("pitch_stability", 0.5)
    jitter = features.get("jitter", 0.05)
    
    # Human sweet-spot: stability ≈ 0.15-0.25 (natural micro-variation)
    # AI tends to be TOO stable (> 0.30) — penalise perfection.
    optimal_stability = HEURISTIC_THRESHOLDS["pitch_optimal_stability"]
    stability_dev = abs(pitch_stability - optimal_stability) / HEURISTIC_THRESHOLDS["pitch_stability_range"]
    stability_score = max(0.0, min(100.0, 100.0 * (1.0 - stability_dev)))
    
    # Human jitter ≈ 0.02-0.06 (natural pitch wobble)
    # AI jitter often < 0.01 (too clean/monotone)
    optimal_jitter = HEURISTIC_THRESHOLDS["pitch_optimal_jitter"]
    jitter_dev = abs(jitter - optimal_jitter) / HEURISTIC_THRESHOLDS["pitch_jitter_range"]
    jitter_score = max(0.0, min(100.0, 100.0 * (1.0 - jitter_dev)))
    
    return (stability_score * 0.6 + jitter_score * 0.4)


def _calculate_spectral_score(features: Dict[str, float]) -> float:
    """Calculate spectral naturalness score (0-100). Higher = more human-like.
    
    Peaked scoring — too-uniform spectrum (low flatness, very high
    entropy) is penalised as suspicious synthetic perfection.
    """
    entropy = features.get("spectral_entropy", 5.0)
    flatness = features.get("spectral_flatness", 0.1)
    
    # Human sweet-spot: entropy ≈ 5.0-6.5 (rich harmonic content)
    # AI can have extremely high entropy (uniform noise floor) or
    # very low entropy (monotone vocoder).
    optimal_entropy = HEURISTIC_THRESHOLDS["spectral_optimal_entropy"]
    entropy_dev = abs(entropy - optimal_entropy) / HEURISTIC_THRESHOLDS["spectral_entropy_range"]
    entropy_score = max(0.0, min(100.0, 100.0 * (1.0 - entropy_dev)))
    
    # Human flatness ≈ 0.03-0.10 (varied spectral shape)
    # AI often has very low (< 0.02) or very high (> 0.15) flatness.
    optimal_flatness = HEURISTIC_THRESHOLDS["spectral_optimal_flatness"]
    flatness_dev = abs(flatness - optimal_flatness) / HEURISTIC_THRESHOLDS["spectral_flatness_range"]
    flatness_score = max(0.0, min(100.0, 100.0 * (1.0 - flatness_dev)))
    
    return (entropy_score * 0.5 + flatness_score * 0.5)


def _calculate_temporal_score(features: Dict[str, float]) -> float:
    """Calculate temporal/rhythm naturalness score (0-100). Higher = more human-like."""
    zcr_var = features.get("zcr_variance", 0.01)
    silence_ratio = features.get("silence_ratio", 0.0)
    perfect_silence = features.get("perfect_silence", 0.0)
    
    # Penalize digital silence (exact zeros) - strong AI indicator
    digital_penalty = min(50, perfect_silence * 500)
    
    zcr_score = min(100, max(0, zcr_var / 0.02 * 100))
    
    return max(0, zcr_score - digital_penalty)


def _calculate_acoustic_anomaly_score(features: Dict[str, float]) -> float:
    """
    Estimate suspicious acoustic artifact intensity (0-100).
    Higher score indicates stronger synthetic/spoof-like signal artifacts.
    """
    perfect_silence = features.get("perfect_silence", 0.0)
    spectral_flatness = features.get("spectral_flatness", 0.1)
    rolloff_var = features.get("spectral_rolloff_var", 50000.0)
    voiced_ratio = features.get("voiced_ratio", 0.65)
    hnr_db = features.get("harmonic_noise_ratio_db", 14.0)

    digital_artifact_score = min(100.0, perfect_silence * 10000.0)
    flatness_artifact_score = min(100.0, max(0.0, (spectral_flatness - HEURISTIC_THRESHOLDS["anomaly_flatness_threshold"]) * 500.0))
    rolloff_score = min(100.0, max(0.0, (np.log10(rolloff_var + 1.0) - 3.8) * 45.0))

    if voiced_ratio < HEURISTIC_THRESHOLDS["anomaly_voiced_low"]:
        voiced_ratio_score = min(100.0, (HEURISTIC_THRESHOLDS["anomaly_voiced_low"] - voiced_ratio) * 180.0)
    elif voiced_ratio > HEURISTIC_THRESHOLDS["anomaly_voiced_high"]:
        voiced_ratio_score = min(100.0, (voiced_ratio - HEURISTIC_THRESHOLDS["anomaly_voiced_high"]) * 180.0)
    else:
        voiced_ratio_score = 0.0

    if hnr_db < HEURISTIC_THRESHOLDS["anomaly_hnr_low"]:
        hnr_score = min(100.0, (HEURISTIC_THRESHOLDS["anomaly_hnr_low"] - hnr_db) * 8.0)
    elif hnr_db > HEURISTIC_THRESHOLDS["anomaly_hnr_high"]:
        # Raised from 28 dB — clean human recordings regularly exceed 28 dB
        hnr_score = min(100.0, (hnr_db - HEURISTIC_THRESHOLDS["anomaly_hnr_high"]) * 4.0)
    else:
        hnr_score = 0.0

    anomaly_score = (
        (digital_artifact_score * 0.35)
        + (flatness_artifact_score * 0.20)
        + (rolloff_score * 0.20)
        + (voiced_ratio_score * 0.15)
        + (hnr_score * 0.10)
    )
    return float(max(0.0, min(100.0, anomaly_score)))


def _explain_ai_detection(
    confidence_tier: str,
    ml_confidence: float,
    authenticity_score: float,
    pitch_score: float,
    spectral_score: float,
    temporal_score: float,
    features: Dict[str, float]
) -> str:
    """Generate explanation for AI-detected audio."""
    
    # Find the weakest scores (most AI-like characteristics)
    scores = {
        "vocal pitch patterns": pitch_score,
        "spectral characteristics": spectral_score,
        "temporal dynamics": temporal_score
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    
    # Build forensic-style explanation
    primary_indicator = sorted_scores[0][0]
    primary_score = sorted_scores[0][1]
    
    if confidence_tier == "high":
        intro = f"Strong synthetic markers detected (confidence: {ml_confidence:.0%}). "
    elif confidence_tier == "moderate":
        intro = f"Synthetic patterns identified (confidence: {ml_confidence:.0%}). "
    else:
        intro = f"Possible synthetic audio (confidence: {ml_confidence:.0%}). "
    
    # Specific findings based on lowest scoring area
    if primary_indicator == "vocal pitch patterns":
        jitter = features.get("jitter", 0)
        stability = features.get("pitch_stability", 0)
        detail = f"Pitch analysis shows unusually consistent patterns (stability: {stability:.3f}, micro-variation: {jitter:.4f}) - typical of synthesized speech."
    elif primary_indicator == "spectral characteristics":
        entropy = features.get("spectral_entropy", 0)
        flatness = features.get("spectral_flatness", 0)
        detail = f"Spectral fingerprint indicates synthetic generation (complexity: {entropy:.2f}, flatness: {flatness:.3f}) - lacking natural harmonic richness."
    else:
        perfect_silence = features.get("perfect_silence", 0)
        if perfect_silence > 0.005:
            detail = f"Digital artifacts detected: {perfect_silence:.1%} exact-zero samples found, indicating synthetic audio processing."
        else:
            detail = f"Temporal patterns suggest algorithmic generation - rhythm lacks natural human irregularities."
    
    # Add authenticity score as a unique metric
    authenticity_label = "very low" if authenticity_score < 25 else "low" if authenticity_score < 50 else "borderline"
    
    return f"{intro}{detail} Authenticity score: {authenticity_score:.0f}/100 ({authenticity_label})."


def _explain_human_detection(
    confidence_tier: str,
    ml_confidence: float,
    authenticity_score: float,
    pitch_score: float,
    spectral_score: float,
    temporal_score: float,
    features: Dict[str, float]
) -> str:
    """Generate explanation for human-detected audio."""
    
    # Find the strongest scores (most human-like characteristics)
    scores = {
        "vocal pitch patterns": pitch_score,
        "spectral characteristics": spectral_score,
        "temporal dynamics": temporal_score
    }
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    primary_indicator = sorted_scores[0][0]
    primary_score = sorted_scores[0][1]
    
    if confidence_tier == "high":
        intro = f"Strong human voice markers detected (confidence: {ml_confidence:.0%}). "
    elif confidence_tier == "moderate":
        intro = f"Human speech patterns identified (confidence: {ml_confidence:.0%}). "
    else:
        intro = f"Likely human voice (confidence: {ml_confidence:.0%}). "
    
    # Specific findings based on highest scoring area
    if primary_indicator == "vocal pitch patterns":
        jitter = features.get("jitter", 0)
        stability = features.get("pitch_stability", 0)
        detail = f"Natural pitch dynamics confirmed (variability: {stability:.3f}, micro-fluctuations: {jitter:.4f}) - consistent with biological speech production."
    elif primary_indicator == "spectral characteristics":
        entropy = features.get("spectral_entropy", 0)
        detail = f"Rich harmonic structure detected (complexity score: {entropy:.2f}) - characteristic of natural vocal tract resonance."
    else:
        zcr_var = features.get("zcr_variance", 0)
        detail = f"Organic speech rhythm detected (variance: {zcr_var:.4f}) - natural breathing and articulation patterns present."
    
    # Add authenticity score
    authenticity_label = "excellent" if authenticity_score > 75 else "good" if authenticity_score > 50 else "moderate"
    
    return f"{intro}{detail} Authenticity score: {authenticity_score:.0f}/100 ({authenticity_label})."


def classify_with_model(audio: np.ndarray, sr: int) -> Tuple[str, float]:
    """
    Classify audio using the Wav2Vec2 model.
    
    Returns:
        Tuple of (classification, confidence)
    """
    import torch
    import librosa
    
    model, processor = load_model()
    device = get_device()
    
    # Normalize audio to prevent clipping issues
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    # Resample to 16kHz if needed (Wav2Vec2 expects 16kHz)
    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Process audio
    inputs = processor(
        audio,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Temperature scaling: divide logits by T > 1 to soften the
        # probability distribution.  The model routinely saturates at
        # 1.00 confidence for browser-mic audio, leaving zero room for
        # the heuristic cross-check to correct a wrong classification.
        temperature = float(settings.MODEL_LOGIT_TEMPERATURE)
        if temperature > 1.0:
            logits = logits / temperature

        probabilities = torch.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map class to label using the model's id2label config.
    # IMPORTANT: HuggingFace stores id2label with STRING keys ("0", "1")
    # but predicted_class from torch.argmax().item() is an int.
    # We must normalise the keys to int so .get() actually matches.
    raw_id2label = getattr(model.config, 'id2label', None) or {}
    id2label = {int(k): v for k, v in raw_id2label.items()}
    label = id2label.get(predicted_class, 'UNKNOWN')

    logger.debug(
        "Model id2label=%s  predicted_class=%d  resolved_label=%s  probs=%s",
        id2label, predicted_class, label,
        [f"{p:.4f}" for p in probabilities[0].cpu().tolist()],
    )

    # ── Label interpretation ──
    # The primary model (shivam-2211/voice-detection-model) was trained with
    # inverted label semantics: its class-0 output actually corresponds to
    # REAL/human audio and class-1 to FAKE/AI-generated, despite the config
    # claiming 0=FAKE and 1=REAL.  Detected once at load time via
    # _detect_label_inversion().
    if _invert_labels:
        # Flip: treat model class-0 as REAL, class-1 as FAKE
        if predicted_class == 0:
            classification = "HUMAN"
        else:
            classification = "AI_GENERATED"
        # confidence stays the same (model's own softmax output)
    else:
        # Standard mapping: use labels from config
        if label.upper() in ['FAKE', 'SPOOF', 'SYNTHETIC', 'AI']:
            classification = "AI_GENERATED"
        else:
            classification = "HUMAN"
    
    return classification, confidence


def analyze_voice(audio: np.ndarray, sr: int, language: str = "English", realtime: bool = False) -> AnalysisResult:
    """
    Analyze a voice sample and classify as AI-generated or Human.
    
    Args:
        audio: Audio waveform as numpy array
        sr: Sample rate
        language: Language of the audio (for context)
        
    Returns:
        AnalysisResult with classification, confidence, and explanation
        
    Raises:
        ValueError: If audio is too short for reliable analysis
    """
    # Validate minimum audio duration (at least 0.5 seconds for reliable analysis)
    min_duration = 0.5  # seconds
    duration = len(audio) / sr
    if duration < min_duration:
        raise ValueError(f"Audio too short ({duration:.2f}s). Minimum {min_duration}s required for reliable analysis.")
    
    fast_mode = bool(realtime and settings.REALTIME_LIGHTWEIGHT_AUDIO)

    # Get model prediction (legacy/deep path) or defer to lightweight realtime heuristic.
    ml_fallback = False
    classification = "HUMAN"
    ml_confidence = 0.5
    if not fast_mode:
        try:
            classification, ml_confidence = classify_with_model(audio, sr)
        except Exception as e:
            logger.error(f"ML model error: {e}, falling back to signal analysis")
            ml_fallback = True
            classification = "HUMAN"
            ml_confidence = 0.5

    # Extract signal features for explainability.
    features = extract_signal_features(audio, sr, fast_mode=fast_mode)

    # Calculate scores explicitly for return.
    pitch_score = _calculate_pitch_score(features)
    spectral_score = _calculate_spectral_score(features)
    temporal_score = _calculate_temporal_score(features)
    authenticity_score = (pitch_score + spectral_score + temporal_score) / 3
    acoustic_anomaly_score = _calculate_acoustic_anomaly_score(features)

    # Lightweight realtime path avoids transformer inference for throughput.
    if fast_mode:
        ai_probability = max(
            acoustic_anomaly_score / 100.0,
            max(0.0, min(1.0, (52.0 - authenticity_score) / 52.0)),
        )
        classification = "AI_GENERATED" if ai_probability >= 0.56 else "HUMAN"
        ml_confidence = ai_probability if classification == "AI_GENERATED" else (1.0 - ai_probability)
        ml_confidence = float(max(0.5, min(0.99, ml_confidence)))

    # ── Authenticity cross-check ──────────────────────────────────────
    # When the model says AI_GENERATED but the signal forensics indicate
    # human-like audio (high authenticity), moderate the confidence.
    # This prevents a poorly-calibrated model from steamrolling the
    # heuristic evidence.  The model was fine-tuned on curated datasets
    # and can misclassify real browser-mic audio as synthetic.
    #
    # Browser-mic audio typically has authenticity 34-60 and anomaly 40-78
    # (naturally higher noise floor and spectral irregularity). The
    # thresholds must reflect these real-world ranges.
    if classification == "AI_GENERATED" and authenticity_score > 35:
        # The higher the authenticity, the more we moderate.
        # authenticity 35 → no change.  authenticity 60 → cap at ~0.75
        # authenticity 80 → cap at ~0.55
        moderation_factor = max(0.50, 1.0 - (authenticity_score - 35) / 100.0)
        if ml_confidence > moderation_factor:
            logger.info(
                "Authenticity cross-check: moderated AI confidence %.2f → %.2f "
                "(authenticity=%.1f, anomaly=%.1f)",
                ml_confidence, moderation_factor,
                authenticity_score, acoustic_anomaly_score,
            )
            ml_confidence = moderation_factor
        # If authenticity indicates human-like features (>40) and anomaly
        # is not extreme (<65), override the classification — the signal
        # evidence strongly contradicts the model. Browser mic audio
        # naturally has anomaly 22-64 and authenticity 34-68, so the
        # thresholds must cover these real-world ranges.
        if authenticity_score > 40 and acoustic_anomaly_score < 65:
            logger.info(
                "Authenticity override: flipping AI_GENERATED → HUMAN "
                "(authenticity=%.1f, anomaly=%.1f, original_conf=%.2f)",
                authenticity_score, acoustic_anomaly_score, ml_confidence,
            )
            classification = "HUMAN"
            ml_confidence = max(0.55, 1.0 - ml_confidence)  # invert confidence

    features["ml_confidence"] = ml_confidence
    features["ml_fallback"] = float(ml_fallback)
    features["realtime_heuristic_mode"] = float(fast_mode)

    # Add computed high-level scores to features for API response.
    features["authenticity_score"] = round(authenticity_score, 1)
    features["pitch_naturalness"] = round(pitch_score, 1)
    features["spectral_naturalness"] = round(spectral_score, 1)
    features["temporal_naturalness"] = round(temporal_score, 1)
    features["acoustic_anomaly_score"] = round(acoustic_anomaly_score, 1)

    # Generate explanation
    explanation = generate_explanation(classification, ml_confidence, features)
    
    return AnalysisResult(
        classification=classification,
        confidence_score=round(ml_confidence, 2),
        explanation=explanation,
        features=features
    )


# Pre-load model at module import (optional, for faster first request)
def preload_model():
    """Pre-load the model to speed up first request."""
    try:
        load_model()
    except Exception as e:
        logger.error(f"Model preload failed: {e}")
