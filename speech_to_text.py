"""
Speech-to-text helper with optional faster-whisper backend.

The module degrades safely when ASR dependencies are unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

_asr_model = None
_asr_load_attempted = False

LANGUAGE_TO_WHISPER = {
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Telugu": "te",
    # Mixed-language: let Whisper auto-detect per segment (best for code-switching)
    "Hinglish": None,
    "Mixed": None,
    "Auto": None,
}


def _load_asr_model():
    """Load faster-whisper model lazily."""
    global _asr_model, _asr_load_attempted
    if _asr_model is not None:
        return _asr_model
    if _asr_load_attempted:
        return None

    _asr_load_attempted = True
    try:
        from faster_whisper import WhisperModel

        _asr_model = WhisperModel(
            model_size_or_path=settings.ASR_MODEL_SIZE,
            device="cpu",
            compute_type=settings.ASR_COMPUTE_TYPE,
        )
        logger.info(
            "ASR model loaded successfully: size=%s compute_type=%s",
            settings.ASR_MODEL_SIZE,
            settings.ASR_COMPUTE_TYPE,
        )
        return _asr_model
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("ASR model unavailable: %s", exc)
        return None


def _decode_segments(segments: Iterable[Any]) -> Dict[str, Any]:
    """Extract transcript and confidence proxy from whisper segments."""
    transcript_parts = []
    confidence_parts = []

    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            transcript_parts.append(text)
        avg_logprob = getattr(seg, "avg_logprob", None)
        if avg_logprob is not None:
            confidence_parts.append(float(np.exp(min(0.0, avg_logprob))))

    transcript = " ".join(transcript_parts).strip()
    confidence = float(np.mean(confidence_parts)) if confidence_parts else (0.0 if not transcript else 0.5)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "transcript": transcript,
        "confidence": confidence,
    }


def _run_transcribe(model: Any, audio: np.ndarray, language_code: Optional[str]) -> Dict[str, Any]:
    """Run one whisper transcription pass with optional language hint."""
    segments, _ = model.transcribe(
        audio,
        language=language_code,
        beam_size=settings.ASR_BEAM_SIZE,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    return _decode_segments(segments)


def transcribe_audio(audio: np.ndarray, sr: int, language: str) -> Dict[str, Any]:
    """
    Transcribe audio to text.

    Returns:
        {
            "transcript": str,
            "confidence": float [0..1],
            "engine": str,
            "available": bool
        }
    """
    if not settings.ASR_ENABLED:
        return {
            "transcript": "",
            "confidence": 0.0,
            "engine": "disabled",
            "available": False,
        }

    model = _load_asr_model()
    if model is None:
        return {
            "transcript": "",
            "confidence": 0.0,
            "engine": "unavailable",
            "available": False,
        }

    try:
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        audio = np.asarray(audio, dtype=np.float32)
        language_code = LANGUAGE_TO_WHISPER.get(language)

        hinted = _run_transcribe(model, audio, language_code)

        # Recovery path: if language hint produced no/poor text, retry with auto-detect.
        # This improves robustness for mixed-language (Hinglish) and accented input.
        needs_retry = (
            not hinted["transcript"]
            or (language_code is not None and hinted["confidence"] < 0.30)
        )
        if needs_retry:
            autodetect = _run_transcribe(model, audio, None)
            if autodetect["transcript"] and (
                not hinted["transcript"]
                or autodetect["confidence"] > hinted["confidence"]
            ):
                return {
                    "transcript": autodetect["transcript"],
                    "confidence": autodetect["confidence"],
                    "engine": "faster-whisper:auto",
                    "available": True,
                }

        return {
            "transcript": hinted["transcript"],
            "confidence": hinted["confidence"],
            "engine": "faster-whisper",
            "available": True,
        }
    except Exception as exc:  # pragma: no cover - runtime/audio dependent
        logger.warning("ASR transcription failed: %s", exc)
        return {
            "transcript": "",
            "confidence": 0.0,
            "engine": "error",
            "available": False,
        }
