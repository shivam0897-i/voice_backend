"""
FastAPI application for AI-Generated Voice Detection.

Endpoint: POST /api/voice-detection
- Accepts Base64-encoded MP3 audio
- Returns classification (AI_GENERATED or HUMAN) with confidence score
"""
import logging
import asyncio
import uuid
import time
import json
import io
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["1000/minute"])

from audio_utils import decode_base64_audio, load_audio_from_bytes
from model import analyze_voice, AnalysisResult
from speech_to_text import transcribe_audio
from fraud_language import analyze_transcript
from llm_semantic_analyzer import analyze_semantic_with_llm, is_llm_semantic_provider_ready
from privacy_utils import mask_sensitive_entities, sanitize_for_logging
from config import settings

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None

# Computed constraints
MAX_AUDIO_BASE64_LENGTH = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024 * 4 // 3


@dataclass
class SessionState:
    """In-memory state for a real-time analysis session (derived data only)."""
    session_id: str
    language: str
    started_at: str
    status: str = "active"
    chunks_processed: int = 0
    alerts_triggered: int = 0
    max_risk_score: int = 0
    max_cpi: float = 0.0
    final_call_label: str = "UNCERTAIN"
    final_voice_classification: str = "UNCERTAIN"
    final_voice_confidence: float = 0.0
    max_voice_ai_confidence: float = 0.0
    voice_ai_chunks: int = 0
    voice_human_chunks: int = 0
    llm_checks_performed: int = 0
    risk_policy_version: str = settings.RISK_POLICY_VERSION
    risk_history: List[int] = field(default_factory=list)
    transcript_counts: Dict[str, int] = field(default_factory=dict)
    semantic_flag_counts: Dict[str, int] = field(default_factory=dict)
    keyword_category_counts: Dict[str, int] = field(default_factory=dict)
    behaviour_score: int = 0
    session_behaviour_signals: List[str] = field(default_factory=list)
    last_transcript: str = ""
    last_update: Optional[str] = None
    alert_history: List[Dict[str, Any]] = field(default_factory=list)
    llm_last_engine: Optional[str] = None


SESSION_STORE: Dict[str, SessionState] = {}
SESSION_LOCK = asyncio.Lock()
SESSION_STORE_BACKEND_ACTIVE = "memory"
REDIS_CLIENT: Any = None
ASR_INFLIGHT_TASKS: set[asyncio.Task] = set()
ASR_INFLIGHT_LOCK = asyncio.Lock()



def use_redis_session_store() -> bool:
    """Return whether redis-backed session store is active."""
    return SESSION_STORE_BACKEND_ACTIVE == "redis" and REDIS_CLIENT is not None


def initialize_session_store_backend() -> None:
    """Initialize configured session backend with safe fallback to memory."""
    global SESSION_STORE_BACKEND_ACTIVE, REDIS_CLIENT

    configured = str(getattr(settings, "SESSION_STORE_BACKEND", "memory") or "memory").strip().lower()
    if configured != "redis":
        SESSION_STORE_BACKEND_ACTIVE = "memory"
        REDIS_CLIENT = None
        logger.info("Session store backend: memory")
        return

    if redis is None:
        logger.warning("Redis backend requested but redis package is not installed. Falling back to memory store.")
        SESSION_STORE_BACKEND_ACTIVE = "memory"
        REDIS_CLIENT = None
        return

    redis_url = getattr(settings, "REDIS_URL", None)
    if not redis_url:
        logger.warning("Redis backend requested but REDIS_URL is empty. Falling back to memory store.")
        SESSION_STORE_BACKEND_ACTIVE = "memory"
        REDIS_CLIENT = None
        return

    try:
        REDIS_CLIENT = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=max(0.1, float(settings.REDIS_CONNECT_TIMEOUT_MS) / 1000.0),
            socket_timeout=max(0.1, float(settings.REDIS_IO_TIMEOUT_MS) / 1000.0),
        )
        REDIS_CLIENT.ping()
        SESSION_STORE_BACKEND_ACTIVE = "redis"
        logger.info("Session store backend: redis")
    except Exception as exc:
        logger.warning("Failed to initialize redis session store (%s). Falling back to memory store.", exc)
        SESSION_STORE_BACKEND_ACTIVE = "memory"
        REDIS_CLIENT = None


def _session_redis_key(session_id: str) -> str:
    return f"{settings.REDIS_PREFIX}:session:{session_id}"


def _serialize_session(session: SessionState) -> str:
    return json.dumps(asdict(session), ensure_ascii=False, separators=(",", ":"))


def _deserialize_session(raw: Optional[str]) -> Optional[SessionState]:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return None
        return SessionState(**payload)
    except Exception as exc:
        logger.warning("Failed to deserialize session payload: %s", exc)
        return None


def get_session_state(session_id: str) -> Optional[SessionState]:
    """Fetch session state from active backend."""
    if use_redis_session_store():
        raw = REDIS_CLIENT.get(_session_redis_key(session_id))
        return _deserialize_session(raw)
    return SESSION_STORE.get(session_id)


def save_session_state(session: SessionState) -> None:
    """Persist session state to active backend."""
    if use_redis_session_store():
        ttl_seconds = max(1, int(session_retention_seconds(session)))
        REDIS_CLIENT.set(_session_redis_key(session.session_id), _serialize_session(session), ex=ttl_seconds)
        return
    SESSION_STORE[session.session_id] = session


def delete_session_state(session_id: str) -> None:
    """Delete session from active backend."""
    if use_redis_session_store():
        REDIS_CLIENT.delete(_session_redis_key(session_id))
        return
    SESSION_STORE.pop(session_id, None)


def _asr_fallback_result(engine: str) -> Dict[str, Any]:
    return {
        "transcript": "",
        "confidence": 0.0,
        "engine": engine,
        "available": False,
    }


def _discard_asr_task(task: asyncio.Task) -> None:
    ASR_INFLIGHT_TASKS.discard(task)


async def transcribe_audio_guarded(
    audio: np.ndarray,
    sr: int,
    language: str,
    timeout_seconds: float,
    request_id: str,
) -> Dict[str, Any]:
    """Run ASR with timeout and bounded in-flight tasks to avoid thread pileups."""
    max_inflight = max(1, int(getattr(settings, "ASR_MAX_INFLIGHT_TASKS", 1)))

    async with ASR_INFLIGHT_LOCK:
        stale_tasks = [task for task in ASR_INFLIGHT_TASKS if task.done()]
        for stale in stale_tasks:
            ASR_INFLIGHT_TASKS.discard(stale)

        if len(ASR_INFLIGHT_TASKS) >= max_inflight:
            logger.warning(
                "[%s] Realtime ASR skipped (inflight=%s, max=%s); continuing without transcript",
                request_id,
                len(ASR_INFLIGHT_TASKS),
                max_inflight,
            )
            return _asr_fallback_result("busy")

        asr_task = asyncio.create_task(asyncio.to_thread(transcribe_audio, audio, sr, language))
        ASR_INFLIGHT_TASKS.add(asr_task)
        asr_task.add_done_callback(_discard_asr_task)

    try:
        return await asyncio.wait_for(asyncio.shield(asr_task), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(
            "[%s] Realtime ASR timed out after %.0fms; continuing without transcript",
            request_id,
            timeout_seconds * 1000,
        )
        return _asr_fallback_result("timeout")
    except Exception as exc:
        logger.warning("[%s] Realtime ASR path failed: %s; continuing without transcript", request_id, exc)
        return _asr_fallback_result("error")


def warmup_audio_pipeline() -> None:
    """Warm audio decoding stack to reduce first-request latency spikes."""
    if not settings.AUDIO_PIPELINE_WARMUP_ENABLED:
        return
    try:
        import soundfile as sf

        warm_audio = np.zeros(16000, dtype=np.float32)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, warm_audio, 16000, format="WAV", subtype="PCM_16")
        load_audio_from_bytes(wav_buffer.getvalue(), 22050, "wav")
        logger.info("Audio pipeline warm-up complete")
    except Exception as exc:
        logger.warning("Audio pipeline warm-up skipped: %s", exc)


def warmup_asr_pipeline() -> None:
    """Warm ASR model and transcription path on startup."""
    if not settings.ASR_ENABLED or not settings.ASR_WARMUP_ENABLED:
        return
    try:
        warm_audio = np.zeros(16000, dtype=np.float32)
        transcribe_audio(warm_audio, 16000, "English")
        logger.info("ASR warm-up complete")
    except Exception as exc:
        logger.warning("ASR warm-up skipped: %s", exc)


def warmup_voice_pipeline() -> None:
    """Run one inference pass to avoid first realtime-model cold latency spike."""
    if not settings.VOICE_WARMUP_ENABLED:
        return
    try:
        sr = 16000
        duration_sec = 1.0
        sample_count = max(1, int(sr * duration_sec))
        t = np.linspace(0.0, duration_sec, sample_count, endpoint=False, dtype=np.float32)
        # Non-silent tone avoids edge-case feature paths and mirrors short speech chunks.
        warm_audio = (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        analyze_voice(warm_audio, sr, "English", True)
        logger.info("Voice model warm-up complete")
    except Exception as exc:
        logger.warning("Voice model warm-up skipped: %s", exc)


def run_startup_warmups() -> None:
    """Run non-critical startup warm-ups for latency-sensitive paths."""
    warmup_audio_pipeline()
    warmup_voice_pipeline()
    warmup_asr_pipeline()


# Detect environment
if settings.SPACE_ID:
    logger.info(f"Running on HuggingFace Spaces: {settings.SPACE_ID}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    logger.info("Starting up - preloading ML model...")
    initialize_session_store_backend()
    try:
        from model import preload_model
        preload_model()
        logger.info("ML model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")

    try:
        await asyncio.to_thread(run_startup_warmups)
    except Exception as exc:
        logger.warning("Startup warm-ups encountered an issue: %s", exc)

    yield
    # Shutdown
    logger.info("Shutting down...")


from fastapi.responses import RedirectResponse

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether a voice sample is AI-generated or spoken by a real human",
    version="1.0.0",
    contact={
        "name": "Shivam",
        "url": settings.WEBSITE_URL,
    },
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware configuration
# CORS
# Note: Set ALLOWED_ORIGINS env var in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "x-api-key", "Authorization"],
)

# Request Logging & Timing Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Generate request ID and start timer
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start_time = time.perf_counter()
    
    # Log request start
    method = request.method
    path = request.url.path
    if method == "POST":
        logger.info(f"[{request_id}] [START] {method} {path}")
    
    # Process request (async)
    response = await call_next(request)
    
    # Calculate duration
    duration_ms = (time.perf_counter() - start_time) * 1000
    status_code = response.status_code
    
    # Log request completion with timing
    if method == "POST":
        status_label = "[OK]" if status_code == 200 else "[ERR]" if status_code >= 400 else "[WARN]"
        logger.info(f"[{request_id}] {status_label} END {method} {path} -> {status_code} ({duration_ms:.0f}ms)")
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.0f}ms"
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Allow embedding in Hugging Face iframe
    # response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    # Relax CSP to allow standard API documentation via CDNs (ReDoc/Swagger)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https://fastapi.tiangolo.com;"
    )
    return response


# Request/Response Models
class VoiceDetectionRequest(BaseModel):
    """Request body for voice detection."""
    language: str = Field(..., description="Language: Tamil, English, Hindi, Malayalam, or Telugu")
    audioFormat: str = Field(default="mp3", description="Audio format (must be mp3)")
    audioBase64: str = Field(..., description="Base64-encoded MP3 audio")

    @field_validator('audioBase64')
    @classmethod
    def validate_audio_size(cls, v: str) -> str:
        """Validate audio data is not too small or too large."""
        if len(v) < 100:
            raise ValueError("Audio data too small - provide valid audio content")
        if len(v) > MAX_AUDIO_BASE64_LENGTH:
            raise ValueError(f"Audio data too large - maximum {settings.MAX_AUDIO_SIZE_MB}MB allowed")
        return v


class ForensicMetrics(BaseModel):
    """Detailed forensic analysis metrics."""
    authenticity_score: float = Field(..., description="Overall voice naturalness score (0-100)")
    pitch_naturalness: float = Field(..., description="Pitch stability and jitter score (0-100)")
    spectral_naturalness: float = Field(..., description="Spectral entropy and flatness score (0-100)")
    temporal_naturalness: float = Field(..., description="Rhythm and silence score (0-100)")


class VoiceDetectionResponse(BaseModel):
    """Successful response from voice detection."""
    status: str = "success"
    language: str
    classification: str  # AI_GENERATED or HUMAN
    confidenceScore: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    forensic_metrics: Optional[ForensicMetrics] = None
    modelUncertain: bool = False
    recommendedAction: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    status: str = "error"
    message: str


class SessionStartRequest(BaseModel):
    """Request body for creating a real-time analysis session."""
    language: str = Field(..., description="Language: Tamil, English, Hindi, Malayalam, or Telugu")


class SessionStartResponse(BaseModel):
    """Response body after creating a session."""
    status: str = "success"
    session_id: str
    language: str
    started_at: str
    message: str


class SessionChunkRequest(BaseModel):
    """Audio chunk request for real-time analysis."""
    audioFormat: str = Field(default="mp3", description="Audio format (must be one of supported formats)")
    audioBase64: str = Field(..., description="Base64-encoded audio chunk")
    language: Optional[str] = Field(default=None, description="Optional override. Defaults to session language")

    @field_validator("audioBase64")
    @classmethod
    def validate_chunk_size(cls, v: str) -> str:
        if len(v) < 100:
            raise ValueError("Audio data too small - provide valid audio content")
        if len(v) > MAX_AUDIO_BASE64_LENGTH:
            raise ValueError(f"Audio data too large - maximum {settings.MAX_AUDIO_SIZE_MB}MB allowed")
        return v


class RiskEvidence(BaseModel):
    """Model evidence used to produce risk score."""
    audio_patterns: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    behaviour: List[str] = Field(default_factory=list)


class RealTimeLanguageAnalysis(BaseModel):
    """Transcript and language risk signals for the current chunk."""
    transcript: str = ""
    transcript_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    asr_engine: str = "unavailable"
    keyword_hits: List[str] = Field(default_factory=list)
    keyword_categories: List[str] = Field(default_factory=list)
    semantic_flags: List[str] = Field(default_factory=list)
    keyword_score: int = Field(default=0, ge=0, le=100)
    semantic_score: int = Field(default=0, ge=0, le=100)
    behaviour_score: int = Field(default=0, ge=0, le=100)
    session_behaviour_signals: List[str] = Field(default_factory=list)
    llm_semantic_used: bool = False
    llm_semantic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    llm_semantic_model: Optional[str] = None


class RealTimeAlert(BaseModel):
    """Alert details emitted by the risk engine."""
    triggered: bool
    alert_type: Optional[str] = None
    severity: Optional[str] = None
    reason_summary: Optional[str] = None
    recommended_action: Optional[str] = None


class ExplainabilitySignal(BaseModel):
    """Per-signal contribution to fused risk score."""
    signal: str
    raw_score: int = Field(..., ge=0, le=100)
    weight: float = Field(..., ge=0.0, le=1.0)
    weighted_score: float = Field(..., ge=0.0, le=100.0)


class RealTimeExplainability(BaseModel):
    """Human-readable explainability block for chunk risk output."""
    summary: str
    top_indicators: List[str] = Field(default_factory=list)
    signal_contributions: List[ExplainabilitySignal] = Field(default_factory=list)
    uncertainty_note: Optional[str] = None


class RealTimeUpdateResponse(BaseModel):
    """Chunk-by-chunk update response."""
    status: str = "success"
    session_id: str
    timestamp: str
    risk_score: int = Field(..., ge=0, le=100)
    cpi: float = Field(..., ge=0.0, le=100.0, description="Conversational Pressure Index")
    risk_level: str
    call_label: str
    model_uncertain: bool = False
    voice_classification: str = "UNCERTAIN"
    voice_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: RiskEvidence
    language_analysis: RealTimeLanguageAnalysis
    alert: RealTimeAlert
    explainability: RealTimeExplainability
    chunks_processed: int = Field(..., ge=1)
    risk_policy_version: str = settings.RISK_POLICY_VERSION


class SessionSummaryResponse(BaseModel):
    """Summary response for a completed or active session."""
    status: str = "success"
    session_id: str
    language: str
    session_status: str
    started_at: str
    last_update: Optional[str] = None
    chunks_processed: int = 0
    alerts_triggered: int = 0
    max_risk_score: int = 0
    max_cpi: float = 0.0
    final_call_label: str = "UNCERTAIN"
    final_voice_classification: str = "UNCERTAIN"
    final_voice_confidence: float = 0.0
    max_voice_ai_confidence: float = 0.0
    voice_ai_chunks: int = 0
    voice_human_chunks: int = 0
    llm_checks_performed: int = 0
    risk_policy_version: str = settings.RISK_POLICY_VERSION


class AlertHistoryItem(BaseModel):
    """One alert event emitted during session analysis."""
    timestamp: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: str
    call_label: str
    alert_type: str
    severity: str
    reason_summary: str
    recommended_action: str


class AlertHistoryResponse(BaseModel):
    """Paginated alert history for one session."""
    status: str = "success"
    session_id: str
    total_alerts: int
    alerts: List[AlertHistoryItem] = Field(default_factory=list)


class RetentionPolicyResponse(BaseModel):
    """Explicit privacy and retention behavior for session processing."""
    status: str = "success"
    raw_audio_storage: str = "not_persisted"
    active_session_retention_seconds: int
    ended_session_retention_seconds: int
    stored_derived_fields: List[str]


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


STORED_DERIVED_FIELDS = [
    "risk_history",
    "behaviour_score",
    "session_behaviour_signals",
    "transcript_counts",
    "semantic_flag_counts",
    "keyword_category_counts",
    "chunks_processed",
    "alerts_triggered",
    "max_risk_score",
    "final_call_label",
    "voice_ai_chunks",
    "voice_human_chunks",
    "max_voice_ai_confidence",
    "final_voice_classification",
    "llm_checks_performed",
]


def parse_iso_timestamp(value: Optional[str]) -> Optional[float]:
    """Convert ISO timestamp to epoch seconds."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def session_reference_timestamp(session: SessionState) -> Optional[float]:
    """Return the best available timestamp for retention checks."""
    return parse_iso_timestamp(session.last_update) or parse_iso_timestamp(session.started_at)


def session_retention_seconds(session: SessionState) -> int:
    """Resolve retention policy from session status."""
    if session.status == "ended":
        return settings.SESSION_ENDED_RETENTION_SECONDS
    return settings.SESSION_ACTIVE_RETENTION_SECONDS


def is_session_expired(session: SessionState, now_ts: Optional[float] = None) -> bool:
    """Check if a session exceeded status-specific retention TTL."""
    reference_ts = session_reference_timestamp(session)
    if reference_ts is None:
        return False
    current = now_ts if now_ts is not None else time.time()
    return (current - reference_ts) > session_retention_seconds(session)


def purge_expired_sessions(now_ts: Optional[float] = None) -> int:
    """Best-effort retention purge for stale sessions (memory backend)."""
    if use_redis_session_store():
        # Redis keys self-expire by TTL; no in-process purge needed.
        return 0

    current = now_ts if now_ts is not None else time.time()
    expired_ids = [sid for sid, state in SESSION_STORE.items() if is_session_expired(state, current)]
    for expired_id in expired_ids:
        delete_session_state(expired_id)
    return len(expired_ids)


def validate_supported_language(language: str) -> None:
    """Validate supported language."""
    if language not in settings.SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Unsupported language. Must be one of: {', '.join(settings.SUPPORTED_LANGUAGES)}"
            }
        )


def validate_supported_format(audio_format: str) -> None:
    """Validate supported audio format."""
    normalized = audio_format.lower()
    if normalized not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Unsupported audio format. Must be one of: {', '.join(settings.SUPPORTED_FORMATS)}"
            }
        )


def normalize_transcript_for_behavior(transcript: str) -> str:
    """Normalize transcript for repetition and trend analysis."""
    lowered = transcript.lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered)
    return " ".join(cleaned.split())


def token_overlap_ratio(text_a: str, text_b: str) -> float:
    """Compute Jaccard overlap between token sets."""
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Return unique string items while preserving first-seen order."""
    seen = set()
    deduped: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped

def update_session_behaviour_state(session: SessionState, language_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Update session-level behaviour score from transcript and semantic trends."""
    transcript_source = str(language_analysis.get("transcript_raw", language_analysis.get("transcript", "")))
    transcript = normalize_transcript_for_behavior(transcript_source)
    semantic_flags = list(language_analysis.get("semantic_flags", []))
    keyword_categories = list(language_analysis.get("keyword_categories", []))

    for flag in semantic_flags:
        session.semantic_flag_counts[flag] = session.semantic_flag_counts.get(flag, 0) + 1
    for category in keyword_categories:
        session.keyword_category_counts[category] = session.keyword_category_counts.get(category, 0) + 1

    behavior_signals: List[str] = []

    if transcript:
        count = session.transcript_counts.get(transcript, 0) + 1
        session.transcript_counts[transcript] = count
        if count >= 2:
            behavior_signals.append("repetition_loop")
        if session.last_transcript:
            overlap = token_overlap_ratio(transcript, session.last_transcript)
            if overlap >= 0.75 and len(transcript.split()) >= 4:
                behavior_signals.append("repetition_loop")
        session.last_transcript = transcript

    urgency_count = session.semantic_flag_counts.get("urgency_language", 0)
    if urgency_count >= 2:
        behavior_signals.append("sustained_urgency")

    has_impersonation = session.semantic_flag_counts.get("authority_impersonation", 0) > 0
    has_credentials = session.semantic_flag_counts.get("credential_request", 0) > 0
    has_payment = session.semantic_flag_counts.get("payment_redirection", 0) > 0
    has_threat = session.semantic_flag_counts.get("coercive_threat_language", 0) > 0
    has_urgency = urgency_count > 0

    if has_impersonation and has_credentials:
        behavior_signals.append("impersonation_plus_credential_request")
    if has_payment and has_urgency:
        behavior_signals.append("persistent_payment_pressure")
    if has_threat and has_urgency:
        behavior_signals.append("repeated_threat_urgency")

    repeated_categories = sum(1 for count in session.keyword_category_counts.values() if count >= 2)
    if repeated_categories >= 2:
        behavior_signals.append("repeated_fraud_categories")

    behavior_signals = sorted(set(behavior_signals))

    score = 0
    if "repetition_loop" in behavior_signals:
        max_repetition = max(session.transcript_counts.values()) if session.transcript_counts else 2
        score += 25 + min(15, (max_repetition - 2) * 5)
    if "sustained_urgency" in behavior_signals:
        score += 15 + min(10, (urgency_count - 2) * 5)
    if "impersonation_plus_credential_request" in behavior_signals:
        score += 30
    if "persistent_payment_pressure" in behavior_signals:
        score += 20
    if "repeated_threat_urgency" in behavior_signals:
        score += 15
    if "repeated_fraud_categories" in behavior_signals:
        score += 10

    session.behaviour_score = max(0, min(100, score))
    session.session_behaviour_signals = behavior_signals

    return {
        "behaviour_score": session.behaviour_score,
        "session_behaviour_signals": session.session_behaviour_signals,
    }


def map_score_to_level(score: int) -> str:
    """Map numeric score to risk level."""
    if score < 35:
        return "LOW"
    if score < 60:
        return "MEDIUM"
    if score < 80:
        return "HIGH"
    return "CRITICAL"


def map_level_to_label(risk_level: str, model_uncertain: bool) -> str:
    """Map risk level to user-friendly label."""
    if model_uncertain:
        return "UNCERTAIN"
    if risk_level == "LOW":
        return "SAFE"
    if risk_level == "MEDIUM":
        return "SPAM"
    return "FRAUD"


def recommendation_for_level(risk_level: str, model_uncertain: bool) -> str:
    """Return a user action recommendation based on severity."""
    if model_uncertain:
        return "Model uncertainty detected. Avoid sharing OTP/PIN and verify caller via official channel."
    if risk_level == "CRITICAL":
        return "High fraud risk. End the call and verify through an official support number."
    if risk_level == "HIGH":
        return "Fraud indicators detected. Do not share OTP, PIN, passwords, or UPI credentials."
    if risk_level == "MEDIUM":
        return "Suspicious call behavior detected. Verify caller identity before taking action."
    return "No high-risk fraud indicators detected in current chunk."



def should_invoke_llm_semantic(
    provisional_scored: Dict[str, Any],
    transcript: str,
    transcript_confidence: float,
    next_chunk_index: int,
) -> bool:
    """Gate optional LLM semantic calls for ambiguous/uncertain chunks."""

    if not settings.LLM_SEMANTIC_ENABLED:
        return False
    if not is_llm_semantic_provider_ready():
        return False
    if not transcript.strip():
        return False
    if len(transcript.strip()) < 8:
        return False
    if transcript_confidence < settings.LLM_SEMANTIC_MIN_ASR_CONFIDENCE:
        return False

    interval = max(1, settings.LLM_SEMANTIC_CHUNK_INTERVAL)
    if next_chunk_index > 1 and (next_chunk_index % interval) != 0:
        return False

    risk_score = int(provisional_scored.get("risk_score", 0))
    model_uncertain = bool(provisional_scored.get("model_uncertain", False))
    ambiguous_band = 35 <= risk_score < 80
    return ambiguous_band or model_uncertain


def normalize_voice_classification(classification: str, model_uncertain: bool) -> str:
    """Normalize realtime voice-authenticity classification."""
    if model_uncertain:
        return "UNCERTAIN"
    normalized = str(classification or "HUMAN").upper()
    if normalized in {"AI_GENERATED", "HUMAN"}:
        return normalized
    return "HUMAN"

def build_explainability_payload(
    risk_level: str,
    call_label: str,
    model_uncertain: bool,
    cpi: float,
    audio_score: int,
    keyword_score: int,
    semantic_score: int,
    behaviour_score: int,
    has_language_signals: bool,
    behaviour_signals: List[str],
    keyword_hits: List[str],
    acoustic_anomaly: float,
    risk_score: int = 0,
    delta_boost: int = 0,
    voice_classification: str = "UNCERTAIN",
    voice_confidence: float = 0.0,
    authenticity_score: float = 50.0,
) -> RealTimeExplainability:
    """Build explicit explainability signals and concise summary."""
    if has_language_signals:
        raw_weights = {
            "audio": settings.RISK_WEIGHT_AUDIO,
            "keywords": settings.RISK_WEIGHT_KEYWORD,
            "semantic": settings.RISK_WEIGHT_SEMANTIC,
            "behaviour": settings.RISK_WEIGHT_BEHAVIOUR,
        }
        total = sum(raw_weights.values()) or 1.0
        weights = {k: v / total for k, v in raw_weights.items()}
    else:
        weights = {
            "audio": 1.00,
            "keywords": 0.00,
            "semantic": 0.00,
            "behaviour": 0.00,
        }

    contributions = [
        ExplainabilitySignal(
            signal="audio",
            raw_score=audio_score,
            weight=weights["audio"],
            weighted_score=round(audio_score * weights["audio"], 2),
        ),
        ExplainabilitySignal(
            signal="keywords",
            raw_score=keyword_score,
            weight=weights["keywords"],
            weighted_score=round(keyword_score * weights["keywords"], 2),
        ),
        ExplainabilitySignal(
            signal="semantic_intent",
            raw_score=semantic_score,
            weight=weights["semantic"],
            weighted_score=round(semantic_score * weights["semantic"], 2),
        ),
        ExplainabilitySignal(
            signal="behaviour",
            raw_score=behaviour_score,
            weight=weights["behaviour"],
            weighted_score=round(behaviour_score * weights["behaviour"], 2),
        ),
    ]

    # Build top indicators from actual detection signals
    indicators: List[str] = []
    if voice_classification == "AI_GENERATED":
        indicators.append("ai_voice_detected")
        if voice_confidence >= 0.90:
            indicators.append("high_confidence_synthetic")
    if authenticity_score < 40:
        indicators.append("low_authenticity_score")
    if acoustic_anomaly >= 45:
        indicators.append("acoustic_anomaly_detected")
    indicators.extend(behaviour_signals)
    indicators.extend(keyword_hits[:3])
    deduped_indicators = list(dict.fromkeys(indicators))[:6]

    base_from_signals = sum(c.weighted_score for c in contributions)
    summary_parts: List[str] = [
        f"{risk_level.title()} risk classified as {call_label}."
    ]
    if delta_boost > 0:
        summary_parts.append(f"Score {risk_score} (base {int(base_from_signals)} + {delta_boost} trend boost).")
    summary_parts.append(f"CPI at {cpi:.1f}/100.")
    if acoustic_anomaly >= 60:
        summary_parts.append("Audio anomalies are materially elevated.")
    if keyword_score >= 45:
        summary_parts.append("Fraud-related keywords contribute to the score.")
    if semantic_score >= 45:
        summary_parts.append("Semantic coercion patterns were detected.")
    if behaviour_score >= 40:
        summary_parts.append("Session behavior trend increases risk.")
    if cpi >= 70:
        summary_parts.append("Pressure escalation velocity is high; early warning triggered.")
    if not has_language_signals:
        summary_parts.append("Assessment is currently audio-dominant.")

    uncertainty_note = None
    if model_uncertain:
        uncertainty_note = (
            "Model confidence is limited for this chunk. Treat this result conservatively and verify through trusted channels."
        )

    return RealTimeExplainability(
        summary=" ".join(summary_parts),
        top_indicators=deduped_indicators,
        signal_contributions=contributions,
        uncertainty_note=uncertainty_note,
    )


def build_risk_update(
    result_features: Dict[str, float],
    classification: str,
    confidence: float,
    language_analysis: Dict[str, Any],
    previous_score: Optional[int],
    llm_semantic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build risk score, evidence and alert from model outputs and session trend."""
    authenticity = float(result_features.get("authenticity_score", 50.0))
    acoustic_anomaly = float(result_features.get("acoustic_anomaly_score", 0.0))
    ml_fallback = bool(result_features.get("ml_fallback", 0.0))
    realtime_heuristic_mode = bool(result_features.get("realtime_heuristic_mode", 0.0))
    normalized_classification = str(classification or "").upper()
    low_confidence_uncertain = bool(
        normalized_classification != "AI_GENERATED"
        and float(confidence) < 0.50
        and int(language_analysis.get("keyword_score", 0)) == 0
        and int(language_analysis.get("semantic_score", 0)) == 0
        and int(language_analysis.get("behaviour_score", 0)) == 0
    )
    heuristic_uncertain = bool(
        realtime_heuristic_mode
        and normalized_classification != "AI_GENERATED"
        and float(confidence) < 0.90
    )
    model_uncertain = ml_fallback or low_confidence_uncertain or heuristic_uncertain
    keyword_score = int(language_analysis.get("keyword_score", 0))
    semantic_score = int(language_analysis.get("semantic_score", 0))
    behaviour_score = int(language_analysis.get("behaviour_score", 0))
    keyword_hits = dedupe_preserve_order(list(language_analysis.get("keyword_hits", [])))
    behavior_from_language = dedupe_preserve_order(list(language_analysis.get("behaviour_signals", [])))
    behavior_from_session = dedupe_preserve_order(list(language_analysis.get("session_behaviour_signals", [])))
    keyword_categories = dedupe_preserve_order(list(language_analysis.get("keyword_categories", [])))
    semantic_flags = dedupe_preserve_order(list(language_analysis.get("semantic_flags", [])))
    transcript = str(language_analysis.get("transcript", "")).strip()

    llm_semantic_used = False
    llm_semantic_confidence = 0.0
    llm_semantic_model: Optional[str] = None
    if llm_semantic and llm_semantic.get("available"):
        blend_weight = max(0.0, min(1.0, settings.LLM_SEMANTIC_BLEND_WEIGHT))
        llm_score = int(max(0, min(100, llm_semantic.get("semantic_score", semantic_score))))
        semantic_score = int(round((semantic_score * (1.0 - blend_weight)) + (llm_score * blend_weight)))
        llm_semantic_confidence = float(max(0.0, min(1.0, llm_semantic.get("confidence", 0.0))))
        llm_semantic_model = str(llm_semantic.get("model") or settings.LLM_SEMANTIC_MODEL)
        llm_semantic_used = True

        keyword_hints = dedupe_preserve_order([str(x) for x in llm_semantic.get("keyword_hints", [])])
        if keyword_hints:
            keyword_hits = dedupe_preserve_order(keyword_hits + keyword_hints)
            keyword_score = min(100, keyword_score + min(18, len(keyword_hints) * 6))

        llm_flags = dedupe_preserve_order([str(x) for x in llm_semantic.get("semantic_flags", [])])
        if llm_flags:
            semantic_flags = dedupe_preserve_order(semantic_flags + llm_flags)

        llm_behaviour = dedupe_preserve_order([str(x) for x in llm_semantic.get("behaviour_signals", [])])
        if llm_behaviour:
            behavior_from_language = dedupe_preserve_order(behavior_from_language + llm_behaviour)

    # Audio signal risk.
    if classification == "AI_GENERATED":
        audio_score = max(
            int(round(confidence * 100)),
            int(max(0.0, min(100.0, acoustic_anomaly * 0.85))),
        )
    else:
        authenticity_audio_score = int(max(0, min(100, (50.0 - authenticity) * 1.2)))
        anomaly_audio_score = int(max(0.0, min(100.0, acoustic_anomaly * 0.90)))
        audio_score = max(authenticity_audio_score, anomaly_audio_score)

    has_language_signals = bool(transcript) or keyword_score > 0 or semantic_score > 0 or behaviour_score > 0
    if has_language_signals:
        raw_weights = {
            "audio": settings.RISK_WEIGHT_AUDIO,
            "keywords": settings.RISK_WEIGHT_KEYWORD,
            "semantic": settings.RISK_WEIGHT_SEMANTIC,
            "behaviour": settings.RISK_WEIGHT_BEHAVIOUR,
        }
        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            raw_weights = {"audio": 0.45, "keywords": 0.20, "semantic": 0.15, "behaviour": 0.20}
            total_weight = 1.0
        normalized = {k: v / total_weight for k, v in raw_weights.items()}

        base_score = int(
            round(
                (audio_score * normalized["audio"])
                + (keyword_score * normalized["keywords"])
                + (semantic_score * normalized["semantic"])
                + (behaviour_score * normalized["behaviour"])
            )
        )
    else:
        base_score = audio_score

    if ml_fallback:
        base_score = max(base_score, 55)

    risk_score = max(0, min(100, base_score))
    behaviour_signals: List[str] = list(behavior_from_language) + list(behavior_from_session)

    if keyword_score >= 60:
        behaviour_signals.append("keyword_cluster_detected")
    if semantic_score >= 60:
        behaviour_signals.append("semantic_coercion_detected")
    if behaviour_score >= 40:
        behaviour_signals.append("behaviour_risk_elevated")
    if acoustic_anomaly >= 60:
        behaviour_signals.append("acoustic_anomaly_detected")

    if previous_score is not None:
        delta = risk_score - previous_score
        if delta >= 15:
            behaviour_signals.append("rapid_risk_escalation")
        if risk_score >= 70 and previous_score >= 70:
            behaviour_signals.append("sustained_high_risk")
    else:
        delta = 0

    delta_boost = 0
    if delta > 0:
        delta_boost = int(delta * settings.RISK_DELTA_BOOST_FACTOR)
        risk_score = min(100, risk_score + delta_boost)

    if previous_score is None:
        cpi = min(100.0, max(0.0, (behaviour_score * 0.35) + (semantic_score * 0.20)))
    else:
        cpi = min(
            100.0,
            max(
                0.0,
                (max(0, delta) * 3.2)
                + (behaviour_score * 0.35)
                + (semantic_score * 0.15),
            ),
        )
    if cpi >= 70:
        behaviour_signals.append("cpi_spike_detected")

    behaviour_signals = dedupe_preserve_order(behaviour_signals)
    risk_level = map_score_to_level(risk_score)
    call_label = map_level_to_label(risk_level, model_uncertain)

    audio_patterns = [
        f"classification:{classification.lower()}",
        f"model_confidence:{confidence:.2f}",
        f"authenticity_score:{authenticity:.1f}",
        f"acoustic_anomaly_score:{acoustic_anomaly:.1f}",
        f"audio_score:{audio_score}",
    ]
    if ml_fallback:
        audio_patterns.append("model_fallback:true")
    audio_patterns = dedupe_preserve_order(audio_patterns)

    strong_intent = {
        "authority_with_credential_request",
        "urgent_payment_pressure",
        "threat_plus_urgency",
        "impersonation_plus_credential_request",
        "persistent_payment_pressure",
        "repeated_threat_urgency",
    }
    alert_triggered = (
        risk_level in {"HIGH", "CRITICAL"}
        or "rapid_risk_escalation" in behaviour_signals
        or cpi >= 70
        or any(signal in behaviour_signals for signal in strong_intent)
    )
    alert_type = None
    severity = None
    reason_summary = None
    recommended_action = None

    if alert_triggered:
        if risk_level == "CRITICAL":
            alert_type = "FRAUD_RISK_CRITICAL"
        elif cpi >= 70:
            alert_type = "EARLY_PRESSURE_WARNING"
        elif "rapid_risk_escalation" in behaviour_signals:
            alert_type = "RISK_ESCALATION"
        else:
            alert_type = "FRAUD_RISK_HIGH"
        severity = risk_level.lower()
        reasons: List[str] = []
        if keyword_hits:
            reasons.append("fraud keywords detected")
        if semantic_score >= 45:
            reasons.append("coercive intent patterns detected")
        if behaviour_score >= 40:
            reasons.append("session behavior risk elevated")
        if "repetition_loop" in behaviour_signals:
            reasons.append("repetition loop detected")
        if "rapid_risk_escalation" in behaviour_signals:
            reasons.append("risk escalated rapidly across chunks")
        if cpi >= 70:
            reasons.append("conversational pressure index spiked")
        if not reasons:
            reasons.append("high-risk audio pattern detected")
        reason_summary = ". ".join(reasons).capitalize() + "."
        recommended_action = recommendation_for_level(risk_level, model_uncertain)

    explainability = build_explainability_payload(
        risk_level=risk_level,
        call_label=call_label,
        model_uncertain=model_uncertain,
        cpi=cpi,
        audio_score=audio_score,
        keyword_score=keyword_score,
        semantic_score=semantic_score,
        behaviour_score=behaviour_score,
        has_language_signals=has_language_signals,
        behaviour_signals=behaviour_signals,
        keyword_hits=keyword_hits,
        acoustic_anomaly=acoustic_anomaly,
        risk_score=risk_score,
        delta_boost=delta_boost,
        voice_classification=classification,
        voice_confidence=confidence,
        authenticity_score=authenticity,
    )

    return {
        "risk_score": risk_score,
        "cpi": round(cpi, 1),
        "risk_level": risk_level,
        "call_label": call_label,
        "model_uncertain": model_uncertain,
        "evidence": RiskEvidence(
            audio_patterns=audio_patterns,
            keywords=keyword_hits,
            behaviour=behaviour_signals
        ),
        "language_analysis": RealTimeLanguageAnalysis(
            transcript=transcript,
            transcript_confidence=float(language_analysis.get("transcript_confidence", 0.0)),
            asr_engine=str(language_analysis.get("asr_engine", "unavailable")),
            keyword_hits=keyword_hits,
            keyword_categories=keyword_categories,
            semantic_flags=semantic_flags,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            behaviour_score=behaviour_score,
            session_behaviour_signals=behavior_from_session,
            llm_semantic_used=llm_semantic_used,
            llm_semantic_confidence=llm_semantic_confidence,
            llm_semantic_model=llm_semantic_model,
        ),
        "alert": RealTimeAlert(
            triggered=alert_triggered,
            alert_type=alert_type,
            severity=severity,
            reason_summary=reason_summary,
            recommended_action=recommended_action
        ),
        "explainability": explainability,
    }


async def process_audio_chunk(
    session_id: str,
    chunk_request: SessionChunkRequest,
    default_language: str,
    request_id: str
) -> RealTimeUpdateResponse:
    """Decode, analyze and score a real-time audio chunk."""
    chunk_language = chunk_request.language or default_language
    validate_supported_language(chunk_language)
    validate_supported_format(chunk_request.audioFormat)

    audio_size_kb = len(chunk_request.audioBase64) * 3 / 4 / 1024
    logger.info(
        f"[{request_id}] Realtime chunk: session={session_id}, language={chunk_language}, "
        f"format={chunk_request.audioFormat}, size~{audio_size_kb:.1f}KB"
    )

    decode_start = time.perf_counter()
    audio_bytes = await asyncio.to_thread(decode_base64_audio, chunk_request.audioBase64)
    decode_ms = (time.perf_counter() - decode_start) * 1000

    load_start = time.perf_counter()
    audio, sr = await asyncio.to_thread(load_audio_from_bytes, audio_bytes, 22050, chunk_request.audioFormat)
    load_ms = (time.perf_counter() - load_start) * 1000

    duration_sec = len(audio) / sr
    logger.info(
        f"[{request_id}] Realtime analyze {duration_sec:.2f}s (decode {decode_ms:.0f}ms, load {load_ms:.0f}ms)"
    )

    analyze_start = time.perf_counter()
    try:
        analysis_result = await asyncio.to_thread(analyze_voice, audio, sr, chunk_language, True)
    except Exception as exc:
        logger.warning("[%s] Realtime model path failed: %s; using conservative fallback", request_id, exc)
        analysis_result = AnalysisResult(
            classification="HUMAN",
            confidence_score=0.5,
            explanation="Realtime model path unavailable; conservative fallback applied.",
            features={
                "ml_fallback": 1.0,
                "authenticity_score": 50.0,
                "pitch_naturalness": 50.0,
                "spectral_naturalness": 50.0,
                "temporal_naturalness": 50.0,
                "acoustic_anomaly_score": 50.0,
            },
        )
    analyze_ms = (time.perf_counter() - analyze_start) * 1000
    logger.info(
        f"[{request_id}] Realtime result: {analysis_result.classification} "
        f"({analysis_result.confidence_score:.0%}) in {analyze_ms:.0f}ms"
    )

    asr_start = time.perf_counter()
    asr_timeout_seconds = max(0.1, float(settings.ASR_TIMEOUT_MS) / 1000.0)
    asr_result = await transcribe_audio_guarded(
        audio=audio,
        sr=sr,
        language=chunk_language,
        timeout_seconds=asr_timeout_seconds,
        request_id=request_id,
    )
    asr_ms = (time.perf_counter() - asr_start) * 1000
    raw_transcript = str(asr_result.get("transcript", ""))
    response_transcript = (
        mask_sensitive_entities(raw_transcript)
        if settings.MASK_TRANSCRIPT_OUTPUT
        else raw_transcript
    )
    language_result = analyze_transcript(raw_transcript, chunk_language)
    language_result["transcript_raw"] = raw_transcript
    language_result["transcript"] = response_transcript
    language_result["transcript_confidence"] = asr_result.get("confidence", 0.0)
    language_result["asr_engine"] = asr_result.get("engine", "unavailable")
    transcript_preview = sanitize_for_logging(raw_transcript, max_chars=90)
    logger.info(
        f"[{request_id}] Realtime ASR: engine={language_result['asr_engine']}, "
        f"confidence={language_result['transcript_confidence']:.2f}, "
        f"text_len={len(raw_transcript)}, preview='{transcript_preview}', asr={asr_ms:.0f}ms"
    )

    # Read-only session snapshot for scoring and optional LLM gating.
    async with SESSION_LOCK:
        purge_expired_sessions()
        session = get_session_state(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Session not found or expired"}
            )
        if session.status != "active":
            raise HTTPException(
                status_code=409,
                detail={"status": "error", "message": "Session is not active. Start a new session to continue."}
            )
        previous_score_snapshot = session.risk_history[-1] if session.risk_history else None
        next_chunk_index = session.chunks_processed + 1

    provisional_scored = build_risk_update(
        analysis_result.features or {},
        analysis_result.classification,
        analysis_result.confidence_score,
        language_result,
        previous_score_snapshot,
    )

    llm_semantic: Optional[Dict[str, Any]] = None
    llm_invoked = should_invoke_llm_semantic(
        provisional_scored=provisional_scored,
        transcript=raw_transcript,
        transcript_confidence=float(language_result.get("transcript_confidence", 0.0)),
        next_chunk_index=next_chunk_index,
    )
    if llm_invoked:
        llm_semantic = await asyncio.to_thread(
            analyze_semantic_with_llm,
            raw_transcript,
            chunk_language,
            settings.LLM_SEMANTIC_TIMEOUT_MS,
        )

    async with SESSION_LOCK:
        purge_expired_sessions()
        session = get_session_state(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Session not found or expired"}
            )
        if session.status != "active":
            raise HTTPException(
                status_code=409,
                detail={"status": "error", "message": "Session is not active. Start a new session to continue."}
            )

        if llm_invoked:
            session.llm_checks_performed += 1
            if llm_semantic and llm_semantic.get("available"):
                session.llm_last_engine = str(llm_semantic.get("engine", "openai-chat-completions"))
            else:
                reason = str((llm_semantic or {}).get("reason", "unavailable"))
                session.llm_last_engine = f"skipped:{reason}"

        behaviour_snapshot = update_session_behaviour_state(session, language_result)
        language_result.update(behaviour_snapshot)
        previous_score = session.risk_history[-1] if session.risk_history else None
        scored = build_risk_update(
            analysis_result.features or {},
            analysis_result.classification,
            analysis_result.confidence_score,
            language_result,
            previous_score,
            llm_semantic=llm_semantic,
        )

        voice_classification = normalize_voice_classification(
            analysis_result.classification,
            scored["model_uncertain"],
        )
        voice_confidence = float(max(0.0, min(1.0, analysis_result.confidence_score)))

        session.chunks_processed += 1
        session.last_update = utc_now_iso()
        session.risk_history.append(scored["risk_score"])
        if scored["risk_score"] >= session.max_risk_score:
            session.final_call_label = scored["call_label"]
        session.max_risk_score = max(session.max_risk_score, scored["risk_score"])
        session.max_cpi = max(session.max_cpi, float(scored["cpi"]))

        if voice_classification == "AI_GENERATED":
            session.voice_ai_chunks += 1
            session.max_voice_ai_confidence = max(session.max_voice_ai_confidence, voice_confidence)
        elif voice_classification == "HUMAN":
            session.voice_human_chunks += 1

        # Use majority vote for session-level classification instead of last-chunk
        if session.voice_ai_chunks > session.voice_human_chunks:
            session.final_voice_classification = "AI_GENERATED"
            session.final_voice_confidence = session.max_voice_ai_confidence
        elif session.voice_human_chunks > session.voice_ai_chunks:
            session.final_voice_classification = "HUMAN"
            session.final_voice_confidence = voice_confidence
        else:
            # Tie — use the latest chunk's decision
            session.final_voice_classification = voice_classification
            session.final_voice_confidence = voice_confidence

        if scored["alert"].triggered:
            alert_obj = scored["alert"]
            alert_entry = {
                "timestamp": session.last_update,
                "risk_score": scored["risk_score"],
                "risk_level": scored["risk_level"],
                "call_label": scored["call_label"],
                "alert_type": alert_obj.alert_type or "FRAUD_RISK_HIGH",
                "severity": alert_obj.severity or scored["risk_level"].lower(),
                "reason_summary": alert_obj.reason_summary or "Fraud indicators detected.",
                "recommended_action": alert_obj.recommended_action
                or recommendation_for_level(scored["risk_level"], scored["model_uncertain"]),
            }

            last_alert = session.alert_history[-1] if session.alert_history else None
            duplicate_keys = ("alert_type", "severity", "reason_summary", "recommended_action", "call_label", "risk_level")
            is_duplicate = bool(
                last_alert
                and all(last_alert.get(key) == alert_entry.get(key) for key in duplicate_keys)
            )

            if is_duplicate:
                last_alert["timestamp"] = session.last_update
                last_alert["risk_score"] = max(int(last_alert.get("risk_score", 0)), scored["risk_score"])
            else:
                session.alerts_triggered += 1
                session.alert_history.append(alert_entry)
                if len(session.alert_history) > 100:
                    session.alert_history = session.alert_history[-100:]

        save_session_state(session)

        return RealTimeUpdateResponse(
            status="success",
            session_id=session_id,
            timestamp=session.last_update,
            risk_score=scored["risk_score"],
            cpi=scored["cpi"],
            risk_level=scored["risk_level"],
            call_label=scored["call_label"],
            model_uncertain=scored["model_uncertain"],
            voice_classification=voice_classification,
            voice_confidence=voice_confidence,
            evidence=scored["evidence"],
            language_analysis=scored["language_analysis"],
            alert=scored["alert"],
            explainability=scored["explainability"],
            chunks_processed=session.chunks_processed,
            risk_policy_version=settings.RISK_POLICY_VERSION,
        )


def session_to_summary(session: SessionState) -> SessionSummaryResponse:
    """Convert session state to response model."""
    return SessionSummaryResponse(
        status="success",
        session_id=session.session_id,
        language=session.language,
        session_status=session.status,
        started_at=session.started_at,
        last_update=session.last_update,
        chunks_processed=session.chunks_processed,
        alerts_triggered=session.alerts_triggered,
        max_risk_score=session.max_risk_score,
        max_cpi=round(session.max_cpi, 1),
        final_call_label=session.final_call_label,
        final_voice_classification=session.final_voice_classification,
        final_voice_confidence=round(session.final_voice_confidence, 2),
        max_voice_ai_confidence=round(session.max_voice_ai_confidence, 2),
        voice_ai_chunks=session.voice_ai_chunks,
        voice_human_chunks=session.voice_human_chunks,
        llm_checks_performed=session.llm_checks_performed,
        risk_policy_version=settings.RISK_POLICY_VERSION,
    )


# Authentication
from fastapi.security import APIKeyHeader
from fastapi import Security

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)  # Changed to False for better error messages

async def verify_api_key(x_api_key: str = Security(api_key_header)) -> str:
    """Dependency to verify API key. Raises 401 if invalid or missing."""
    if x_api_key is None:
        logger.warning("API request without x-api-key header")
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Missing API key. Include 'x-api-key' header."}
        )
    if x_api_key != settings.API_KEY:
        logger.warning(f"API request with invalid key: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "Invalid API key"}
        )
    return x_api_key


def verify_websocket_api_key(websocket: WebSocket) -> bool:
    """Validate API key for websocket connections."""
    key = websocket.headers.get("x-api-key") or websocket.query_params.get("api_key")
    return key == settings.API_KEY


# Routes
@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check for monitoring - verifies ML model is loaded."""
    try:
        from model import _model
        model_loaded = _model is not None
    except Exception:
        model_loaded = False
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "session_store_backend": SESSION_STORE_BACKEND_ACTIVE,
    }


@app.post("/api/voice-detection/v1/session/start", response_model=SessionStartResponse)
async def start_realtime_session(
    session_request: SessionStartRequest,
    api_key: str = Depends(verify_api_key)
):
    """Create a new real-time fraud analysis session."""
    validate_supported_language(session_request.language)

    session_id = str(uuid.uuid4())
    started_at = utc_now_iso()

    async with SESSION_LOCK:
        purged = purge_expired_sessions()
        if purged:
            logger.info("Retention purge removed %s expired sessions", purged)

        session_state = SessionState(
            session_id=session_id,
            language=session_request.language,
            started_at=started_at
        )
        save_session_state(session_state)

    return SessionStartResponse(
        status="success",
        session_id=session_id,
        language=session_request.language,
        started_at=started_at,
        message="Session created. Send chunks using /api/voice-detection/v1/session/{session_id}/chunk or websocket stream."
    )


@app.post("/api/voice-detection/v1/session/{session_id}/chunk", response_model=RealTimeUpdateResponse)
async def analyze_realtime_chunk(
    request: Request,
    session_id: str,
    chunk_request: SessionChunkRequest,
    api_key: str = Depends(verify_api_key)
):
    """Analyze one chunk for an active real-time session."""
    request_id = getattr(request.state, "request_id", f"sess-{session_id[:8]}")

    async with SESSION_LOCK:
        purge_expired_sessions()
        session = get_session_state(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Session not found or expired"}
            )
        if session.status != "active":
            raise HTTPException(
                status_code=409,
                detail={"status": "error", "message": "Session is not active. Start a new session to continue."}
            )
        session_language = session.language

    try:
        return await process_audio_chunk(session_id, chunk_request, session_language, request_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(e)}) from e


@app.websocket("/api/voice-detection/v1/session/{session_id}/stream")
async def stream_realtime_session(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for continuous chunk-based analysis."""
    if not verify_websocket_api_key(websocket):
        await websocket.close(code=1008, reason="Invalid API key")
        return

    async with SESSION_LOCK:
        purge_expired_sessions()
        session = get_session_state(session_id)
        if session is None:
            await websocket.close(code=1008, reason="Session not found or expired")
            return
        if session.status != "active":
            await websocket.close(code=1008, reason="Session is not active")
            return
        session_language = session.language

    await websocket.accept()
    request_id = f"ws-{session_id[:8]}"

    try:
        while True:
            payload = await websocket.receive_json()
            try:
                chunk_request = SessionChunkRequest.model_validate(payload)
            except ValidationError as e:
                await websocket.send_json({
                    "status": "error",
                    "message": "Invalid chunk payload",
                    "details": e.errors()
                })
                continue

            try:
                update = await process_audio_chunk(session_id, chunk_request, session_language, request_id)
                await websocket.send_json(update.model_dump())
            except HTTPException as e:
                detail = e.detail if isinstance(e.detail, dict) else {"status": "error", "message": str(e.detail)}
                await websocket.send_json(detail)
            except ValueError as e:
                await websocket.send_json({"status": "error", "message": str(e)})
    except WebSocketDisconnect:
        logger.info(f"[{request_id}] WebSocket disconnected")


@app.get("/api/voice-detection/v1/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Return current summary for a real-time session."""
    async with SESSION_LOCK:
        purge_expired_sessions()
        session = get_session_state(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Session not found or expired"}
            )
        return session_to_summary(session)


@app.get("/api/voice-detection/v1/session/{session_id}/alerts", response_model=AlertHistoryResponse)
async def get_session_alerts(
    session_id: str,
    limit: int = 20,
    api_key: str = Depends(verify_api_key),
):
    """Return recent alert history for a real-time session."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "limit must be between 1 and 100"},
        )

    async with SESSION_LOCK:
        purge_expired_sessions()
        session = get_session_state(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Session not found or expired"},
            )

        alerts = [AlertHistoryItem(**item) for item in session.alert_history[-limit:]]
        return AlertHistoryResponse(
            status="success",
            session_id=session_id,
            total_alerts=len(session.alert_history),
            alerts=alerts,
        )


@app.get("/api/voice-detection/v1/privacy/retention-policy", response_model=RetentionPolicyResponse)
async def get_retention_policy(api_key: str = Depends(verify_api_key)):
    """Return explicit privacy defaults for raw audio and session-derived data."""
    return RetentionPolicyResponse(
        status="success",
        raw_audio_storage="not_persisted",
        active_session_retention_seconds=settings.SESSION_ACTIVE_RETENTION_SECONDS,
        ended_session_retention_seconds=settings.SESSION_ENDED_RETENTION_SECONDS,
        stored_derived_fields=STORED_DERIVED_FIELDS,
    )


@app.post("/api/voice-detection/v1/session/{session_id}/end", response_model=SessionSummaryResponse)
async def end_realtime_session(
    session_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Mark a session as ended and return final summary."""
    async with SESSION_LOCK:
        purge_expired_sessions()
        session = get_session_state(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "Session not found or expired"}
            )
        session.status = "ended"
        session.last_update = utc_now_iso()
        save_session_state(session)
        return session_to_summary(session)


@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)
@limiter.limit("1000/minute")  # Rate limit: 1000 requests per minute per IP
async def detect_voice(
    request: Request,  # Required for rate limiter
    voice_request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)  # Use dependency injection
):
    """
    Returns classification result with confidence score and explanation.
    """
    # Log request info for debugging
    request_id = getattr(request.state, 'request_id', 'unknown')
    audio_size_kb = len(voice_request.audioBase64) * 3 / 4 / 1024  # Approximate decoded size
    logger.info(f"[{request_id}] Voice detection request: language={voice_request.language}, format={voice_request.audioFormat}, size~{audio_size_kb:.1f}KB")
    
    validate_supported_language(voice_request.language)
    validate_supported_format(voice_request.audioFormat)
    
    try:
        # Step 1: Decode Base64 (async - runs in thread pool)
        logger.info(f"[{request_id}]   -> Decoding Base64...")
        decode_start = time.perf_counter()
        audio_bytes = await asyncio.to_thread(decode_base64_audio, voice_request.audioBase64)
        decode_time = (time.perf_counter() - decode_start) * 1000
        
        # Step 2: Load audio (async - runs in thread pool)
        logger.info(f"[{request_id}]   -> Loading audio... (decode took {decode_time:.0f}ms)")
        load_start = time.perf_counter()
        audio, sr = await asyncio.to_thread(load_audio_from_bytes, audio_bytes, 22050, voice_request.audioFormat)
        load_time = (time.perf_counter() - load_start) * 1000
        
        # Step 3: ML Analysis (async - runs in thread pool, CPU-bound)
        duration_sec = len(audio) / sr
        logger.info(f"[{request_id}]   -> Analyzing {duration_sec:.1f}s audio... (load took {load_time:.0f}ms)")
        analyze_start = time.perf_counter()
        result = await asyncio.to_thread(analyze_voice, audio, sr, voice_request.language)
        analyze_time = (time.perf_counter() - analyze_start) * 1000
        
        logger.info(f"[{request_id}]   -> Analysis complete: {result.classification} ({result.confidence_score:.0%}) in {analyze_time:.0f}ms")
        
        # Extract metrics if available
        metrics = None
        if result.features:
            metrics = ForensicMetrics(
                authenticity_score=result.features.get("authenticity_score", 0),
                pitch_naturalness=result.features.get("pitch_naturalness", 0),
                spectral_naturalness=result.features.get("spectral_naturalness", 0),
                temporal_naturalness=result.features.get("temporal_naturalness", 0)
            )

        model_uncertain = bool((result.features or {}).get("ml_fallback", 0.0))
        explanation = result.explanation
        recommended_action = None
        response_classification = result.classification
        if model_uncertain:
            explanation = (
                "Model uncertainty detected due fallback inference. "
                "Treat result as cautionary and verify through trusted channels. "
                f"{result.explanation}"
            )
            recommended_action = (
                "Do not share OTP, PIN, passwords, or payment credentials. "
                "Verify caller identity through official support channels."
            )
            if settings.LEGACY_FALLBACK_RETURNS_UNCERTAIN:
                response_classification = "UNCERTAIN"

        # Return response
        return VoiceDetectionResponse(
            status="success",
            language=voice_request.language,
            classification=response_classification,
            confidenceScore=result.confidence_score,
            explanation=explanation,
            forensic_metrics=metrics,
            modelUncertain=model_uncertain,
            recommendedAction=recommended_action,
        )
        
    except ValueError as e:
        logger.warning(f"[{request_id}]   [VALIDATION_ERROR] {e}")
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": str(e)}
        )
    except Exception as e:
        logger.error(f"[{request_id}]   [PROCESSING_ERROR] {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Internal Server Error (request_id={request_id})"}
        )


# Exception handlers
from fastapi.exceptions import RequestValidationError

def to_json_safe(value: Any) -> Any:
    """Recursively convert values to JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    return str(value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for 422 Validation Errors.
    Provides clearer error messages for common issues.
    """
    errors = to_json_safe(exc.errors())
    logger.warning("Validation error: %s", errors)
    
    # Build user-friendly error message
    error_messages = []
    for error in errors:
        loc = " -> ".join(str(l) for l in error.get("loc", []))
        msg = error.get("msg", "Invalid value")
        error_messages.append(f"{loc}: {msg}")
    
    # Common issue detection
    if any("audioBase64" in str(e.get("loc", [])) for e in errors):
        hint = " Hint: Ensure 'audioBase64' is a valid Base64-encoded string."
    elif any("language" in str(e.get("loc", [])) for e in errors):
        hint = f" Hint: 'language' must be one of: {', '.join(settings.SUPPORTED_LANGUAGES)}."
    else:
        hint = ""
    
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": f"Request validation failed: {'; '.join(error_messages)}.{hint}",
            "details": errors
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom exception handler to ensure consistent error format."""
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global handler to catch unhandled exceptions and prevent stack traces."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal Server Error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)




















