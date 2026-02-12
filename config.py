"""
Configuration management using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from typing import List
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration."""

    # Core API Settings
    API_KEY: str = Field(..., description="API Key for authentication")
    PORT: int = Field(7860, description="Server port")
    WEBSITE_URL: str = Field(
        default="https://voice-detection-nu.vercel.app/",
        description="Project or Portfolio URL"
    )

    # CORS Settings
    # Use str field with alias to read env var safely (avoids Pydantic trying to parse as JSON)
    ALLOWED_ORIGINS_RAW: str = Field(default="*", alias="ALLOWED_ORIGINS")

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Parse the raw CORS origins string into a list."""
        raw_value: str = self.ALLOWED_ORIGINS_RAW
        if raw_value.strip().startswith("["):
            import json
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                pass
        return [origin.strip() for origin in raw_value.split(",") if origin.strip()]

    # Audio Constraints
    MAX_AUDIO_SIZE_MB: int = 10
    SUPPORTED_LANGUAGES: List[str] = [
        "Tamil", "English", "Hindi", "Malayalam", "Telugu"
    ]
    SUPPORTED_FORMATS: List[str] = [
        "mp3", "wav", "flac", "ogg", "m4a", "mp4"
    ]

    # ASR settings
    ASR_ENABLED: bool = Field(default=True, description="Enable speech-to-text analysis for realtime sessions")
    ASR_MODEL_SIZE: str = Field(default="tiny", description="faster-whisper model size")
    ASR_COMPUTE_TYPE: str = Field(default="int8", description="faster-whisper compute type")
    ASR_BEAM_SIZE: int = Field(default=1, description="Beam size for ASR decoding")
    ASR_TIMEOUT_MS: int = Field(
        default=1200,
        ge=200,
        le=15000,
        description="Max realtime ASR duration per chunk before timeout fallback"
    )
    ASR_MAX_INFLIGHT_TASKS: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Maximum concurrent ASR background tasks allowed to prevent thread pileups"
    )
    ASR_WARMUP_ENABLED: bool = Field(
        default=True,
        description="Warm faster-whisper model during startup to avoid first-chunk latency spike"
    )
    AUDIO_PIPELINE_WARMUP_ENABLED: bool = Field(
        default=True,
        description="Warm audio decoding/resampling pipeline during startup"
    )
    VOICE_WARMUP_ENABLED: bool = Field(
        default=True,
        description="Run one startup inference through voice analyzer to avoid first-chunk latency spikes"
    )

    # Voice classification model settings
    VOICE_MODEL_ID: str = Field(
        default="shivam-2211/voice-detection-model",
        description="Primary Hugging Face model id for AI voice detection"
    )
    VOICE_MODEL_BACKUP_ID: str = Field(
        default="mo-thecreator/Deepfake-audio-detection",
        description="Backup model id if primary model load fails"
    )
    VOICE_MODEL_LOCAL_PATH: str = Field(
        default="./fine_tuned_model",
        description="Optional local model path that takes priority when present"
    )
    REALTIME_LIGHTWEIGHT_AUDIO: bool = Field(
        default=True,
        description="Use lightweight audio analysis path for realtime chunk processing (set true for throughput-first mode)"
    )
    LEGACY_FALLBACK_RETURNS_UNCERTAIN: bool = Field(
        default=True,
        description="Return UNCERTAIN classification on legacy endpoint when ML fallback occurs"
    )

    # Risk policy (versioned + configurable weights)
    RISK_POLICY_VERSION: str = Field(default="v1.2", description="Version tag for realtime risk policy")
    RISK_WEIGHT_AUDIO: float = Field(default=0.45, ge=0.0, le=1.0)
    RISK_WEIGHT_KEYWORD: float = Field(default=0.20, ge=0.0, le=1.0)
    RISK_WEIGHT_SEMANTIC: float = Field(default=0.15, ge=0.0, le=1.0)
    RISK_WEIGHT_BEHAVIOUR: float = Field(default=0.20, ge=0.0, le=1.0)
    RISK_DELTA_BOOST_FACTOR: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="How strongly risk increases when per-chunk delta is positive"
    )

    # Optional LLM semantic verifier (second-layer, not primary classifier)
    LLM_SEMANTIC_ENABLED: bool = Field(default=False)
    LLM_PROVIDER: str = Field(default="gemini", description="LLM provider: openai or gemini")
    LLM_SEMANTIC_MODEL: str = Field(default="gemini-2.5-flash", description="Model name for selected LLM provider (optional)")
    LLM_SEMANTIC_TIMEOUT_MS: int = Field(default=900, ge=100, le=5000)
    LLM_SEMANTIC_MIN_ASR_CONFIDENCE: float = Field(default=0.35, ge=0.0, le=1.0)
    LLM_SEMANTIC_CHUNK_INTERVAL: int = Field(default=2, ge=1, le=20)
    LLM_SEMANTIC_BLEND_WEIGHT: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight assigned to LLM semantic score in fused semantic score"
    )
    OPENAI_API_KEY: str | None = Field(default=None, description="Optional OpenAI API key for LLM semantic verifier")
    GEMINI_API_KEY: str | None = Field(default=None, description="Optional Gemini API key for LLM semantic verifier")

    # Session store backend
    SESSION_STORE_BACKEND: str = Field(
        default="redis",
        description="Session store backend: memory or redis"
    )
    REDIS_URL: str | None = Field(
        default=None,
        description="Redis URL for session state and queue (required when SESSION_STORE_BACKEND=redis)"
    )
    REDIS_PREFIX: str = Field(
        default="ai_call_shield",
        description="Redis key prefix namespace"
    )
    REDIS_CONNECT_TIMEOUT_MS: int = Field(default=2000, ge=100, le=30000)
    REDIS_IO_TIMEOUT_MS: int = Field(default=2000, ge=100, le=30000)

    # Deep-lane async verification controls
    DEEP_LANE_ENABLED: bool = Field(
        default=False,
        description="Enable asynchronous deep-lane verification after fast-lane decision"
    )
    DEEP_LANE_QUEUE_BACKEND: str = Field(
        default="memory",
        description="Queue backend: memory or redis"
    )
    DEEP_LANE_MAX_WORKERS: int = Field(default=2, ge=1, le=16)
    DEEP_LANE_MAX_RETRIES: int = Field(default=1, ge=0, le=10)
    DEEP_LANE_RETRY_BACKOFF_MS: int = Field(default=500, ge=0, le=60000)
    DEEP_LANE_TARGET_LATENCY_MS: int = Field(default=3000, ge=200, le=10000)

    # Performance targets (for harness/reporting and CI gates)
    PERF_CHUNK_P95_TARGET_MS: int = Field(default=1200, ge=100, le=10000)
    PERF_ALERT_P95_TARGET_MS: int = Field(default=2500, ge=100, le=10000)

    # Session retention and privacy controls
    SESSION_ACTIVE_RETENTION_SECONDS: int = Field(
        default=1800,
        description="Retention TTL for active sessions with no updates"
    )
    SESSION_ENDED_RETENTION_SECONDS: int = Field(
        default=300,
        description="Retention TTL for ended sessions before purge"
    )
    MASK_TRANSCRIPT_OUTPUT: bool = Field(
        default=True,
        description="Mask sensitive entities from transcript before returning response"
    )

    # Environment Specific
    SPACE_ID: str | None = Field(default=None, description="Hugging Face Space ID if running in Spaces")

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings()
