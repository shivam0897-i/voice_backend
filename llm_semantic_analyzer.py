"""
Optional LLM semantic verifier for realtime transcript analysis.

This is a second-layer signal meant for ambiguous/uncertain chunks.
It must never block realtime flow.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import httpx

from config import settings
from privacy_utils import mask_sensitive_entities

logger = logging.getLogger(__name__)


def _clamp_int(value: Any, lo: int = 0, hi: int = 100) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, parsed))


def _clamp_float(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, parsed))


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _resolve_provider() -> str:
    provider = str(getattr(settings, "LLM_PROVIDER", "openai") or "openai").strip().lower()
    if provider in {"gemini", "google"}:
        return "gemini"
    return "openai"


def _resolve_model(provider: str) -> str:
    configured = str(getattr(settings, "LLM_SEMANTIC_MODEL", "") or "").strip()
    if configured:
        return configured
    if provider == "gemini":
        return "gemini-1.5-flash"
    return "gpt-4o-mini"


def _provider_api_key(provider: str) -> Optional[str]:
    if provider == "gemini":
        return getattr(settings, "GEMINI_API_KEY", None)
    return getattr(settings, "OPENAI_API_KEY", None)


def is_llm_semantic_provider_ready() -> bool:
    """Return True when selected provider has required credentials."""
    provider = _resolve_provider()
    return bool(_provider_api_key(provider))


def _normalized_response(data: Dict[str, Any], model_name: str, engine_name: str) -> Dict[str, Any]:
    semantic_flags = data.get("semantic_flags") or []
    behaviour_signals = data.get("behaviour_signals") or []
    keyword_hints = data.get("keyword_hints") or []

    if not isinstance(semantic_flags, list):
        semantic_flags = []
    if not isinstance(behaviour_signals, list):
        behaviour_signals = []
    if not isinstance(keyword_hints, list):
        keyword_hints = []

    return {
        "available": True,
        "semantic_score": _clamp_int(data.get("semantic_score", 0)),
        "confidence": _clamp_float(data.get("confidence", 0.0)),
        "semantic_flags": [str(x) for x in semantic_flags if x],
        "behaviour_signals": [str(x) for x in behaviour_signals if x],
        "keyword_hints": [str(x) for x in keyword_hints if x],
        "model": model_name,
        "engine": engine_name,
    }


def _build_prompts(language: str, safe_transcript: str) -> tuple[str, str]:
    system_prompt = (
        "You are a telecom fraud intent classifier. "
        "Return ONLY strict JSON with keys: "
        "semantic_score (0-100), confidence (0-1), semantic_flags (string[]), "
        "behaviour_signals (string[]), keyword_hints (string[])."
    )

    user_prompt = (
        f"Language: {language}\n"
        "Task: detect coercion, impersonation, credential request, and payment pressure.\n"
        f"Transcript: {safe_transcript}"
    )
    return system_prompt, user_prompt


def _call_openai_semantic(
    client: httpx.Client,
    model_name: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    response = client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    parsed = _extract_json_object(content)
    if parsed is None:
        return {"available": False, "reason": "invalid_json"}
    return _normalized_response(parsed, model_name=model_name, engine_name="openai-chat-completions")


def _call_gemini_semantic(
    client: httpx.Client,
    model_name: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
        },
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    response = client.post(url, params={"key": api_key}, json=payload)
    response.raise_for_status()
    data = response.json()

    content = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )
    parsed = _extract_json_object(content)
    if parsed is None:
        return {"available": False, "reason": "invalid_json"}
    return _normalized_response(parsed, model_name=model_name, engine_name="gemini-generate-content")


def analyze_semantic_with_llm(transcript: str, language: str, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze transcript semantics via an optional LLM.

    Returns a normalized dict with `available` bool and semantic fields.
    """
    if not settings.LLM_SEMANTIC_ENABLED:
        return {"available": False, "reason": "disabled"}

    if not transcript or len(transcript.strip()) < 8:
        return {"available": False, "reason": "insufficient_transcript"}

    provider = _resolve_provider()
    api_key = _provider_api_key(provider)
    if not api_key:
        return {"available": False, "reason": f"missing_{provider}_api_key"}

    safe_transcript = mask_sensitive_entities(transcript).strip()
    if not safe_transcript:
        return {"available": False, "reason": "empty_after_masking"}

    timeout_seconds = max(0.1, (timeout_ms or settings.LLM_SEMANTIC_TIMEOUT_MS) / 1000.0)
    model_name = _resolve_model(provider)
    system_prompt, user_prompt = _build_prompts(language, safe_transcript)

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            if provider == "openai":
                return _call_openai_semantic(
                    client=client,
                    model_name=model_name,
                    api_key=api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            if provider == "gemini":
                return _call_gemini_semantic(
                    client=client,
                    model_name=model_name,
                    api_key=api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )

        return {"available": False, "reason": "unsupported_provider"}
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        logger.warning("LLM semantic verifier unavailable (%s): %s", provider, exc)
        return {"available": False, "reason": "request_failed"}
