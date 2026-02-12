"""
Privacy helpers for masking sensitive entities in transcripts and logs.
"""
from __future__ import annotations

import re


PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
# UPI IDs: must end with known UPI provider handles (ybl, okaxis, paytm, etc.)
UPI_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._-]{2,}@(?:ybl|okaxis|okhdfcbank|okicici|oksbi|paytm|apl|upi"
    r"|axl|ibl|sbi|icici|hdfcbank|axisbank|kotak|indus|unionbank|boi|pnb|freecharge"
    r"|idfcbank|dbs|rbl|federal|yes|fino|jio|slice|groww|cred|amazonpay|phonepe)\b",
    re.IGNORECASE,
)
ACCOUNT_OR_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){9,19}(?!\d)")
OTP_CONTEXT_PATTERN = re.compile(r"\b(otp|pin)\s*[:\-]?\s*(\d{4,8})\b", re.IGNORECASE)


def _mask_numeric_token(token: str, preserve_tail: int = 2) -> str:
    digits = re.sub(r"\D", "", token)
    if len(digits) <= preserve_tail:
        return "[REDACTED_NUM]"
    return f"[REDACTED_NUM_XX{digits[-preserve_tail:]}]"


def _mask_account_or_card(match: re.Match[str]) -> str:
    token = match.group(0)
    digits = re.sub(r"\D", "", token)
    if len(digits) < 9:
        return token
    return _mask_numeric_token(token)


def _mask_otp(match: re.Match[str]) -> str:
    return f"{match.group(1)} [REDACTED_OTP]"


def mask_sensitive_entities(text: str) -> str:
    """Redact common scam-sensitive entities from plain text."""
    if not text:
        return ""

    masked = OTP_CONTEXT_PATTERN.sub(_mask_otp, text)
    masked = UPI_PATTERN.sub("[REDACTED_UPI]", masked)
    masked = PHONE_PATTERN.sub("[REDACTED_PHONE]", masked)
    masked = ACCOUNT_OR_CARD_PATTERN.sub(_mask_account_or_card, masked)
    return masked


def sanitize_for_logging(text: str, max_chars: int = 120) -> str:
    """
    Mask and compact text for safe structured logging.
    """
    masked = mask_sensitive_entities(text)
    compact = " ".join(masked.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."
