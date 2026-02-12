"""
Keyword and semantic fraud signal extraction from transcripts.
"""
from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Set

# Baseline keywords that are language-agnostic or commonly spoken in English/Hinglish.
COMMON_FRAUD_KEYWORDS: Dict[str, Set[str]] = {
    "financial": {
        "bank account", "account", "credit card", "debit card", "loan", "khata",
    },
    "payment": {
        "upi", "upi id", "gpay", "google pay", "phonepe", "paytm", "neft", "rtgs",
        "send money", "transfer money", "payment",
    },
    "authentication": {
        "otp", "pin", "password", "cvv", "verification code", "passcode",
    },
    "urgency": {
        "urgent", "immediately", "right now", "now", "last chance", "today only",
        "abhi", "turant", "jaldi",
    },
    "threat": {
        "blocked", "suspended", "legal action", "police", "arrest", "freeze",
    },
    "impersonation": {
        "rbi", "bank manager", "government", "income tax", "customs", "official",
    },
    "offer_lure": {
        "lottery", "prize", "winner", "cashback", "free", "reward",
    },
}

# Language-specific script and phrase variants to improve 5-language support.
LANGUAGE_FRAUD_KEYWORDS: Dict[str, Dict[str, Set[str]]] = {
    "Hindi": {
        "financial": {"बैंक", "खाता", "अकाउंट", "लोन"},
        "payment": {"यूपीआई", "युपीआई", "भुगतान", "पैसे भेजो", "ट्रांसफर", "गूगल पे", "फोनपे", "पेटीएम"},
        "authentication": {"ओटीपी", "पिन", "पासवर्ड", "सत्यापन कोड"},
        "urgency": {"अभी", "तुरंत", "जल्दी", "फौरन", "अंतिम मौका"},
        "threat": {"ब्लॉक", "निलंबित", "कानूनी कार्रवाई", "गिरफ्तार", "फ्रीज"},
        "impersonation": {"आरबीआई", "सरकारी अधिकारी", "बैंक मैनेजर", "इनकम टैक्स"},
        "offer_lure": {"लॉटरी", "इनाम", "कैशबैक", "फ्री", "रिवॉर्ड"},
    },
    "Tamil": {
        "financial": {"வங்கி", "கணக்கு", "அக்கவுண்ட்", "கடன்"},
        "payment": {"யுபிஐ", "கூகுள் பே", "போன்பே", "பேடிஎம்", "பணம் அனுப்பு", "பணம் பரிமாற்றம்", "கட்டணம்"},
        "authentication": {"ஓடிபி", "பின்", "கடவுச்சொல்", "சரிபார்ப்பு குறியீடு"},
        "urgency": {"உடனே", "இப்போதே", "விரைவாக", "இப்போது", "அவசரம்"},
        "threat": {"முடக்கப்படும்", "தடைசெய்யப்படும்", "சட்ட நடவடிக்கை", "காவல்", "உறையவைக்கப்படும்"},
        "impersonation": {"ஆர்பிஐ", "அரசு அதிகாரி", "வங்கி மேலாளர்", "வருமானவரி"},
        "offer_lure": {"லாட்டரி", "பரிசு", "கேஷ்பேக்", "இலவசம்", "வெற்றி"},
    },
    "Malayalam": {
        "financial": {"ബാങ്ക്", "അക്കൗണ്ട്", "ഖാത", "ലോൺ"},
        "payment": {"യുപിഐ", "ഗൂഗിൾ പേ", "ഫോൺപേ", "പേടിഎം", "പണം അയക്കൂ", "പേയ്മെന്റ്", "ട്രാൻസ്ഫർ"},
        "authentication": {"ഒടിപി", "പിൻ", "പാസ്‌വേഡ്", "സ്ഥിരീകരണ കോഡ്"},
        "urgency": {"ഉടൻ", "ഇപ്പോള്", "തൽക്ഷണം", "വേഗം", "അവസരം"},
        "threat": {"ബ്ലോക്ക്", "സസ്പെൻഡ്", "നിയമ നടപടി", "അറസ്റ്റ്", "ഫ്രീസ്"},
        "impersonation": {"ആർബിഐ", "സർക്കാർ ഓഫീസർ", "ബാങ്ക് മാനേജർ", "ഇൻകം ടാക്സ്"},
        "offer_lure": {"ലോട്ടറി", "സമ്മാനം", "കാഷ്ബാക്ക്", "ഫ്രീ", "റിവാർഡ്"},
    },
    "Telugu": {
        "financial": {"బ్యాంక్", "ఖాతా", "అకౌంట్", "లోన్"},
        "payment": {"యూపీఐ", "గూగుల్ పే", "ఫోన్‌పే", "పేటిఎం", "డబ్బు పంపండి", "చెల్లింపు", "ట్రాన్స్‌ఫర్"},
        "authentication": {"ఓటిపి", "పిన్", "పాస్‌వర్డ్", "ధృవీకరణ కోడ్"},
        "urgency": {"వెంటనే", "ఇప్పుడే", "తక్షణం", "త్వరగా", "చివరి అవకాశం"},
        "threat": {"బ్లాక్", "సస్పెండ్", "చట్టపరమైన చర్య", "అరెస్ట్", "ఫ్రీజ్"},
        "impersonation": {"ఆర్బిఐ", "ప్రభుత్వ అధికారి", "బ్యాంక్ మేనేజర్", "ఇన్కమ్ ట్యాక్స్"},
        "offer_lure": {"లాటరీ", "బహుమతి", "క్యాష్‌బ్యాక్", "ఉచితం", "రివార్డు"},
    },
}

PUNCT_TRANSLATION = str.maketrans({ch: " " for ch in (string.punctuation + "“”‘’…–—।॥،؛")})


def _normalize_text(text: str) -> str:
    """
    Normalize text while preserving non-Latin scripts.

    We avoid ASCII-only regex stripping so Indic scripts remain searchable.
    """
    normalized = text.casefold().translate(PUNCT_TRANSLATION)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _combined_keyword_catalog(language: str | None) -> Dict[str, Set[str]]:
    """Merge common keywords with optional language-specific keywords."""
    merged: Dict[str, Set[str]] = {category: set(values) for category, values in COMMON_FRAUD_KEYWORDS.items()}

    if language and language in LANGUAGE_FRAUD_KEYWORDS:
        language_maps = [LANGUAGE_FRAUD_KEYWORDS[language]]
    else:
        # Fallback: support mixed-language transcripts by checking all known script maps.
        language_maps = list(LANGUAGE_FRAUD_KEYWORDS.values())

    for language_map in language_maps:
        for category, keywords in language_map.items():
            merged.setdefault(category, set()).update(keywords)

    return merged


def _contains_keyword(normalized_text: str, token_set: Set[str], keyword: str) -> bool:
    key = _normalize_text(keyword)
    if not key:
        return False
    if " " in key:
        return key in normalized_text
    return key in token_set


def _match_keywords(normalized_text: str, catalog: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    by_category: Dict[str, List[str]] = {}
    token_set = set(normalized_text.split())

    for category, keywords in catalog.items():
        hits = [kw for kw in keywords if _contains_keyword(normalized_text, token_set, kw)]
        if hits:
            by_category[category] = sorted(hits)
    return by_category


def analyze_transcript(transcript: str, language: str | None = None) -> Dict[str, Any]:
    """Extract keyword and semantic signals from transcript text."""
    if not transcript:
        return {
            "keyword_hits": [],
            "keyword_categories": [],
            "keyword_score": 0,
            "semantic_flags": [],
            "semantic_score": 0,
            "behaviour_signals": [],
        }

    text = _normalize_text(transcript)
    category_hits = _match_keywords(text, _combined_keyword_catalog(language))

    keyword_hits: List[str] = []
    for category, hits in sorted(category_hits.items()):
        keyword_hits.extend([f"{category}:{hit}" for hit in hits])

    categories = sorted(category_hits.keys())
    keyword_score = min(100, len(keyword_hits) * 7 + len(categories) * 12)

    semantic_flags: List[str] = []
    behaviour_signals: List[str] = []

    has_urgency = "urgency" in category_hits
    has_impersonation = "impersonation" in category_hits
    has_auth = "authentication" in category_hits
    has_payment = "payment" in category_hits
    has_threat = "threat" in category_hits

    if has_urgency:
        semantic_flags.append("urgency_language")
        behaviour_signals.append("urgency_escalation")
    if has_impersonation:
        semantic_flags.append("authority_impersonation")
    if has_auth:
        semantic_flags.append("credential_request")
    if has_payment:
        semantic_flags.append("payment_redirection")
    if has_threat:
        semantic_flags.append("coercive_threat_language")
    if "offer_lure" in category_hits:
        semantic_flags.append("incentive_lure")

    semantic_score = min(100, len(semantic_flags) * 14)
    if has_impersonation and has_auth:
        semantic_score = min(100, semantic_score + 18)
        behaviour_signals.append("authority_with_credential_request")
    if has_payment and has_urgency:
        semantic_score = min(100, semantic_score + 14)
        behaviour_signals.append("urgent_payment_pressure")
    if has_threat and has_urgency:
        semantic_score = min(100, semantic_score + 10)
        behaviour_signals.append("threat_plus_urgency")

    return {
        "keyword_hits": keyword_hits,
        "keyword_categories": categories,
        "keyword_score": keyword_score,
        "semantic_flags": semantic_flags,
        "semantic_score": semantic_score,
        "behaviour_signals": sorted(set(behaviour_signals)),
    }
