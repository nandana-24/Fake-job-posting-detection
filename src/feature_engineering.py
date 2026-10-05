"""
feature_engineering.py
Domain feature extraction and preprocessing utilities for Job Scam Detection.
Extracts structural, heuristic, and behavioral signals from job posting text.
"""

import re
import pandas as pd
import numpy as np

FREE_EMAIL_REGEX = r"@(?:gmail|yahoo|hotmail|outlook|proton|protonmail|aol|rediffmail|zoho|mail|yandex|gmx)\.com"
MESSAGING_REGEX = r"\b(?:whatsapp|telegram|wa\.me|t\.me)\b|contact\s+(?:the\s+)?(?:[\w\s]{0,20})?(?:on|through|via)\s+(?:whatsapp|telegram)"
FEE_REGEX = (
    r"(?:refundable\s+(?:(?:₹|rs\.?|inr|\$)?\s?\d+[\w\s]{0,25})?(?:fee|deposit|charges?))|"
    r"(?:registration|application|training|security|processing|verification|portal|account|joining)\s+(?:(?:and|&)\s+\w+\s+)?(?:fee|charges?|deposit)|"
    r"(?:pay|complete|deposit)\s+(?:a\s+)?(?:refundable\s+)?(?:(?:₹|rs\.?|inr|\$)?\s?\d+[\w\s]{0,25})?(?:fee|deposit|charges?)|"
    r"pay\s+(?:₹|rs\.?|inr|\$)\s?\d+|(?:wallet\s+)?deposit\s+(?:of|required|to)|deposit\s+(?:₹|rs\.?|inr|\$)\s?\d+|prepaid\s+tasks?"
)
URGENCY_REGEX = r"\b(?:urgent|urgently|immediate|immediately|hurry|limited\s+(?:seats|vacancies|slots)|only\s+today|apply\s+now|within\s+\d+\s*hours?)\b"
GUARANTEE_REGEX = r"\b(?:guaranteed\s+(?:job|income|placement|salary)|100%\s+placement|direct\s+(?:selection|joining)|no\s+(?:technical\s+)?interview)\b"

ENGINEERED_FEATURE_COLS = [
    "has_free_email",
    "has_messaging_app",
    "has_fee_request",
    "urgency_count",
    "caps_ratio",
    "exclamation_count",
    "word_count",
    "has_guarantee_claim",
]


def extract_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts numerical and binary domain fraud signals from text fields in the dataframe.
    Returns a DataFrame with columns defined in ENGINEERED_FEATURE_COLS.
    """
    text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits', 'application_process']

    combined_text = df[text_fields[0]].fillna('').astype(str) if text_fields[0] in df.columns else pd.Series('', index=df.index)
    for col in text_fields[1:]:
        if col in df.columns:
            combined_text = combined_text + " " + df[col].fillna('').astype(str)

    features = pd.DataFrame(index=df.index)

    # 1. Free consumer email detection
    features["has_free_email"] = combined_text.str.contains(
        FREE_EMAIL_REGEX, case=False, regex=True
    ).astype(int)

    # 2. WhatsApp / Telegram recruitment
    features["has_messaging_app"] = combined_text.str.contains(
        MESSAGING_REGEX, case=False, regex=True
    ).astype(int)

    # 3. Upfront fee or deposit demand
    features["has_fee_request"] = combined_text.str.contains(
        FEE_REGEX, case=False, regex=True
    ).astype(int)

    # 4. Urgency count
    features["urgency_count"] = combined_text.apply(
        lambda t: len(re.findall(URGENCY_REGEX, t, flags=re.IGNORECASE))
    )

    # 5. ALL CAPS ratio (uppercase letters / total letters)
    def calc_caps_ratio(text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        caps = [c for c in letters if c.isupper()]
        return round(len(caps) / len(letters), 4)

    features["caps_ratio"] = combined_text.apply(calc_caps_ratio)

    # 6. Exclamation mark count
    features["exclamation_count"] = combined_text.apply(
        lambda t: t.count("!")
    )

    # 7. Word count
    features["word_count"] = combined_text.apply(
        lambda t: len([w for w in t.split() if len(w) > 1])
    )

    # 8. Unrealistic guarantee claims
    features["has_guarantee_claim"] = combined_text.str.contains(
        GUARANTEE_REGEX, case=False, regex=True
    ).astype(int)

    return features
