"""
predictor.py
Loads trained artifacts and provides pre-processing and fraud prediction.
Uses the Calibrated Ensemble Model (Logistic Regression + Random Forest with Feature Fusion)
for maximum precision and recall, and maintains Logistic Regression for SHAP explainability.
"""

import joblib
import pandas as pd
from pathlib import Path
from feature_engineering import extract_domain_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = Path("../models")

TEXT_COLS = ['title', 'company_profile', 'description', 'requirements', 'benefits']
CATE_COLS = ['employment_type', 'required_experience', 'required_education',
             'industry', 'function', 'country']
BINARY_COLS = ['telecommuting', 'has_company_logo', 'has_questions',
               'salary_missing', 'location_missing']

# Common country aliases mapped to ISO 3166-1 alpha-2 codes used in EMSCAD
COUNTRY_ALIASES = {
    "india": "IN", "in": "IN",
    "united states": "US", "usa": "US", "us": "US",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "gb": "GB",
    "canada": "CA", "ca": "CA",
    "australia": "AU", "au": "AU",
    "germany": "DE", "de": "DE",
    "france": "FR", "fr": "FR",
    "greece": "GR", "gr": "GR",
    "malaysia": "MY", "my": "MY",
    "philippines": "PH", "ph": "PH",
    "poland": "PL", "pl": "PL",
    "singapore": "SG", "sg": "SG",
    "united arab emirates": "AE", "uae": "AE", "ae": "AE",
    "netherlands": "NL", "nl": "NL",
    "new zealand": "NZ", "nz": "NZ",
    "ireland": "IE", "ie": "IE",
}

_preprocessor = None
_model = None
_lr_model = None
_feature_names = None


def load_artifacts():
    """Load the preprocessor, ensemble model, logistic regression, and feature names."""
    global _preprocessor, _model, _lr_model, _feature_names
    if _preprocessor is None:
        _preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    
    # Load calibrated ensemble model (falls back to logistic regression if needed)
    if _model is None:
        ensemble_path = MODELS_DIR / "ensemble_model.joblib"
        if ensemble_path.exists():
            _model = joblib.load(ensemble_path)
        else:
            _model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
            
    if _lr_model is None:
        _lr_model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
        
    if _feature_names is None:
        _feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
        
    return _preprocessor, _model, _feature_names


def load_lr_model():
    """Load the Logistic Regression model specifically for SHAP explanations."""
    global _lr_model
    if _lr_model is None:
        _lr_model = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    return _lr_model


def build_input_row(user_input: dict) -> pd.DataFrame:
    """
    Convert a dict of USER-PROVIDED fields into a single-row DataFrame
    matching the exact schema the preprocessor was fit on.

    Handles default values cleanly:
      - "Select", empty, or unspecified dropdowns map to "Unknown"
      - Country aliases (e.g. "India" -> "IN") are normalized to EMSCAD ISO codes
      - Binary flags default to 0, missing indicator flags computed accurately
      - Domain engineered features (free email, messaging app, fee demand, caps ratio, etc.)
    """
    row = {}

    for col in TEXT_COLS:
        row[col] = str(user_input.get(col, "") or "")

    app_process = str(user_input.get('application_process', '') or '').strip()
    if app_process:
        desc = row.get('description', '')
        row['description'] = f"{desc}\n\nApplication Process:\n{app_process}" if desc else app_process
    row['application_process'] = app_process

    for col in CATE_COLS:
        val = user_input.get(col)
        if val in (None, "", "Select", "None", "Unspecified", "Not Specified", "Not Applicable / None"):
            val = "Unknown"
        row[col] = str(val).strip()

    # Normalize country to 2-letter ISO code if possible
    country_raw = str(user_input.get('country', '') or '').strip().lower()
    if country_raw in COUNTRY_ALIASES:
        row['country'] = COUNTRY_ALIASES[country_raw]
    elif len(country_raw) == 2 and country_raw.isalpha():
        row['country'] = country_raw.upper()
    elif country_raw in ("", "select", "unknown", "remote", "not specified", "none"):
        row['country'] = "Unknown"

    row['telecommuting'] = int(user_input.get('telecommuting', 0))
    row['has_company_logo'] = int(user_input.get('has_company_logo', 0))
    row['has_questions'] = int(user_input.get('has_questions', 0))

    salary = user_input.get('salary')
    row['salary_missing'] = 0 if salary not in (None, "", 0, "Select") else 1

    country_val = row['country']
    row['location_missing'] = 0 if country_val not in (None, "", "Unknown") else 1

    df_row = pd.DataFrame([row])

    # Extract engineered domain features
    domain_feats = extract_domain_features(df_row)
    df_row = pd.concat([df_row, domain_feats], axis=1)

    return df_row


def predict_posting(user_input: dict):
    """
    Full prediction pipeline for ONE new job posting using the Calibrated Ensemble Model.
    Returns: (prediction_label, fraud_probability, X_transformed, input_row)

    Applies calibrated 3-tier risk categories:
      - Low Risk (< 20%): Standard legitimate indicators
      - Moderate Risk (20% - 50%): Inconclusive or elevated baseline risk
      - High Risk (>= 50%): Strong fraud indicators
    """
    preprocessor, model, _ = load_artifacts()

    input_row = build_input_row(user_input)

    # IMPORTANT: transform() only -- never fit_transform() at inference time
    X_transformed = preprocessor.transform(input_row)

    proba = float(model.predict_proba(X_transformed)[0, 1])

    if proba < 0.20:
        label = "Low Scam Risk (Likely Legitimate)"
    elif proba < 0.50:
        label = "Moderate Risk / Inconclusive"
    else:
        label = "High Scam Risk (Potentially Fraudulent)"

    return label, proba, X_transformed, input_row
