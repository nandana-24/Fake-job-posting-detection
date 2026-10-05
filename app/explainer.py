"""
explainer.py
Generates a SHAP explanation for a SINGLE new job posting, reusing the
Logistic Regression model from the enhanced pipeline and the background dataset.
"""

import joblib
import numpy as np
import shap
from pathlib import Path

from predictor import load_lr_model, TEXT_COLS, CATE_COLS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = Path("../models")

_explainer = None


def load_explainer():
    global _explainer
    if _explainer is None:
        model = load_lr_model()
        background = joblib.load(MODELS_DIR / "shap_background.joblib")
        _explainer = shap.LinearExplainer(model, background)
    return _explainer


def humanize_feature_name(feat_name: str):
    """
    Convert a raw ColumnTransformer feature name into a human-readable
    (field, term) pair.
    """
    if "__" not in feat_name:
        return ("other", feat_name)

    prefix, rest = feat_name.split("__", 1)

    if prefix == "description":
        if rest.startswith("word__"):
            return ("description word", f'"{rest[6:]}"')
        elif rest.startswith("char__"):
            return ("subword pattern", f'"{rest[6:]}"')
        else:
            return ("description", f'"{rest}"')

    if prefix in TEXT_COLS:
        return (prefix, f'"{rest}"')

    if prefix == "categorical":
        for col in CATE_COLS:
            if rest.startswith(col + "_"):
                value = rest[len(col) + 1:]
                return (col, value)
        return ("categorical", rest)

    if prefix == "binary":
        clean_name = rest.replace("has_", "").replace("_", " ").title()
        return ("flag", clean_name)

    if prefix == "numeric_engineered":
        clean_name = rest.replace("_", " ").title()
        return ("signal metric", clean_name)

    return (prefix, rest)


def explain_posting(X_transformed, feature_names, top_n=8, only_present_features=True):
    """
    Returns (explanations, base_value):
      explanations: list of dicts for the top_n most influential features
                    for THIS single prediction (by absolute SHAP value)
      base_value:   the model's expected log-odds output before any
                    feature contributions
    """
    explainer = load_explainer()
    shap_explanation = explainer(X_transformed)

    shap_values = shap_explanation.values[0]
    base_value = float(shap_explanation.base_values[0])

    if only_present_features:
        if hasattr(X_transformed, "tocoo"):
            coo = X_transformed.tocoo()
            present_cols = set(coo.col)
        elif hasattr(X_transformed, "toarray"):
            arr = X_transformed.toarray()[0]
            present_cols = set(np.where(arr != 0)[0])
        else:
            present_cols = set(np.where(np.asarray(X_transformed)[0] != 0)[0])
    else:
        present_cols = set(range(len(shap_values)))

    candidates = [
        (i, abs(shap_values[i]), shap_values[i])
        for i in present_cols
        if shap_values[i] != 0
    ]

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_indices = candidates[:top_n]

    explanations = []
    for idx, abs_val, raw_val in top_indices:
        field, term = humanize_feature_name(feature_names[idx])
        direction = "increases fraud likelihood" if raw_val > 0 else "decreases fraud likelihood"
        explanations.append({
            "field": field,
            "term": term,
            "shap_value": float(raw_val),
            "direction": direction,
            "raw_feature": feature_names[idx],
        })

    return explanations, base_value
