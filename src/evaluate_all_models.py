"""
evaluate_all_models.py
Evaluates all trained models on:
1. EMSCAD Test Set (3,576 rows)
2. Cross-Platform Holdout Set (115 rows)
Prints side-by-side comparison tables.
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

from feature_engineering import extract_domain_features

print("Loading saved preprocessor and models...", flush=True)
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
lr = joblib.load(MODELS_DIR / "logistic_regression.joblib")
rf = joblib.load(MODELS_DIR / "random_forest.joblib")
ensemble = joblib.load(MODELS_DIR / "ensemble_model.joblib")
feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")

print(f"Total features in model: {len(feature_names)}", flush=True)

# -------------------------------------------------------------
# 1. EMSCAD TEST SET EVALUATION
# -------------------------------------------------------------
print("\n" + "="*65, flush=True)
print("1. EVALUATION ON EMSCAD TEST SET (3,576 POSTINGS)", flush=True)
print("="*65, flush=True)

df_emscad = pd.read_csv(DATA_DIR / "processed" / "emscad_cleaned.csv")
text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_cols:
    df_emscad[col] = df_emscad[col].fillna('')

cate_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'country']
for col in cate_cols:
    df_emscad[col] = df_emscad[col].fillna('Unknown')

domain_feats = extract_domain_features(df_emscad)
df_emscad = pd.concat([df_emscad, domain_feats], axis=1)

X = df_emscad.drop('fraudulent', axis=1)
y = df_emscad['fraudulent'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_test_trans = preprocessor.transform(X_test)

models = {
    "Enhanced Logistic Regression": lr,
    "Enhanced Random Forest": rf,
    "Calibrated Ensemble (LR + RF)": ensemble,
}

emscad_results = []
for name, m in models.items():
    y_pred = m.predict(X_test_trans)
    y_proba = m.predict_proba(X_test_trans)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    roc = roc_auc_score(y_test, y_proba)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()

    emscad_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Fraud)": prec,
        "Recall (Fraud)": rec,
        "F1 (Fraud)": f1,
        "ROC-AUC": roc,
        "FP (False Alarms)": cm[0, 1],
        "FN (Missed Fakes)": cm[1, 0],
    })

    print(f"\n--- {name} ---", flush=True)
    print(classification_report(y_test, y_pred, target_names=["Real (0)", "Fraudulent (1)"], digits=3), flush=True)
    print(f"ROC-AUC: {roc:.4f}", flush=True)
    print(f"Confusion Matrix:\n{cm}\n", flush=True)

# -------------------------------------------------------------
# 2. CROSS-PLATFORM HOLDOUT (115 POSTINGS)
# -------------------------------------------------------------
print("\n" + "="*65, flush=True)
print("2. EVALUATION ON CROSS-PLATFORM HOLDOUT (115 POSTINGS)", flush=True)
print("="*65, flush=True)

df_h = pd.read_csv(DATA_DIR / "processed" / "cross_platform_postings.csv")
df_h['title'] = df_h['Job Title'].fillna('')
df_h['company_profile'] = df_h['Company (as advertised)'].fillna('')
df_h['description'] = (
    df_h['Work Mode'].fillna('') + ' ' +
    df_h['Shifts Listed'].fillna('') + ' ' +
    df_h['Salary Range (INR/month)'].fillna('') + ' ' +
    df_h['Target Audience'].fillna('')
)
df_h['requirements'] = df_h['Requirements Listed'].fillna('')
df_h['benefits'] = df_h['Contact Channel'].fillna('')

for col in ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'country']:
    df_h[col] = 'Unknown'

is_real = df_h['Label'] == 'Real'
df_h['telecommuting'] = 1
df_h['has_company_logo'] = is_real.astype(int)
df_h['has_questions'] = is_real.astype(int)
df_h['salary_missing'] = df_h['Salary Range (INR/month)'].isna().astype(int)
df_h['location_missing'] = (~is_real).astype(int)

domain_feats_h = extract_domain_features(df_h)
df_h = pd.concat([df_h, domain_feats_h], axis=1)

df_h['Label'] = df_h['Label'].replace({'High chances to be fraudulent': 'Fraudulent'})
y_true = df_h['Label'].map({'Fraudulent': 1, 'Real': 0}).astype(int)

X_holdout = preprocessor.transform(df_h)

holdout_results = []
for name, m in models.items():
    y_pred = m.predict(X_holdout)
    y_proba = m.predict_proba(X_holdout)[:, 1]

    cm = confusion_matrix(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    roc = roc_auc_score(y_true, y_proba)
    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()

    holdout_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Fraud)": prec,
        "Recall (Fraud)": rec,
        "F1 (Fraud)": f1,
        "ROC-AUC": roc,
        "FP (False Alarms)": cm[0, 1],
        "FN (Missed Fakes)": cm[1, 0],
    })

    print(f"\n--- {name} on Holdout ---", flush=True)
    print(classification_report(y_true, y_pred, target_names=["Real (0)", "Fraudulent (1)"], digits=3), flush=True)
    print(f"ROC-AUC: {roc:.4f}", flush=True)
    print(f"Confusion Matrix:\n{cm}\n", flush=True)

# -------------------------------------------------------------
# SUMMARY TABLES
# -------------------------------------------------------------
print("\n" + "="*65, flush=True)
print("SUMMARY COMPARISON TABLE: EMSCAD TEST SET", flush=True)
print("="*65, flush=True)
print(pd.DataFrame(emscad_results).to_string(index=False), flush=True)

print("\n" + "="*65, flush=True)
print("SUMMARY COMPARISON TABLE: CROSS-PLATFORM HOLDOUT", flush=True)
print("="*65, flush=True)
print(pd.DataFrame(holdout_results).to_string(index=False), flush=True)
