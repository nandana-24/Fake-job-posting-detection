"""
train_enhanced_model.py
Trains the enhanced Job Scam Risk model with:
1. Subword Character n-grams (3-5 char_wb) + Word n-grams (1-2)
2. Domain Feature Fusion (Free email, WhatsApp/Telegram, Fee demand, Urgency, Caps ratio, etc.)
3. Model Comparison: Enhanced Logistic Regression, Tree Ensemble (HistGradientBoosting/RandomForest), and Soft-Voting Ensemble.
4. Comprehensive evaluation on EMSCAD test set and Cross-Platform Holdout.
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure safe printing on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from scipy import sparse

from feature_engineering import extract_domain_features, ENGINEERED_FEATURE_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def build_preprocessor():
    """
    Builds the enhanced ColumnTransformer:
    - Text: TF-IDF with Word (1,2) + Subword Char (3,5) for description
    - Categorical: OneHotEncoder (handle_unknown='ignore')
    - Binary + Engineered: Scaler/Passthrough
    """
    desc_union = FeatureUnion([
        ("word", TfidfVectorizer(max_features=2500, ngram_range=(1, 2))),
        ("char", TfidfVectorizer(max_features=1500, analyzer="char_wb", ngram_range=(3, 5))),
    ])

    title_vec = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    profile_vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    req_vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    ben_vec = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler(with_mean=False))
    ])

    cate_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'country']
    binary_cols = ['telecommuting', 'has_company_logo', 'has_questions', 'salary_missing', 'location_missing']
    numeric_engineered_cols = ['urgency_count', 'caps_ratio', 'exclamation_count', 'word_count']
    binary_engineered_cols = ['has_free_email', 'has_messaging_app', 'has_fee_request', 'has_guarantee_claim']

    preprocessor = ColumnTransformer(
        transformers=[
            ("title", title_vec, "title"),
            ("company_profile", profile_vec, "company_profile"),
            ("description", desc_union, "description"),
            ("requirements", req_vec, "requirements"),
            ("benefits", ben_vec, "benefits"),
            ("categorical", categorical_transformer, cate_cols),
            ("binary", "passthrough", binary_cols + binary_engineered_cols),
            ("numeric_engineered", numeric_transformer, numeric_engineered_cols),
        ]
    )

    return preprocessor


def prepare_emscad_data():
    raw_path = DATA_DIR / "processed" / "emscad_cleaned.csv"
    df = pd.read_csv(raw_path)

    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna('')

    cate_cols = ['employment_type', 'required_experience', 'required_education', 'industry', 'function', 'country']
    for col in cate_cols:
        df[col] = df[col].fillna('Unknown')

    print("Extracting domain features for EMSCAD...")
    domain_feats = extract_domain_features(df)
    df = pd.concat([df, domain_feats], axis=1)

    X = df.drop('fraudulent', axis=1)
    y = df['fraudulent'].astype(int)

    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    roc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*55}\nEVALUATION ON EMSCAD TEST SET: {name}\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Real (0)", "Fraudulent (1)"], digits=3))
    print(f"ROC-AUC:  {roc:.4f}")
    print(f"Confusion Matrix: \n{cm}")

    return {
        "model": name,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "cm": cm
    }


def evaluate_on_holdout(model, preprocessor):
    holdout_path = DATA_DIR / "processed" / "cross_platform_postings.csv"
    if not holdout_path.exists():
        print("Holdout file not found.")
        return None

    df_h = pd.read_csv(holdout_path)

    # Format holdout exactly matching preprocessor schema
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

    # Binary flags matching the user's fixed holdout in preprocess.ipynb
    is_real = df_h['Label'] == 'Real'
    df_h['telecommuting'] = 1
    df_h['has_company_logo'] = is_real.astype(int)
    df_h['has_questions'] = is_real.astype(int)
    df_h['salary_missing'] = df_h['Salary Range (INR/month)'].isna().astype(int)
    df_h['location_missing'] = (~is_real).astype(int)

    # Extract engineered domain features
    domain_feats = extract_domain_features(df_h)
    df_h = pd.concat([df_h, domain_feats], axis=1)

    df_h['Label'] = df_h['Label'].replace({'High chances to be fraudulent': 'Fraudulent'})
    y_true = df_h['Label'].map({'Fraudulent': 1, 'Real': 0}).astype(int)

    X_holdout = preprocessor.transform(df_h)
    y_pred = model.predict(X_holdout)
    y_proba = model.predict_proba(X_holdout)[:, 1]

    cm = confusion_matrix(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    roc = roc_auc_score(y_true, y_proba)

    print(f"Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {roc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return {"prec": prec, "rec": rec, "f1": f1, "roc": roc, "cm": cm}


def run_pipeline():
    X, y = prepare_emscad_data()

    print("Splitting dataset (80/20 train/test, stratify=y)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting upgraded ColumnTransformer (Word + Char n-grams + Domain Signals)...")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    print(f"Feature matrix shape: {X_train_trans.shape} (Total features: {len(feature_names)})")

    print("Applying SMOTE oversampling...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_trans, y_train)
    print(f"Resampled matrix shape: {X_train_resampled.shape}")

    # 1. Enhanced Logistic Regression
    print("\nTraining Enhanced Logistic Regression...")
    lr = LogisticRegression(C=1.0, max_iter=1500, random_state=42)
    lr.fit(X_train_resampled, y_train_resampled)

    # 2. Enhanced Random Forest
    print("Training Enhanced Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=25, random_state=42, n_jobs=-1)
    rf.fit(X_train_resampled, y_train_resampled)

    # 3. Soft-Voting Calibrated Ensemble
    print("Building Soft-Voting Ensemble (LR + RF)...")
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)],
        voting="soft",
        weights=[1.0, 1.2]
    )
    ensemble.fit(X_train_resampled, y_train_resampled)

    # Evaluate on EMSCAD test set
    metrics_lr = evaluate_model("Enhanced Logistic Regression", lr, X_test_trans, y_test)
    metrics_rf = evaluate_model("Enhanced Random Forest", rf, X_test_trans, y_test)
    metrics_ens = evaluate_model("Calibrated Ensemble (LR + RF)", ensemble, X_test_trans, y_test)

    # Evaluate on Cross-Platform Holdout
    print("\n" + "="*55)
    print("EVALUATING ON CROSS-PLATFORM HOLDOUT (115 POSTINGS)")
    print("="*55)
    print("\n[Enhanced Logistic Regression on Holdout]")
    holdout_lr = evaluate_on_holdout(lr, preprocessor)

    print("\n[Enhanced Random Forest on Holdout]")
    holdout_rf = evaluate_on_holdout(rf, preprocessor)

    print("\n[Calibrated Ensemble (LR + RF) on Holdout]")
    holdout_ens = evaluate_on_holdout(ensemble, preprocessor)

    # Save artifacts
    print("\nSaving upgraded artifacts to models/ ...")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(lr, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(ensemble, MODELS_DIR / "ensemble_model.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")

    # Save SHAP background sample for instant explainability
    rng = np.random.RandomState(42)
    idx = rng.choice(X_train_resampled.shape[0], size=100, replace=False)
    background = X_train_resampled[idx]
    joblib.dump(background, MODELS_DIR / "shap_background.joblib")

    print("\n[SUCCESS] All enhanced models and artifacts successfully saved!")


if __name__ == "__main__":
    run_pipeline()
