Fake Job Posting Detection (Fake-JD)

An NLP and Machine Learning system for detecting fraudulent and scam job postings using job description text, metadata features, and explainable AI (XAI).

Project Overview
Online employment scams target vulnerable job seekers. This project provides a machine-learning pipeline to classify job advertisements as legitimate or fraudulent. The system emphasizes transparency through **SHAP explainability** and domain-specific **red-flag indicators**.

Key Highlights
Preprocessing & Cleaning: Text normalization, missing feature indicators, and metadata encoding.
Handling Class Imbalance: SMOTE (Synthetic Minority Over-sampling Technique) applied to minority fraudulent postings.
Text & Metadata Features: TF-IDF vectorization on combined text fields ('description', 'requirements', 'company_profile', 'benefits') combined with categorical/binary metadata features via `ColumnTransformer`.
Classification Model: Tuned Logistic Regression optimized for fraud detection and balanced precision-recall.
Explainability (XAI): Linear SHAP analysis explaining individual posting risk scores.
Cross-Platform Generalization: Evaluated against external job postings across multiple platforms to measure real-world transferability.

Repository Structure

```text
├── data/
│   ├── raw/                  # Place raw datasets here (EMSCAD / Kaggle)
│   └── processed/            # Cleaned data and cross-platform holdout sets
├── models/                   # Serialized model weights and preprocessors
├── notebooks/                # Jupyter Notebooks detailing research & experiments
│   ├── data_extraction.ipynb
│   ├── preprocess.ipynb
│   ├── feature_extraction.ipynb
│   ├── model_training.ipynb
│   └── shap_analysis.ipynb
├── src/                      # Reusable modules and utilities
├── .gitignore                # Git ignore configuration
└── README.md
