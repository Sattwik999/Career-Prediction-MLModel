from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline  # noqa: F401  (kept for extensibility)
import joblib


def _resolve_path(relative_path: str) -> str:
    # If the provided path exists as-is, use it
    if os.path.exists(relative_path):
        return relative_path
    # Try resolving relative to this file (../relative_path)
    here = os.path.dirname(__file__)
    candidate = os.path.normpath(os.path.join(here, "..", relative_path))
    return candidate


def preprocess_dataset(file_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer, LabelEncoder]:
    """Load dataset, normalize columns, and prepare encoders.

    Returns:
        X: raw feature DataFrame (not yet transformed)
        y_encoded: encoded target labels (LabelEncoder applied)
        preprocessor: ColumnTransformer (unfitted; will be fit in train.py on train split)
        le: fitted LabelEncoder for the target
    """
    if file_path is None:
        file_path = _resolve_path(os.path.join("data", "career_dataset.csv"))
    else:
        file_path = _resolve_path(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Normalize key column names for robustness
    rename_map = {}
    cols = set(df.columns)
    if "Job Profession" in cols:
        rename_map["Job Profession"] = "Job profession"
    if "Logical-Mathematical" in cols:
        rename_map["Logical-Mathematical"] = "Logical - Mathematical"
    if "Spatial Visualization" in cols:
        rename_map["Spatial Visualization"] = "Spatial-Visualization"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Clean/strip whitespace in string columns including target
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    # Fill missing values for Course and P1..P8 style columns
    if "Course" in df.columns:
        df["Course"] = df["Course"].fillna("Unknown")
    for p in [f"P{i}" for i in range(1, 9) if f"P{i}" in df.columns]:
        df[p] = df[p].fillna("AVG")
    if "s/p" in df.columns:
        df["s/p"] = df["s/p"].fillna("Unknown")

    # Target handling (support either capitalization variant)
    target_col = "Job profession" if "Job profession" in df.columns else None
    if target_col is None and "Job Profession" in df.columns:
        target_col = "Job Profession"
    if target_col is None:
        raise KeyError("Target column 'Job profession' not found in dataset.")

    # Drop known non-feature identifiers if present
    drop_cols = [c for c in ["Student", "Sr.No.", target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Define feature groups with fallbacks
    cat_cols = [c for c in ["Course"] if c in X.columns] + [c for c in [f"P{i}" for i in range(1, 9)] if c in X.columns]
    if "s/p" in X.columns:
        cat_cols.append("s/p")

    num_candidates = [
        "Linguistic",
        "Musical",
        "Bodily",
        "Logical - Mathematical",
        "Spatial-Visualization",
        "Interpersonal",
        "Intrapersonal",
        "Naturalist",
    ]
    num_cols = [c for c in num_candidates if c in X.columns]

    # Basic numeric NA handling for safety (median)
    for c in num_cols:
        if X[c].isna().any():
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median())

    # Build preprocessor (unfitted here; fitted on train split in train.py to avoid leakage)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ]
    )

    # Label encode the target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Persist label encoder and the (unfitted) preprocessor scaffold now
    models_dir = _resolve_path(os.path.join("models"))
    os.makedirs(models_dir, exist_ok=True)
    # Do NOT save the preprocessor here; it will be fitted and saved in train.py
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))

    print(f"Dataset loaded: {df.shape}. Features: {X.shape[1]} | Target classes: {len(le.classes_)}")
    print(f"Categorical cols: {len(cat_cols)} | Numeric cols: {len(num_cols)}")

    return X, y_encoded, preprocessor, le


def load_preprocessors():
    """Load saved (possibly fitted) preprocessor and label encoder."""
    models_dir = _resolve_path(os.path.join("models"))
    pre = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
    print("Preprocessors loaded successfully!")
    return pre, le


if __name__ == "__main__":
    X, y_encoded, preprocessor, le = preprocess_dataset()
    print("\nPreprocessing completed.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y_encoded.shape}")
    print(f"Unique job professions: {len(le.classes_)}")