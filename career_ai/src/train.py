from __future__ import annotations

import os
from datetime import datetime

import joblib
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from preprocess import preprocess_dataset, _resolve_path


def train_random_forest(n_estimators: int = 200, random_state: int = 42):
    # Load raw features, encoded target, and preprocessor scaffold
    X, y, preprocessor, le = preprocess_dataset()

    # Split before fitting preprocessor to prevent data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Fit preprocessor on train only, then transform train/test
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Train RandomForest
    model_rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model_rf.fit(X_train, y_train)

    # Evaluate
    y_pred = model_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"RandomForest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Persist artifacts
    models_dir = _resolve_path("models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model_rf, os.path.join(models_dir, "model_rf.pkl"))
    # Save the FITTED preprocessor (overwrites scaffold)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))

    # Save summary
    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_classes": int(len(le.classes_)),
        "rf_accuracy": float(acc),
        "n_train": n_train,
        "n_test": n_test,
    }
    with open(os.path.join(models_dir, "training_summary.txt"), "w", encoding="utf-8") as f:
        f.write("CAREER PREDICTION MODEL TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("\nRandomForest model and fitted preprocessor saved to models/.")
    return summary

def train_xgboost(n_estimators: int = 200, random_state: int = 42):
    if XGBClassifier is None:
        print("XGBoost is not installed. Skipping XGBoost training.")
        return None
    X, y, preprocessor, le = preprocess_dataset()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    model_xgb = XGBClassifier(
        n_estimators=n_estimators, random_state=random_state, use_label_encoder=False, eval_metric='mlogloss'
    )
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    models_dir = _resolve_path("models")
    joblib.dump(model_xgb, os.path.join(models_dir, "model_xgb.pkl"))
    print("XGBoost model saved to models/model_xgb.pkl")
    return acc


if __name__ == "__main__":
    try:
        print("Training RandomForest...")
        train_random_forest()
        print("\nTraining XGBoost...")
        train_xgboost()
        print("\nTraining completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")