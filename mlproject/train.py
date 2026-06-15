"""Entraînement baseline — Logistic Regression."""
from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from mlproject.config import MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    print("=== Chargement des données ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")

    print("\n=== Entraînement (LogisticRegression) ===")
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            C=c,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_train, y_train)
    print(f"  C={c}, max_iter={max_iter}, class_weight=balanced")

    print("\n=== Métriques ===")
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f1       = f1_score(y_test, y_pred)
    f1_w     = f1_score(y_test, y_pred, average="weighted")
    roc_auc  = roc_auc_score(y_test, y_proba)
    precision = f1_score(y_test, y_pred, average=None)[1]
    recall    = f1_score(y_test, y_pred, average=None, labels=[1])[0]

    print(f"  F1 (Yes / class 1)  : {f1:.4f}")
    print(f"  F1 weighted         : {f1_w:.4f}")
    print(f"  ROC-AUC             : {roc_auc:.4f}")
    print(f"  Precision (Yes)     : {precision:.4f}")
    print(f"  Recall    (Yes)     : {recall:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))

    print("=== Confusion Matrix ===")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}   TP={tp}")

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n[OK] Modèle sauvegardé -> {model_path}")

    return {"f1": f1, "f1_weighted": f1_w, "roc_auc": roc_auc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline LogisticRegression — IBM HR Attrition")
    parser.add_argument("--c",        type=float, default=1.0,   help="Regularisation C (défaut: 1.0)")
    parser.add_argument("--max-iter", type=int,   default=1000,  help="Iterations max (défaut: 1000)")
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)
