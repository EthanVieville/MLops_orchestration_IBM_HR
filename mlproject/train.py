"""Entraînement baseline — Logistic Regression avec tracking MLflow."""
from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlproject.config import MLFLOW, MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.evaluation import (
    compute_metrics,
    log_classification_report,
    log_confusion_matrix,
)
from mlproject.features import build_preprocessor


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    # ── MLflow ──────────────────────────────────────────────────────────────
    # tracking_uri défini dans .env (sqlite:///mlruns.db par défaut)
    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    mlflow.set_experiment(MLFLOW.experiment)

    print("=== Chargement des données ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")

    with mlflow.start_run(run_name="baseline-logreg"):

        print("\n=== Entraînement (LogisticRegression) ===")
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(
                C=c,
                max_iter=max_iter,
                class_weight="balanced",  # compense le déséquilibre 84/16
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ])
        pipeline.fit(X_train, y_train)
        print(f"  C={c}, max_iter={max_iter}, class_weight=balanced")

        mlflow.log_params({
            "C":            c,
            "max_iter":     max_iter,
            "class_weight": "balanced",
            "solver":       "lbfgs",
        })

        print("\n=== Métriques ===")
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        print(f"  F1 (Yes)      : {metrics['f1']:.4f}")
        print(f"  F1 weighted   : {metrics['f1_weighted']:.4f}")
        print(f"  ROC-AUC       : {metrics['roc_auc']:.4f}")
        print(f"  Precision     : {metrics['precision']:.4f}")
        print(f"  Recall        : {metrics['recall']:.4f}")

        print("\n=== Classification Report ===")
        log_classification_report(y_test, y_pred)
        log_confusion_matrix(y_test, y_pred)

        mlflow.sklearn.log_model(pipeline, name="model")

        MODEL_DIR.mkdir(exist_ok=True)
        model_path = MODEL_DIR / "model.joblib"
        joblib.dump(pipeline, model_path)

        print(f"[OK] Modèle sauvegardé   -> {model_path}")
        print(f"[OK] Run MLflow          -> {MLFLOW.tracking_uri}")
        print(f"     Expérience          -> {MLFLOW.experiment}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline LogisticRegression — IBM HR Attrition")
    parser.add_argument("--c",        type=float, default=1.0,  help="Régularisation C (défaut: 1.0)")
    parser.add_argument("--max-iter", type=int,   default=1000, help="Itérations max (défaut: 1000)")
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)
