"""Entraînement baseline — Logistic Regression.

Tracking MLflow délégué à mlproject.tracking (aucun import mlflow direct).
"""
from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8")

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import mlproject.tracking as tracking
from mlproject.config import MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.evaluation import compute_metrics
from mlproject.features import build_preprocessor


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    tracking.setup()

    print("=== Chargement des données ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")

    with tracking.start_run("baseline-logreg") as run:

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

        tracking.log_params({
            "C": c, "max_iter": max_iter,
            "class_weight": "balanced", "solver": "lbfgs",
        })

        print("\n=== Métriques ===")
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        tracking.log_metrics(metrics, prefix="test_")
        tracking.log_confusion_matrix(y_test, y_pred)
        tracking.log_classification_report(y_test, y_pred)
        tracking.log_model(pipeline)

        print(f"  F1 (Yes)    : {metrics['f1']:.4f}")
        print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
        print(f"  Precision   : {metrics['precision']:.4f}")
        print(f"  Recall      : {metrics['recall']:.4f}")

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_DIR / "model.joblib")
        from mlproject.model_registry import save_model_snapshot
        save_model_snapshot(
            pipeline,
            label=f"LogisticRegression (C={c}, max_iter={max_iter})",
            params={"C": c, "max_iter": max_iter, "class_weight": "balanced"},
            metrics=metrics,
            run_id=run.info.run_id,
            save_as_champion=True,
        )
        print(f"\n[OK] Modèle sauvegardé  -> {MODEL_DIR / 'model.joblib'}")
        print(f"[OK] Run MLflow         -> {run.info.run_id}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline LogisticRegression — IBM HR Attrition")
    parser.add_argument("--c",        type=float, default=1.0,  help="Régularisation C")
    parser.add_argument("--max-iter", type=int,   default=1000, help="Itérations max")
    args = parser.parse_args()
    train(c=args.c, max_iter=args.max_iter)
