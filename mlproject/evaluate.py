"""Évaluation d'un modèle sauvegardé sur le test set.

Charge le champion (depuis model.joblib ou le MLflow Registry),
calcule les métriques et les log dans MLflow comme run indépendant.

Utile pour ré-évaluer sans réentraîner, ou pour auditer un déploiement.

Usage :
    make evaluate                   # charge models/model.joblib (défaut)
    make evaluate SOURCE=registry   # charge @champion depuis MLflow Registry
    uv run python -m mlproject.evaluate [--source file|registry]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

import joblib

import mlproject.tracking as tracking
from mlproject.config import MODEL_DIR, MLFLOW
from mlproject.data import load_data, split
from mlproject.evaluation import compute_metrics


def evaluate(source: str = "file") -> None:
    """Évalue le modèle champion sur le test set et log les résultats dans MLflow.

    Args:
        source: "file" charge models/model.joblib ;
                "registry" charge l'alias @champion depuis le Model Registry MLflow.
    """
    tracking.setup()

    print("=== Chargement des données ===")
    df = load_data()
    _, X_test, _, y_test = split(df)
    print(f"  Test : {X_test.shape} | Attrition rate : {y_test.mean():.1%}")

    print(f"\n=== Chargement du modèle (source={source}) ===")
    if source == "registry":
        import mlflow
        model_uri = f"models:/{MLFLOW.model_name}@champion"
        print(f"  URI : {model_uri}")
        pipeline = mlflow.sklearn.load_model(model_uri)
        source_label = model_uri
    else:
        model_path = MODEL_DIR / "model.joblib"
        if not model_path.exists():
            print(f"[ERREUR] Modèle introuvable : {model_path}")
            print("  Lance d'abord : make train  ou  make train_models")
            sys.exit(1)
        print(f"  Fichier : {model_path}")
        pipeline = joblib.load(model_path)
        source_label = str(model_path)

    print("\n=== Inférence ===")
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    run_name = f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with tracking.start_run(run_name) as run:
        tracking.log_params({"source": source_label, "n_test": len(y_test)})
        tracking.log_metrics(metrics, prefix="eval_")
        tracking.log_confusion_matrix(y_test, y_pred)
        tracking.log_classification_report(y_test, y_pred)
        run_id = run.info.run_id

    print("\n=== Métriques (test set) ===")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  F1 (Yes)  : {metrics['f1']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"\n[OK] Run MLflow -> {run_id}")
    print("[OK] MLflow UI  -> make mlflow_ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du champion — IBM HR Attrition")
    parser.add_argument(
        "--source",
        choices=["file", "registry"],
        default="file",
        help="Source du modèle : 'file' = models/model.joblib, 'registry' = MLflow @champion",
    )
    args = parser.parse_args()
    evaluate(source=args.source)
