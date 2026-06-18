"""Interface centralisée pour MLflow.

Toutes les interactions MLflow du projet passent par ce module.
Les scripts d'entraînement (train.py, train_models.py) n'importent
jamais mlflow directement — ils appellent uniquement ces fonctions.

Usage type :
    import mlproject.tracking as tracking

    tracking.setup()
    with tracking.start_run("mon-run") as run:
        tracking.log_params({"lr": 0.01})
        tracking.log_metrics(metrics, prefix="test_")
        tracking.log_confusion_matrix(y_test, y_pred)
        tracking.log_model(pipeline)
        tracking.register_champion(run.info.run_id)
"""
from __future__ import annotations

import contextlib
import sys
from typing import Generator

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from mlproject.config import MLFLOW, DeployStage


# ── Setup ───────────────────────────────────────────────────────────────────

def setup(experiment: str | None = None) -> None:
    """Configure MLflow (tracking URI + expérience).

    À appeler une fois en début de script, avant tout start_run().
    L'URI est lue depuis MLFLOW.tracking_uri (défini dans .env).
    """
    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    mlflow.set_experiment(experiment or MLFLOW.experiment)


# ── Runs ────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def start_run(name: str, nested: bool = False) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager qui ouvre un run MLflow (simple ou enfant).

    Args:
        name:   Nom affiché dans l'UI MLflow.
        nested: True pour un run enfant (sous un run parent actif).

    Yields:
        L'objet ActiveRun — son .info.run_id est utile pour le Registry.
    """
    with mlflow.start_run(run_name=name, nested=nested) as run:
        yield run


# ── Logging ─────────────────────────────────────────────────────────────────

def log_params(params: dict) -> None:
    """Log un dictionnaire de paramètres dans le run actif."""
    mlflow.log_params(params)


def log_tags(tags: dict) -> None:
    """Log un dictionnaire de tags dans le run actif.

    Les tags permettent de filtrer et regrouper des runs dans l'UI MLflow
    sans modifier la hiérarchie parent/enfant.
    Exemple : {"optimizer": "optuna", "model": "LightGBM"}
    """
    mlflow.set_tags(tags)


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    """Log un dictionnaire de métriques, avec préfixe optionnel.

    Exemple : log_metrics({"f1": 0.5, "roc_auc": 0.8}, prefix="test_")
              → logue "test_f1" et "test_roc_auc"

    Les valeurs NaN/Inf sont silencieusement ignorées (SQLite lève une
    contrainte d'unicité quand plusieurs NaN partagent la même clé/timestamp).
    """
    import math
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    valid = {k: v for k, v in metrics.items()
             if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
    if valid:
        mlflow.log_metrics(valid)
    if len(valid) < len(metrics):
        print(f"[WARN] Métriques NaN/Inf ignorées : {set(metrics) - set(valid)}")


# Types de classifieurs non-sklearn qui utilisent skops pour la sérialisation.
# MLflow >= 2.13 exige de les déclarer explicitement lors du log_model.
_SKOPS_TRUSTED = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.basic.Booster",
]


def log_model(pipeline, name: str = "model") -> None:
    """Log un pipeline scikit-learn comme artefact MLflow.

    Le modèle est rechargeable via :
        mlflow.sklearn.load_model(f"runs:/{run_id}/{name}")
    """
    mlflow.sklearn.log_model(pipeline, name=name, skops_trusted_types=_SKOPS_TRUSTED)


def log_confusion_matrix(y_true, y_pred) -> None:
    """Génère et log la matrice de confusion comme artefact PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["No (0)", "Yes (1)"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    # mlflow.log_figure() évite de passer par un fichier temporaire
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def log_classification_report(y_true, y_pred) -> None:
    """Log le classification report comme artefact texte et l'affiche."""
    report = classification_report(
        y_true, y_pred,
        target_names=["No (0)", "Yes (1)"],
        zero_division=0,
    )
    print(report)
    mlflow.log_text(report, "classification_report.txt")


# ── Model Registry ───────────────────────────────────────────────────────────

def register_champion(run_id: str, model_name: str | None = None) -> None:
    """Enregistre le modèle d'un run dans le Model Registry avec l'alias @champion.

    L'alias "@champion" permet de charger le meilleur modèle sans connaître
    son numéro de version :
        mlflow.sklearn.load_model(f"models:/{model_name}@champion")

    Args:
        run_id:     ID du run MLflow contenant l'artefact "model/".
        model_name: Nom dans le Registry (défaut : MLFLOW.model_name du .env).
    """
    name = model_name or MLFLOW.model_name
    try:
        registered = mlflow.register_model(f"runs:/{run_id}/model", name)
        mlflow.MlflowClient().set_registered_model_alias(
            name, DeployStage.CHAMPION.value, registered.version
        )
        print(f"[OK] Registry -> {name} v{registered.version} (@{DeployStage.CHAMPION.value})")
    except Exception as exc:
        # Le Registry nécessite sqlite:///mlruns.db comme MLFLOW_TRACKING_URI
        print(f"[WARN] Model Registry ignoré : {exc}")
