"""Métriques et artefacts MLflow pour les expériences de classification.

Fonctions réutilisées par train.py et train_models.py.
Toutes les fonctions log_* supposent qu'un run MLflow est actif.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # pas d'affichage — génération de fichiers uniquement
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calcule les métriques clés pour une classification binaire déséquilibrée.

    On privilégie ROC-AUC et F1 (classe positive) plutôt que l'accuracy,
    car la classe Yes (attrition) ne représente que ~16 % du dataset.
    """
    return {
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
    }


def log_confusion_matrix(y_true, y_pred) -> None:
    """Génère la matrice de confusion et la logue comme artefact MLflow (PNG)."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No (0)", "Yes (1)"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Confusion Matrix")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "confusion_matrix.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        mlflow.log_artifact(str(path))
    plt.close(fig)


def log_classification_report(y_true, y_pred) -> None:
    """Logue le classification report comme artefact texte MLflow et l'affiche."""
    report = classification_report(
        y_true, y_pred,
        target_names=["No (0)", "Yes (1)"],
        zero_division=0,
    )
    print(report)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "classification_report.txt"
        Path(path).write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(path))
