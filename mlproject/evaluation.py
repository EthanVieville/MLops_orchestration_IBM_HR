"""Calcul des métriques de classification — aucune dépendance MLflow.

Le logging des résultats (artefacts, métriques) est géré par tracking.py.
Ce module se concentre uniquement sur le calcul.
"""
from __future__ import annotations

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calcule les métriques clés pour une classification binaire déséquilibrée.

    On privilégie ROC-AUC et F1 (classe positive = Yes) plutôt que l'accuracy,
    car la classe Yes (attrition) ne représente que ~16 % du dataset.

    Returns:
        dict avec les clés : f1, f1_weighted, roc_auc, precision, recall
    """
    return {
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_proba),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
    }
