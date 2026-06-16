"""Tests — calcul des métriques de classification."""
from __future__ import annotations

import numpy as np
import pytest

from mlproject.evaluation import compute_metrics


# Cas parfait : classifieur oracle
_Y_TRUE  = np.array([0, 0, 1, 1, 0, 1])
_Y_PRED  = np.array([0, 0, 1, 1, 0, 1])
_Y_PROBA = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.7])

# Cas imparfait : quelques erreurs
_Y_PRED_IMPERFECT  = np.array([0, 1, 0, 1, 0, 1])
_Y_PROBA_IMPERFECT = np.array([0.2, 0.6, 0.4, 0.9, 0.1, 0.8])


def test_compute_metrics_keys():
    """compute_metrics doit retourner exactement les 5 clés attendues."""
    metrics = compute_metrics(_Y_TRUE, _Y_PRED, _Y_PROBA)
    expected_keys = {"f1", "f1_weighted", "roc_auc", "precision", "recall"}
    assert set(metrics.keys()) == expected_keys


def test_compute_metrics_perfect_classifier():
    """Un classifieur parfait doit obtenir f1=1, roc_auc=1, precision=1, recall=1."""
    metrics = compute_metrics(_Y_TRUE, _Y_PRED, _Y_PROBA)
    assert metrics["f1"]        == pytest.approx(1.0)
    assert metrics["roc_auc"]   == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"]    == pytest.approx(1.0)


def test_compute_metrics_range():
    """Toutes les métriques doivent être dans [0, 1]."""
    metrics = compute_metrics(_Y_TRUE, _Y_PRED_IMPERFECT, _Y_PROBA_IMPERFECT)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"{name}={value} hors de [0, 1]"


def test_compute_metrics_returns_floats():
    """Toutes les valeurs retournées doivent être des float."""
    metrics = compute_metrics(_Y_TRUE, _Y_PRED, _Y_PROBA)
    for name, value in metrics.items():
        assert isinstance(value, float), f"{name} n'est pas un float : {type(value)}"


def test_compute_metrics_on_pipeline_output(df, trained_pipeline):
    """compute_metrics doit fonctionner sur les sorties réelles d'un pipeline sklearn."""
    from mlproject.data import split

    _, X_test, _, y_test = split(df)
    y_pred  = trained_pipeline.predict(X_test)
    y_proba = trained_pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    assert 0.5 < metrics["roc_auc"] < 1.0, "ROC-AUC anormalement bas sur données synthétiques"
