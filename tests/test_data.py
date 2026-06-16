"""Tests — chargement et split du dataset IBM HR."""
from __future__ import annotations

import pandas as pd

from mlproject.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from mlproject.data import split


def test_load_data_shape(df):
    """Le dataset doit avoir exactement 30 colonnes (4 constantes supprimées + target)."""
    assert isinstance(df, pd.DataFrame)
    expected_cols = 1 + len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)  # target + features
    assert df.shape[1] == expected_cols, f"Attendu {expected_cols} colonnes, obtenu {df.shape[1]}"


def test_load_data_target_encoded(df):
    """La colonne target doit être encodée en 0/1 (pas 'Yes'/'No')."""
    assert df[TARGET].dtype in ("int32", "int64", "bool")
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_load_data_no_dropped_columns(df):
    """Les 4 colonnes constantes doivent être absentes du DataFrame."""
    dropped = {"EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"}
    assert dropped.isdisjoint(df.columns), f"Colonnes trouvées : {dropped & set(df.columns)}"


def test_load_data_all_features_present(df):
    """Toutes les features de config.py doivent être présentes."""
    missing = (set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)) - set(df.columns)
    assert not missing, f"Features absentes : {missing}"


def test_split_ratio(df):
    """Le split 80/20 doit produire des tailles cohérentes (±1 ligne pour l'arrondi)."""
    X_train, X_test, y_train, y_test = split(df)
    n = len(df)
    assert abs(len(X_train) - int(n * 0.8)) <= 2
    assert abs(len(X_test)  - int(n * 0.2)) <= 2


def test_split_no_overlap(df):
    """Les index train et test ne doivent pas se chevaucher."""
    X_train, X_test, _, _ = split(df)
    assert len(set(X_train.index) & set(X_test.index)) == 0


def test_split_stratified(df):
    """La proportion de la classe positive doit être similaire dans train et test."""
    _, _, y_train, y_test = split(df)
    tol = 0.05  # 5 points de pourcentage de tolérance
    assert abs(y_train.mean() - y_test.mean()) < tol, (
        f"Déséquilibre entre train ({y_train.mean():.2%}) et test ({y_test.mean():.2%})"
    )
