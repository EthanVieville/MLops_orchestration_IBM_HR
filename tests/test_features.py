"""Tests — pipeline de préprocessing (ColumnTransformer)."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer

from mlproject.data import split
from mlproject.features import build_preprocessor


def test_preprocessor_type():
    """build_preprocessor() doit retourner un ColumnTransformer."""
    preprocessor = build_preprocessor()
    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessor_output_no_nan(df):
    """La sortie transformée ne doit contenir aucun NaN."""
    X_train, X_test, y_train, _ = split(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    assert not np.isnan(X_transformed).any(), "NaN détectés dans la sortie du préprocesseur"


def test_preprocessor_output_shape(df):
    """La sortie doit avoir plus de colonnes que l'entrée (OneHotEncoder crée des dummies)."""
    X_train, _, _, _ = split(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X_train)

    from mlproject.config import NUMERIC_FEATURES
    # Après OHE, on attend au moins autant de colonnes que les features numériques
    assert X_transformed.shape[0] == len(X_train)
    assert X_transformed.shape[1] > len(NUMERIC_FEATURES)


def test_preprocessor_train_test_same_shape(df):
    """Après fit sur train, transform(test) doit produire le même nombre de colonnes."""
    X_train, X_test, _, _ = split(df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    assert X_train_t.shape[1] == X_test_t.shape[1]
