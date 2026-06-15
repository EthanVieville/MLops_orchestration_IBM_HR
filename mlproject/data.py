"""Chargement et découpage des données IBM HR Attrition."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from mlproject.config import COLUMNS_TO_DROP, DATA_PATH, RANDOM_STATE, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    # Encode la cible Yes/No → 1/0
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def split(df: pd.DataFrame, test_size: float = 0.2):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
