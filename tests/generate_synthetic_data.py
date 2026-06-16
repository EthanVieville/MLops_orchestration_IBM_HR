"""Génère un dataset IBM HR synthétique pour les tests CI.

Produit un CSV avec le même schéma que le dataset IBM HR Attrition réel
(35 colonnes identiques, mêmes types, mêmes valeurs possibles),
mais avec des données aléatoires.

Utilisé quand le vrai CSV n'est pas disponible (CI/CD, onboarding).
Si le vrai CSV est déjà présent, ce script ne fait rien.

Lancement :
    uv run python tests/generate_synthetic_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from mlproject.config import DATA_PATH

N_ROWS = 400  # assez pour un split 80/20 stratifié avec les deux classes


def generate(n: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    """Génère un DataFrame avec le schéma exact du dataset IBM HR Attrition.

    La répartition des classes reproduit le déséquilibre du dataset réel :
    ~16 % d'attrition (Yes) pour ~84 % de non-attrition (No).
    """
    rng = np.random.default_rng(seed)

    n_yes = int(n * 0.16)
    n_no  = n - n_yes
    attrition = ["Yes"] * n_yes + ["No"] * n_no
    rng.shuffle(attrition)

    return pd.DataFrame({
        # Colonnes constantes — seront supprimées par data.py
        "EmployeeCount":          1,
        "EmployeeNumber":         range(1, n + 1),
        "Over18":                 "Y",
        "StandardHours":          80,
        # Target
        "Attrition":              attrition,
        # Features numériques
        "Age":                    rng.integers(18, 61,  n),
        "DailyRate":              rng.integers(102, 1500, n),
        "DistanceFromHome":       rng.integers(1, 30,   n),
        "Education":              rng.integers(1, 6,    n),
        "EnvironmentSatisfaction":rng.integers(1, 5,    n),
        "HourlyRate":             rng.integers(30, 101, n),
        "JobInvolvement":         rng.integers(1, 5,    n),
        "JobLevel":               rng.integers(1, 6,    n),
        "JobSatisfaction":        rng.integers(1, 5,    n),
        "MonthlyIncome":          rng.integers(1009, 20000, n),
        "MonthlyRate":            rng.integers(2094, 27000, n),
        "NumCompaniesWorked":     rng.integers(0, 10,   n),
        "PercentSalaryHike":      rng.integers(11, 26,  n),
        "PerformanceRating":      rng.choice([3, 4],    n),
        "RelationshipSatisfaction":rng.integers(1, 5,   n),
        "StockOptionLevel":       rng.integers(0, 4,    n),
        "TotalWorkingYears":      rng.integers(0, 41,   n),
        "TrainingTimesLastYear":  rng.integers(0, 7,    n),
        "WorkLifeBalance":        rng.integers(1, 5,    n),
        "YearsAtCompany":         rng.integers(0, 41,   n),
        "YearsInCurrentRole":     rng.integers(0, 19,   n),
        "YearsSinceLastPromotion":rng.integers(0, 16,   n),
        "YearsWithCurrManager":   rng.integers(0, 18,   n),
        # Features catégorielles
        "BusinessTravel":  rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n),
        "Department":      rng.choice(["Sales", "Research & Development", "Human Resources"], n),
        "EducationField":  rng.choice(["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"], n),
        "Gender":          rng.choice(["Male", "Female"], n),
        "JobRole":         rng.choice([
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ], n),
        "MaritalStatus":   rng.choice(["Single", "Married", "Divorced"], n),
        "OverTime":        rng.choice(["Yes", "No"], n),
    })


if __name__ == "__main__":
    if DATA_PATH.exists():
        print(f"[OK] Dataset déjà présent : {DATA_PATH} — rien à faire.")
        sys.exit(0)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"[OK] Dataset synthétique généré : {DATA_PATH}  ({len(df)} lignes, {df.columns.size} colonnes)")
