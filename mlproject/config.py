"""Configuration centrale — IBM HR Attrition.

Seul fichier à adapter si le dataset change.
Structure en dataclasses pour la clarté et l'immutabilité.
Les alias plats en bas de fichier assurent la rétrocompatibilité
avec data.py, features.py et train.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


# ── Niveaux de déploiement (Model Registry MLflow) ─────────────────────────

class DeployStage(str, Enum):
    """Alias MLflow associés au modèle enregistré."""
    CHAMPION  = "champion"   # meilleur modèle validé en production
    CHALLENGER = "challenger" # candidat en cours d'évaluation


# ── Schémas de configuration ────────────────────────────────────────────────

@dataclass(frozen=True)
class DataConfig:
    """Paramètres liés au dataset."""
    path: Path
    target: str
    columns_to_drop: tuple[str, ...]       # colonnes inutiles (constantes / IDs)
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    random_state: int = 42
    test_size: float  = 0.2


@dataclass(frozen=True)
class MLflowConfig:
    """Paramètres de tracking et de registry MLflow."""
    tracking_uri: str   # sqlite:///mlruns.db  (local) ou http://host:port (serveur)
    experiment: str     # nom de l'expérience dans MLflow
    model_name: str     # nom du modèle dans le Model Registry
    default_stage: DeployStage = DeployStage.CHALLENGER


# ── Instances ───────────────────────────────────────────────────────────────

DATA = DataConfig(
    path   = ROOT / "data" / "ibm_hr_attrition.csv",
    target = "Attrition",
    columns_to_drop = (
        "EmployeeCount",   # toujours 1
        "EmployeeNumber",  # identifiant unique — aucune valeur prédictive
        "Over18",          # toujours "Y"
        "StandardHours",   # toujours 80
    ),
    numeric_features = (
        "Age", "DailyRate", "DistanceFromHome", "Education",
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
        "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
    ),
    categorical_features = (
        "BusinessTravel", "Department", "EducationField",
        "Gender", "JobRole", "MaritalStatus", "OverTime",
    ),
)

MLFLOW = MLflowConfig(
    tracking_uri  = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlruns.db'}"),
    experiment    = os.getenv("MLFLOW_EXPERIMENT",   "ibm-hr-attrition"),
    model_name    = os.getenv("MODEL_NAME",          "attrition-classifier"),
    default_stage = DeployStage(os.getenv("DEPLOY_STAGE", DeployStage.CHALLENGER.value)),
)

# Répertoire de sauvegarde des modèles joblib
MODEL_DIR = ROOT / "models"

# ── Alias plats (rétrocompatibilité) ────────────────────────────────────────
# data.py, features.py et train.py importent ces noms directement.

DATA_PATH            = DATA.path
TARGET               = DATA.target
COLUMNS_TO_DROP: list[str]  = list(DATA.columns_to_drop)
NUMERIC_FEATURES: list[str]     = list(DATA.numeric_features)
CATEGORICAL_FEATURES: list[str] = list(DATA.categorical_features)
RANDOM_STATE         = DATA.random_state
MLFLOW_TRACKING_URI  = MLFLOW.tracking_uri
MLFLOW_EXPERIMENT    = MLFLOW.experiment
MODEL_NAME           = MLFLOW.model_name
API_URL              = os.getenv("API_URL", "http://127.0.0.1:8000")
