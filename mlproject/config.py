"""Configuration centrale du projet — IBM HR Attrition.

Seul fichier à adapter si le dataset change. Tous les autres modules
lisent leurs paramètres depuis ces constantes.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --- Données ---------------------------------------------------------------
DATA_PATH = ROOT / "data" / "ibm_hr_attrition.csv"
MODEL_DIR = ROOT / "models"

# Colonne cible : Attrition encodée en 1 (Yes) / 0 (No)
TARGET = "Attrition"

# Colonnes supprimées au chargement (constantes ou identifiants sans valeur)
COLUMNS_TO_DROP: list[str] = [
    "EmployeeCount",   # toujours 1
    "EmployeeNumber",  # identifiant unique, pas prédictif
    "Over18",          # toujours "Y"
    "StandardHours",   # toujours 80
]

# --- Features numériques ---------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# --- Features catégorielles ------------------------------------------------
CATEGORICAL_FEATURES: list[str] = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# --- Divers ----------------------------------------------------------------
RANDOM_STATE = 42

# --- MLflow (surcharge possible via .env) ----------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT  = os.getenv("MLFLOW_EXPERIMENT", "ibm-hr-attrition")
MODEL_NAME         = os.getenv("MODEL_NAME", "attrition-classifier")
