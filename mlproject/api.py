"""API d'inférence — Prédiction d'attrition employé (FastAPI).

Prédit si un employé va quitter l'entreprise (Attrition = 1)
à partir de ses caractéristiques RH.

Endpoints :
    GET  /health      -> statut de l'API
    POST /predict     -> prédiction (0 = reste, 1 = quitte) + probabilité
    GET  /model-info  -> informations sur le modèle servi

Lancement (dev, reload automatique) :
    make api
    uvicorn mlproject.api:app --reload --host 127.0.0.1 --port 8000

Documentation interactive :
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MLFLOW, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modèle au démarrage, libère les ressources à l'arrêt."""
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        logger.warning("Modèle introuvable : %s — /predict retournera 503", model_path)
    else:
        pipeline = joblib.load(model_path)
        ml["model"]      = pipeline
        ml["model_type"] = type(pipeline.steps[-1][1]).__name__
        ml["model_path"] = str(model_path)
        logger.info("Modèle chargé : %s depuis %s", ml["model_type"], model_path)
    yield
    ml.clear()
    logger.info("Ressources libérées.")


# ── Métadonnées de l'API ──────────────────────────────────────────────────────

_DESCRIPTION = """
## Prédiction du risque d'attrition — IBM HR Analytics

Cette API expose le modèle champion entraîné sur le **IBM HR Analytics Employee Attrition Dataset**.

### Contexte
- **Problème** : Classification binaire — un employé va-t-il quitter l'entreprise ?
- **Classe 0** : l'employé **reste**
- **Classe 1** : l'employé **quitte** (attrition)
- **Dataset** : 1 470 employés, 30 features (23 numériques + 7 catégorielles)
- **Déséquilibre** : 83.9 % restent / 16.1 % quittent → le modèle est entraîné avec `class_weight="balanced"`

### Utilisation rapide
1. `GET /health` — vérifier que l'API est opérationnelle
2. `POST /predict` — soumettre les caractéristiques d'un employé, recevoir la prédiction
3. `GET /model-info` — consulter le type et la version du modèle servi
"""

_TAGS = [
    {
        "name": "Monitoring",
        "description": "Statut du service et informations sur le modèle déployé.",
    },
    {
        "name": "Inférence",
        "description": "Prédiction d'attrition à partir des caractéristiques d'un employé.",
    },
]

app = FastAPI(
    title="IBM HR Attrition — API de prédiction",
    description=_DESCRIPTION,
    version="0.1.0",
    openapi_tags=_TAGS,
    lifespan=lifespan,
    contact={"name": "Ethan Viéville"},
    license_info={"name": "ESGI — Projet MLOps"},
)


# ── Schéma d'entrée ───────────────────────────────────────────────────────────

class Features(BaseModel):
    """Caractéristiques d'un employé — 23 features numériques + 7 catégorielles.

    Correspond exactement aux colonnes utilisées à l'entraînement
    (NUMERIC_FEATURES + CATEGORICAL_FEATURES dans config.py).
    Les `Literal` garantissent que seules les valeurs présentes dans le dataset sont acceptées.
    """

    # ── Numériques (23) ──────────────────────────────────────────────────────
    Age:                     int = Field(..., ge=18, le=65, description="Âge de l'employé")
    DailyRate:               int = Field(..., ge=0,         description="Taux journalier")
    DistanceFromHome:        int = Field(..., ge=1, le=29,  description="Distance domicile-travail (km)")
    Education:               int = Field(..., ge=1, le=5,   description="Niveau d'éducation · 1=Below College 2=College 3=Bachelor 4=Master 5=Doctor")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4,   description="Satisfaction de l'environnement de travail · 1=Low 2=Medium 3=High 4=Very High")
    HourlyRate:              int = Field(..., ge=0,         description="Taux horaire")
    JobInvolvement:          int = Field(..., ge=1, le=4,   description="Implication au travail · 1=Low 2=Medium 3=High 4=Very High")
    JobLevel:                int = Field(..., ge=1, le=5,   description="Niveau hiérarchique du poste (1 à 5)")
    JobSatisfaction:         int = Field(..., ge=1, le=4,   description="Satisfaction au travail · 1=Low 2=Medium 3=High 4=Very High")
    MonthlyIncome:           int = Field(..., ge=0,         description="Revenu mensuel brut (USD)")
    MonthlyRate:             int = Field(..., ge=0,         description="Taux mensuel")
    NumCompaniesWorked:      int = Field(..., ge=0, le=9,   description="Nombre d'entreprises précédentes")
    PercentSalaryHike:       int = Field(..., ge=0, le=25,  description="Augmentation salariale cette année (%)")
    PerformanceRating:       int = Field(..., ge=1, le=4,   description="Évaluation annuelle · 1=Low 2=Good 3=Excellent 4=Outstanding")
    RelationshipSatisfaction:int = Field(..., ge=1, le=4,   description="Satisfaction des relations de travail · 1=Low 2=Medium 3=High 4=Very High")
    StockOptionLevel:        int = Field(..., ge=0, le=3,   description="Niveau d'options sur actions (0 à 3)")
    TotalWorkingYears:       int = Field(..., ge=0,         description="Années d'expérience professionnelle totale")
    TrainingTimesLastYear:   int = Field(..., ge=0, le=6,   description="Nombre de formations suivies l'an dernier")
    WorkLifeBalance:         int = Field(..., ge=1, le=4,   description="Équilibre vie pro / vie perso · 1=Bad 2=Good 3=Better 4=Best")
    YearsAtCompany:          int = Field(..., ge=0,         description="Années dans l'entreprise actuelle")
    YearsInCurrentRole:      int = Field(..., ge=0,         description="Années dans le rôle actuel")
    YearsSinceLastPromotion: int = Field(..., ge=0,         description="Années depuis la dernière promotion")
    YearsWithCurrManager:    int = Field(..., ge=0,         description="Années sous la direction du manager actuel")

    # ── Catégorielles (7) ────────────────────────────────────────────────────
    BusinessTravel: Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"] = Field(
        ..., description="Fréquence des déplacements professionnels"
    )
    Department: Literal["Sales", "Research & Development", "Human Resources"] = Field(
        ..., description="Département de l'employé"
    )
    EducationField: Literal[
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Human Resources", "Other"
    ] = Field(..., description="Domaine d'études")
    Gender: Literal["Male", "Female"] = Field(..., description="Genre")
    JobRole: Literal[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ] = Field(..., description="Intitulé du poste")
    MaritalStatus: Literal["Single", "Married", "Divorced"] = Field(
        ..., description="Situation familiale"
    )
    OverTime: Literal["Yes", "No"] = Field(
        ..., description="L'employé fait-il régulièrement des heures supplémentaires ?"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Age": 35, "DailyRate": 800, "DistanceFromHome": 5,
                    "Education": 3, "EnvironmentSatisfaction": 3, "HourlyRate": 60,
                    "JobInvolvement": 3, "JobLevel": 2, "JobSatisfaction": 3,
                    "MonthlyIncome": 5000, "MonthlyRate": 10000, "NumCompaniesWorked": 2,
                    "PercentSalaryHike": 15, "PerformanceRating": 3,
                    "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
                    "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
                    "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3,
                    "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
                    "BusinessTravel": "Travel_Rarely",
                    "Department": "Research & Development",
                    "EducationField": "Life Sciences",
                    "Gender": "Male",
                    "JobRole": "Research Scientist",
                    "MaritalStatus": "Married",
                    "OverTime": "No",
                }
            ]
        }
    }


# ── Schéma de sortie ─────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    """Résultat de prédiction retourné par /predict."""
    prediction:  int   = Field(..., description="Classe prédite · **0** = l'employé reste · **1** = risque de départ")
    probability: float = Field(..., description="Probabilité d'attrition P(Attrition=1) · entre 0.0 et 1.0")

    model_config = {
        "json_schema_extra": {
            "examples": [{"prediction": 0, "probability": 0.1342}]
        }
    }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Statut de l'API",
)
def health() -> dict:
    """Retourne `{"status": "ok"}` si l'API est opérationnelle.

    Utilisé pour les health checks (Docker, load balancer, etc.).
    """
    return {"status": "ok"}


@app.get(
    "/model-info",
    tags=["Monitoring"],
    summary="Informations sur le modèle déployé",
)
def model_info() -> dict:
    """Retourne les métadonnées du modèle actuellement en mémoire.

    - **loaded** : indique si le modèle est bien chargé
    - **model_type** : classe sklearn du dernier estimateur (ex. `LogisticRegression`)
    - **model_name** : nom dans le MLflow Model Registry
    - **version** : valeur de la variable d'environnement `MODEL_VERSION`
    """
    loaded = ml.get("model") is not None
    return {
        "loaded":     loaded,
        "model_type": ml.get("model_type", "unknown"),
        "model_name": MLFLOW.model_name,
        "version":    os.environ.get("MODEL_VERSION", "unknown"),
    }


@app.post(
    "/predict",
    tags=["Inférence"],
    summary="Prédire le risque d'attrition d'un employé",
    response_model=PredictionOut,
    responses={
        200: {"description": "Prédiction réussie"},
        422: {"description": "Payload invalide (champ manquant ou valeur hors contrainte)"},
        503: {"description": "Modèle non chargé — lancer d'abord `make train_models`"},
    },
)
def predict(features: Features) -> PredictionOut:
    """Prédit si un employé va quitter l'entreprise.

    **Entrée** : les 30 caractéristiques de l'employé (23 numériques + 7 catégorielles).

    **Sortie** :
    - `prediction = 0` → l'employé devrait **rester**
    - `prediction = 1` → risque de **départ** (attrition)
    - `probability` → score de confiance P(Attrition=1), entre 0 et 1

    > Le seuil de décision est fixé à **0.5**. Pour un contexte RH où les faux négatifs
    > (rater un départ) coûtent cher, on peut abaisser ce seuil côté client.
    """
    model = ml.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé — lance d'abord : make train  ou  make train_models",
        )

    row   = pd.DataFrame([features.model_dump()])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))
