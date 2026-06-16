"""Tests — endpoints FastAPI (/health, /predict, /model-info)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from mlproject.api import app, ml

# Payload valide correspondant à l'exemple dans Features.model_config
SAMPLE_PAYLOAD = {
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


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_status_ok():
    """GET /health doit retourner 200 avec {"status": "ok"}."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── /model-info ───────────────────────────────────────────────────────────────

def test_model_info_no_model():
    """/model-info doit indiquer loaded=False quand aucun modèle n'est chargé."""
    ml.clear()
    client = TestClient(app)
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["loaded"] is False
    assert "model_name" in data


def test_model_info_with_model(trained_pipeline):
    """/model-info doit indiquer loaded=True et le bon model_type."""
    ml["model"]      = trained_pipeline
    ml["model_type"] = type(trained_pipeline.steps[-1][1]).__name__
    try:
        client = TestClient(app)
        resp = client.get("/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["loaded"] is True
        assert data["model_type"] == "LogisticRegression"
    finally:
        ml.clear()


# ── /predict ─────────────────────────────────────────────────────────────────

def test_predict_no_model_returns_503():
    """/predict doit retourner 503 si le modèle n'est pas chargé."""
    ml.clear()
    client = TestClient(app)
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 503


def test_predict_with_model_returns_200(trained_pipeline):
    """/predict doit retourner 200 avec un modèle chargé."""
    ml["model"]      = trained_pipeline
    ml["model_type"] = type(trained_pipeline.steps[-1][1]).__name__
    try:
        client = TestClient(app)
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200
    finally:
        ml.clear()


def test_predict_output_schema(trained_pipeline):
    """/predict doit retourner prediction (0 ou 1) et probability (float dans [0,1])."""
    ml["model"]      = trained_pipeline
    ml["model_type"] = type(trained_pipeline.steps[-1][1]).__name__
    try:
        client = TestClient(app)
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0
    finally:
        ml.clear()


def test_predict_invalid_payload_returns_422(trained_pipeline):
    """/predict doit retourner 422 si un champ obligatoire est manquant."""
    ml["model"] = trained_pipeline
    try:
        client = TestClient(app)
        # Payload incomplet : sans "Age"
        bad_payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "Age"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422
    finally:
        ml.clear()


def test_predict_invalid_categorical_returns_422(trained_pipeline):
    """/predict doit retourner 422 si une valeur catégorielle est invalide."""
    ml["model"] = trained_pipeline
    try:
        client = TestClient(app)
        bad_payload = {**SAMPLE_PAYLOAD, "Gender": "Unknown"}
        resp = client.post("/predict", json=bad_payload)
        assert resp.status_code == 422
    finally:
        ml.clear()


def test_predict_uses_lifespan_model(saved_model, monkeypatch):
    """/predict doit charger le modèle via le lifespan quand model.joblib existe."""
    monkeypatch.setattr("mlproject.api.MODEL_DIR", saved_model)

    with TestClient(app) as client:
        resp = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert resp.status_code == 200
        data = resp.json()
        assert data["prediction"] in (0, 1)
