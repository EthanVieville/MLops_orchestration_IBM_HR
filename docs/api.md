# API FastAPI

`mlproject/api.py` — expose le modèle champion via une API REST.

## Démarrage

```bash
make api                 # http://127.0.0.1:8000
make api PORT=8080       # port personnalisé
```

Documentation interactive Swagger : **http://127.0.0.1:8000/docs**

!!! warning "Prérequis"
    Un modèle entraîné doit exister dans `models/model.joblib`.
    Lance d'abord `make train` ou `make train_models`.

---

## Endpoints

### `GET /health`

Vérifie que l'API est opérationnelle.

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

---

### `GET /model-info`

Retourne les informations sur le modèle chargé en mémoire.

```bash
curl http://127.0.0.1:8000/model-info
```

```json
{
  "loaded": true,
  "model_type": "LogisticRegression",
  "model_name": "attrition-classifier",
  "version": "unknown"
}
```

`version` est lue depuis la variable d'environnement `MODEL_VERSION` (utile en Docker/CI).

---

### `POST /predict`

Prédit le risque d'attrition pour un employé.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "OverTime": "No"
  }'
```

```json
{"prediction": 0, "probability": 0.1342}
```

| Champ | Type | Description |
|-------|------|-------------|
| `prediction` | `int` | `0` = reste · `1` = quitte |
| `probability` | `float` | P(Attrition=1) entre 0.0 et 1.0 |

### Codes de retour

| Code | Cas |
|------|-----|
| `200` | Prédiction réussie |
| `422` | Payload invalide (champ manquant ou valeur hors contrainte) |
| `503` | Modèle non chargé |

---

## Validation Pydantic

Le schéma `Features` valide strictement :

- **Numériques** : type `int`, contraintes `ge` / `le` (ex. `Age` entre 18 et 65)
- **Catégorielles** : type `Literal` — seules les valeurs du dataset sont acceptées

```python
# Exemple : valeur catégorielle invalide → 422 automatiquement
{"Gender": "Unknown"}  # ❌ doit être "Male" ou "Female"
```

---

## Client de test

```bash
make predict_client   # envoie 3 requêtes /predict depuis le dataset
```

`scripts/predict_client.py` — construit des payloads depuis le dataset réel et affiche les réponses dans les logs.
