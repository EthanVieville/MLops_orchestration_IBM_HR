# Prédiction de l'attrition des employés — IBM HR Analytics

![CI](https://github.com/EthanVIEVILLE/td_orchestrationML_EthanVIEVILLE/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/EthanVIEVILLE/td_orchestrationML_EthanVIEVILLE/actions/workflows/cd.yml/badge.svg)

Pipeline MLOps de classification binaire pour prédire si un employé va quitter l'entreprise.

**Auteur** : VIEVILLE Ethan

---

## Problématique

> **Un employé va-t-il quitter l'entreprise ?** (`Attrition = Yes/No`)

Anticiper les départs permet aux RH de prioriser les actions de rétention sur les profils à risque, réduisant les coûts de recrutement et la perte de compétences.

- `1` — l'employé quitte l'entreprise
- `0` — l'employé reste

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- 1 470 employés, 31 colonnes exploitables (4 colonnes constantes supprimées)
- Target : `Attrition` — 83.9% No / 16.1% Yes (dataset déséquilibré)
- 23 features numériques, 7 features catégorielles

---

## Installation

```bash
make install
```

## Commandes

```bash
make install                            # installe les dépendances
make train                              # baseline LogisticRegression (C=1.0)
make train C=0.1                        # avec régularisation plus forte
make train_models                       # compare 5 modèles via GridSearchCV + MLflow
make train_models CV=3                  # run rapide (3 folds au lieu de 5)
make train_optuna                       # optimisation Optuna TPE (LightGBM, 50 trials)
make train_optuna MODEL=XGBoost N_TRIALS=100   # autre modèle, plus de trials
make evaluate                           # évalue model.joblib sur le test set
make evaluate SOURCE=registry           # évalue le @champion MLflow Registry
make api                                # lance l'API FastAPI (http://127.0.0.1:8000/docs)
make api PORT=8080                      # sur un port personnalisé
make predict_client                     # envoie 3 requêtes de test à l'API locale
make test                               # tests unitaires pytest
make pipeline                           # pipeline complet : data -> train -> evaluate
make ci                                 # CI locale : ruff + pytest + pipeline
make mlflow_ui                          # ouvre l'UI MLflow sur http://127.0.0.1:5000
make check_pipeline                     # vérifie le pipeline de données
make doctor                             # diagnostique l'environnement

# Docker
make docker-build                       # construit l'image API localement
make docker-build IMAGE=myrepo/api TAG=1.0.0   # nom et tag personnalisés
make docker-run                         # lance l'API dans un conteneur (port 8000)
make docker-run PORT=8080               # sur un port personnalisé
make docker-push IMAGE=ghcr.io/... TAG=1.0.0   # pousse l'image sur GHCR
```

---

## Docker

L'image Docker de l'API est construite automatiquement par la pipeline CD et publiée sur **GitHub Container Registry** (GHCR) à chaque push sur `main` ou tag de version.

### Utilisation locale

```bash
# 1. Entraîner un modèle (génère models/model.joblib)
make train_models

# 2. Construire l'image
make docker-build

# 3. Lancer le conteneur (le dossier models/ est monté en lecture)
make docker-run

# 4. Tester
curl http://127.0.0.1:8000/health
```

### Utilisation de l'image GHCR

```bash
# Récupérer la dernière image publiée
docker pull ghcr.io/ethanvieville/td_orchestrationml_ethanvieville/mlproject-api:latest

# Lancer avec un modèle local
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  ghcr.io/ethanvieville/td_orchestrationml_ethanvieville/mlproject-api:latest
```

> **Note** : le modèle (`models/model.joblib`) est monté en volume — il n'est pas inclus dans l'image. L'API démarre sans modèle et retourne 503 sur `/predict` jusqu'à ce qu'un modèle soit disponible.

---

## Pipeline complète avec Docker Compose

`docker compose up` exécute la **pipeline de bout en bout** : données → entraînement → prédiction.

```
trainer  ──► (écrit model.joblib)
                     │
                     ▼
api      ──► http://127.0.0.1:8000
```

### Commandes

```bash
# Pipeline standard (entraînement + API)
make compose-up

# Pipeline + interface MLflow sur http://127.0.0.1:5000
make compose-monitoring

# Arrêter (les volumes model_vol et mlruns_vol sont conservés)
make compose-down

# Repartir de zéro (supprime aussi les volumes)
make compose-clean

# Voir les logs en direct
make compose-logs
```

### Ce qui se passe step by step

1. **`trainer`** démarre : génère 400 lignes synthétiques si `data/` est vide, puis lance `train_models.py --cv 3` (GridSearchCV sur LR, RF, XGBoost, LightGBM, MLP). Écrit `model.joblib` dans le volume partagé et s'arrête.
2. **`api`** démarre dès que `trainer` se termine avec succès (`service_completed_successfully`) : charge `model.joblib` et sert les prédictions sur `:8000`.
3. (optionnel) **`mlflow-ui`** expose l'interface MLflow sur `:5000` avec toutes les métriques des runs.

> Si tu as le vrai CSV IBM HR dans `data/`, il est monté automatiquement — sinon le dataset synthétique (400 lignes) est généré.

### Architecture des services

| Service | Image | Rôle | Port |
|---|---|---|---|
| `trainer` | `docker/Dockerfile.trainer` | GridSearchCV 5 modèles → sauvegarde champion | — |
| `api` | `docker/Dockerfile.api` | FastAPI inference `/predict` | 8000 |
| `mlflow-ui` | `python:3.13-slim` + mlflow | Visualisation des runs | 5000 |

---

## Résultats — Comparaison multi-modèles (GridSearchCV, CV=5)

Critère de sélection : **ROC-AUC** (seuil-indépendant, robuste aux classes déséquilibrées)

| Modèle | ROC-AUC | F1 (Yes) | Recall | Precision |
|---|---|---|---|---|
| **LogisticRegression** | **0.8124** | 0.4885 | 0.6809 | 0.3810 |
| RandomForest | 0.7955 | 0.4771 | 0.5532 | 0.4194 |
| LightGBM | 0.7735 | 0.4110 | 0.3191 | 0.5769 |
| MLP (PyTorch) | 0.7678 | 0.5321 | 0.6170 | 0.4688 |
| XGBoost | 0.7600 | 0.4421 | 0.4468 | 0.4375 |

> Champion actuel : **LogisticRegression** (C=0.1, class_weight=balanced) — enregistré dans le Model Registry MLflow sous l'alias `@champion`.  
> Le MLP PyTorch obtient le meilleur F1 (0.53) et un Recall élevé (0.62) grâce au `pos_weight` interne — bon profil pour le contexte RH où rater un départ coûte cher.

---

## Structure actuelle

```
.
├── data/
│   └── ibm_hr_attrition.csv       # Dataset IBM (non versionné)
├── models/
│   └── model.joblib               # Meilleur modèle (non versionné)
├── docker/
│   ├── Dockerfile.api             # Image Docker de l'API FastAPI (slim, ~500 Mo)
│   ├── Dockerfile.trainer         # Image Docker pour l'entraînement (torch CPU, ~2 Go)
│   ├── requirements-api.txt       # Dépendances minimales de serving
│   └── requirements-train.txt     # Dépendances d'entraînement (ML stack)
├── docker-compose.yml             # Pipeline complète : trainer → api (+ mlflow-ui optionnel)
├── mlproject/
│   ├── config.py                  # DataConfig, MLflowConfig, DeployStage + aliases
│   ├── data.py                    # Chargement & split 80/20 stratifié
│   ├── features.py                # Préprocessing (StandardScaler + OneHotEncoder)
│   ├── evaluation.py              # Calcul des métriques (f1, roc_auc, precision, recall)
│   ├── tracking.py                # Interface centralisée MLflow (aucun autre fichier n'importe mlflow)
│   ├── models.py                  # TorchMLPClassifier — MLP PyTorch compatible sklearn Pipeline
│   ├── train.py                   # Baseline Logistic Regression + MLflow
│   ├── train_models.py            # GridSearchCV sur 5 modèles + Model Registry
│   ├── train_optuna.py            # Optimisation Optuna TPE + MLflow (un run par trial)
│   ├── evaluate.py                # Évaluation du champion sauvegardé sur le test set
│   └── api.py                     # API FastAPI : /health, /predict, /model-info
├── scripts/
│   └── predict_client.py          # Client HTTP de test pour l'API
├── tests/
│   ├── conftest.py                # Fixtures pytest + génération dataset synthétique
│   ├── generate_synthetic_data.py # Génère ibm_hr_attrition.csv synthétique (CI)
│   ├── check_pipeline.py          # Validation rapide du pipeline de données (standalone)
│   ├── test_data.py               # Tests chargement et split du dataset
│   ├── test_features.py           # Tests pipeline de préprocessing
│   ├── test_evaluation.py         # Tests calcul des métriques
│   └── test_api.py                # Tests endpoints FastAPI
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI : tests unitaires + smoke test pipeline ML
│       └── cd.yml                 # CD : build & push image Docker sur GHCR
├── .dockerignore                  # Exclut .venv, data/, mlruns/ du contexte Docker
├── .env                           # Variables d'environnement (non versionné)
├── .env.example                   # Template .env
├── Makefile
└── pyproject.toml
```

---

## Avancement

| Séance | Objectif | Statut |
|---|---|---|
| S0 | Configuration dataset IBM + pipeline de données | ✅ |
| S5 | Baseline Logistic Regression + MLflow tracking | ✅ |
| S6/S7 | Comparaison multi-modèles (GridSearchCV) + MLflow Registry | ✅ |
| S7bis | Optimisation Optuna TPE + évaluation standalone | ✅ |
| S12 | API FastAPI (/predict, /health, /model-info) | ✅ |
| S15 | Tests pytest (25 tests) + CI GitHub Actions | ✅ |
| S19 | Dockerfile API (multi-stage) + CD GitHub Actions → GHCR | ✅ |
| S14 | Docker Compose (trainer + api + mlflow-ui) — pipeline from scratch | ✅ |
| S14bis | Frontend Streamlit | 🔲 |
| S17 | DAG Airflow | 🔲 |
