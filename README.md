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
- Target : `Attrition` — 83.9 % No / 16.1 % Yes (dataset déséquilibré)
- 23 features numériques, 7 features catégorielles

---

## Installation

```bash
make install
```

## Commandes

```bash
# ── Environnement ────────────────────────────────────────────────────────────
make install                              # installe les dépendances (uv sync)
make doctor                               # diagnostique l'environnement
make ci                                   # CI locale : ruff + pytest + pipeline

# ── Entraînement ─────────────────────────────────────────────────────────────
make train                                # baseline LogisticRegression (C=1.0)
make train C=0.1 MAX_ITER=500             # paramètres personnalisés
make train_models                         # compare 5 modèles via GridSearchCV + MLflow
make train_models CV=3                    # run rapide (3 folds au lieu de 5)
make train_optuna                         # optimisation Optuna TPE (LightGBM, 50 trials)
make train_optuna MODEL=XGBoost N_TRIALS=100

# ── Évaluation & API ─────────────────────────────────────────────────────────
make evaluate                             # évalue model.joblib sur le test set
make evaluate SOURCE=registry             # évalue le @champion MLflow Registry
make api                                  # API FastAPI sur http://127.0.0.1:8000/docs
make predict_client                       # envoie 3 requêtes de test à l'API locale
make mlflow_ui                            # MLflow UI sur http://127.0.0.1:5000

# ── Dashboard Streamlit ───────────────────────────────────────────────────────
make streamlit                            # dashboard sur http://127.0.0.1:8501

# ── Airflow — Orchestration ───────────────────────────────────────────────────
make airflow                              # Airflow seul sur http://127.0.0.1:8080
make airflow-stack                        # stack complète (baseline) + Airflow
make airflow-trigger                      # déclenche model_retraining manuellement
make airflow-logs                         # logs Airflow en temps réel
make airflow-down                         # arrête Airflow (conserve les données)
make airflow-clean                        # repart from scratch (supprime le volume)

# ── Tests ────────────────────────────────────────────────────────────────────
make test                                 # tests unitaires pytest

# ── Docker (image API seule) ─────────────────────────────────────────────────
make docker-build                         # construit l'image API localement
make docker-run                           # lance l'API dans un conteneur (port 8000)
make docker-push IMAGE=ghcr.io/... TAG=1.0.0

# ── Docker Compose (pipeline complète) ───────────────────────────────────────
make compose-up                           # baseline (~30 s) → API → Dashboard
make compose-full                         # GridSearchCV 5 modèles (~10 min) → API → Dashboard
make compose-optuna                       # Optuna LightGBM 20 trials (~5 min) → API → Dashboard
make compose-monitoring                   # + MLflow UI sur http://127.0.0.1:5000
make compose-down                         # arrête (conserve les volumes)
make compose-clean                        # arrête + supprime les volumes
make compose-logs                         # logs en temps réel
```

---

## Docker

L'image Docker de l'API est construite automatiquement par la pipeline CD et publiée sur **GitHub Container Registry** (GHCR) à chaque push sur `main` ou tag de version.

### Utilisation locale (image API seule)

```bash
# 1. Entraîner un modèle (génère models/model.joblib)
make train

# 2. Construire l'image
make docker-build

# 3. Lancer le conteneur (le dossier models/ est monté en lecture)
make docker-run

# 4. Tester
curl http://127.0.0.1:8000/health
```

### Utilisation de l'image GHCR

```bash
docker pull ghcr.io/ethanvieville/td_orchestrationml_ethanvieville/mlproject-api:latest

docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  ghcr.io/ethanvieville/td_orchestrationml_ethanvieville/mlproject-api:latest
```

> **Note** : le modèle (`models/model.joblib`) est monté en volume — il n'est pas inclus dans l'image. L'API retourne 503 sur `/predict` jusqu'à ce qu'un modèle soit disponible.

---

## Pipeline complète avec Docker Compose

`make compose-up` exécute la **pipeline de bout en bout** en une seule commande : données → entraînement → API → dashboard.

```
trainer  ──► (écrit model.joblib)
                     │
                     ▼
api      ──► http://127.0.0.1:8000
                     │
                     ▼
streamlit ──► http://127.0.0.1:8501
```

### Ce qui se passe step by step

1. **`trainer`** — génère 400 lignes synthétiques si `data/` est vide, entraîne le modèle selon `TRAIN_MODE`, écrit `model.joblib` + les snapshots `.meta.json` dans le volume partagé, puis s'arrête.
2. **`api`** — démarre dès que `trainer` termine avec succès, charge `model.joblib`, sert les prédictions sur `:8000`.
3. **`streamlit`** — démarre dès que l'API est healthy (healthcheck OK), connecte sur `http://api:8000`, expose le dashboard sur `:8501`.
4. *(optionnel)* **`mlflow-ui`** — expose l'interface MLflow sur `:5000` (profil `monitoring`).

### Modes d'entraînement

| Commande | Mode | Algorithme | Durée |
|---|---|---|---|
| `make compose-up` | `baseline` | LogisticRegression (max_iter=300) | ~30 s |
| `make compose-full` | `full` | GridSearchCV — 5 modèles, CV=2 | ~10 min |
| `make compose-optuna` | `optuna` | Optuna TPE — LightGBM, 20 trials | ~5 min |

### Architecture des services

| Service | Image | Rôle | Port | Profil |
|---|---|---|---|---|
| `trainer` | `docker/Dockerfile.trainer` | Entraînement → sauvegarde champion | — | (toujours) |
| `api` | `docker/Dockerfile.api` | FastAPI `/predict`, `/models`, `/models/switch` | 8000 | (toujours) |
| `streamlit` | `docker/Dockerfile.streamlit` | Dashboard MLOps 6 onglets | 8501 | (toujours) |
| `airflow` | `docker/Dockerfile.airflow` | Orchestration DAGs — UI Airflow | 8080 | `airflow` |
| `mlflow-ui` | `python:3.13-slim` + mlflow | Visualisation des runs | 5000 | `monitoring` |

---

## Dashboard Streamlit

Interface complète accessible sur **http://127.0.0.1:8501** (local) ou via Docker Compose.

| Onglet | Contenu |
|---|---|
| 🏠 Contexte | Problématique IBM HR, statistiques dataset, facteurs de risque, architecture MLOps |
| 🔮 Prédiction | Formulaire employé (30 features) + jauge de risque + analyse des facteurs · mode batch CSV |
| 📊 Modèles | Runs MLflow, comparaison métriques (graphes par métrique), meilleur run |
| 🖼️ Artefacts | Matrice de confusion + classification report par run MLflow, téléchargement |
| ⚗️ Entraînement | Lancer Personnalisé / GridSearchCV / Optuna avec logs temps réel · historique des runs |
| 🔍 Exploration | Distributions par feature, corrélation avec l'attrition, stats descriptives |

### Sélecteur de modèle

Le sidebar liste tous les modèles entraînés disponibles (avec leurs métriques ROC-AUC / F1).  
Un clic sur **↺ Charger ce modèle** switche instantanément le modèle actif dans l'API sans redémarrage.

---

## Orchestration Airflow

Airflow orchestre le ré-entraînement planifié et les prédictions quotidiennes. Il tourne dans son propre conteneur Docker (profil `airflow`), partageant les volumes `model_vol` et `mlruns_vol` avec la stack principale.

```
┌─────────────────────────────────────────────────┐
│  Airflow UI — http://localhost:8080             │
│                                                 │
│  DAG model_retraining  (lundi 3h)               │
│    prepare_data → train → check_quality         │
│                                                 │
│  DAG daily_predictions  (tous les jours 10h)    │
│    send_predictions → log_summary               │
└─────────────────────────────────────────────────┘
```

### DAGs

| DAG | Schedule | Tâches | Description |
|---|---|---|---|
| `model_retraining` | `0 3 * * 1` (lundi 3h) | prepare_data → train → check_quality | Ré-entraîne la baseline, valide F1 ≥ 0.35 via XCom |
| `daily_predictions` | `0 10 * * *` (10h/jour) | send_predictions → log_summary | Envoie 20 employés à `/predict`, journalise le taux d'attrition |

### Démarrage

```bash
# Airflow seul (retrain DAG uniquement)
make airflow

# Stack complète + Airflow (predictions DAG aussi)
make airflow-stack
```

Login : **airflow / airflow** — UI : **http://localhost:8080**

### Tester le garde-fou qualité

Dans l'UI Airflow → **Admin → Variables** → ajouter `QUALITY_THRESHOLD = 0.99`,  
puis re-déclencher le DAG → `check_quality` échoue, le modèle n'est pas livré.

---

### Entraînement depuis le dashboard (mode local)

L'onglet **⚗️ Entraînement** propose trois modes :

- **🎯 Personnalisé** — choisir parmi les 5 modèles, configurer ses hyperparamètres, lancer avec logs live
- **🔬 GridSearchCV** — comparer les 5 modèles, choisir le nombre de folds CV
- **⚡ Optuna TPE** — optimiser le modèle de son choix avec le nombre de trials voulu

> En mode Docker, seules les commandes à lancer sont affichées (le training s'exécute dans le conteneur `trainer`).

---

## API — Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API |
| `GET` | `/model-info` | Modèle actif : type, display_name, métriques, params |
| `GET` | `/models` | Liste tous les modèles disponibles dans `models/` |
| `POST` | `/models/switch` | Change le modèle actif sans redémarrage |
| `POST` | `/predict` | Prédiction d'attrition (30 features → 0/1 + probabilité) |

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

> Champion actuel : **LogisticRegression** (C=0.1, class_weight=balanced).  
> Le MLP PyTorch obtient le meilleur F1 (0.53) et Recall (0.62) — bon profil pour le contexte RH où rater un départ coûte cher.

---

## Structure

```
.
├── data/
│   └── ibm_hr_attrition.csv            # Dataset IBM (non versionné)
├── models/
│   ├── model.joblib                    # Champion courant (non versionné)
│   ├── model.meta.json                 # Métadonnées du champion
│   └── model_<label>_<ts>.joblib       # Snapshots horodatés par entraînement
├── docker/
│   ├── Dockerfile.api                  # Image API slim (~500 Mo)
│   ├── Dockerfile.trainer              # Image entraînement (torch CPU, ~2 Go)
│   ├── Dockerfile.streamlit            # Image dashboard Streamlit (~300 Mo)
│   ├── Dockerfile.airflow              # Image Airflow 2.9 + dépendances ML
│   ├── requirements-api.txt
│   ├── requirements-train.txt
│   ├── requirements-streamlit.txt
│   ├── requirements-airflow.txt
│   ├── train-entrypoint.sh             # Dispatch baseline / full / optuna
│   └── airflow-entrypoint.sh           # Init DB + user + standalone
├── docker-compose.yml                  # trainer → api → streamlit (+ airflow + mlflow-ui en profils)
├── dags/
│   ├── retrain_dag.py                  # DAG ré-entraînement (lundi 3h, XCom F1)
│   └── predictions_dag.py              # DAG prédictions quotidiennes (10h/jour)
├── streamlit/
│   └── app.py                          # Dashboard Streamlit 6 onglets
├── mlproject/
│   ├── config.py                       # DataConfig, MLflowConfig, DeployStage
│   ├── data.py                         # Chargement & split 80/20 stratifié
│   ├── features.py                     # Préprocessing (StandardScaler + OneHotEncoder)
│   ├── evaluation.py                   # Métriques (f1, roc_auc, precision, recall)
│   ├── tracking.py                     # Interface centralisée MLflow
│   ├── model_registry.py               # Sauvegarde modèles + .meta.json
│   ├── models.py                       # TorchMLPClassifier compatible sklearn Pipeline
│   ├── train.py                        # Baseline LogisticRegression
│   ├── train_models.py                 # GridSearchCV 5 modèles + Model Registry
│   ├── train_optuna.py                 # Optimisation Optuna TPE
│   ├── train_custom.py                 # Entraînement personnalisé (modèle + params libres)
│   ├── evaluate.py                     # Évaluation du champion sur le test set
│   └── api.py                          # FastAPI : /health /predict /models /models/switch
├── scripts/
│   └── predict_client.py               # Client HTTP de test
├── tests/
│   ├── conftest.py
│   ├── generate_synthetic_data.py
│   ├── check_pipeline.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_evaluation.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       ├── ci.yml                      # CI : ruff + pytest + smoke test pipeline
│       └── cd.yml                      # CD : build & push image API sur GHCR
├── .dockerignore
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
| S19 | Dockerfile API + CD GitHub Actions → GHCR | ✅ |
| S14 | Docker Compose (trainer → api → streamlit) — pipeline from scratch | ✅ |
| S14bis | Dashboard Streamlit (6 onglets, entraînement live, sélecteur de modèle) | ✅ |
| S17 | DAG Airflow (retrain + predictions, XCom, garde-fou qualité) | ✅ |
