# Pipeline MLOps — Prédiction de l'Attrition Employée (IBM HR Analytics)

![CI](https://github.com/EthanVIEVILLE/td_orchestrationML_EthanVIEVILLE/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/EthanVIEVILLE/td_orchestrationML_EthanVIEVILLE/actions/workflows/cd.yml/badge.svg)

**Auteur** : VIEVILLE Ethan

Pipeline MLOps complète de classification binaire pour prédire si un employé va quitter l'entreprise — du chargement des données jusqu'au déploiement automatisé sur Oracle Cloud, avec orchestration Airflow, dashboard Streamlit et tracking MLflow.

---

## Problématique

> **Un employé va-t-il quitter l'entreprise ?** (`Attrition = Yes / No`)

Anticiper les départs permet aux RH de prioriser les actions de rétention sur les profils à risque, réduisant les coûts de recrutement et la perte de compétences.

| Classe | Signification |
|---|---|
| `0` | L'employé **reste** |
| `1` | L'employé **quitte** (attrition) |

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- **1 470 employés**, 31 colonnes exploitables (4 colonnes constantes supprimées)
- **Target** : `Attrition` — 83.9 % No / 16.1 % Yes (dataset déséquilibré → `class_weight="balanced"`)
- **23 features numériques** : Age, MonthlyIncome, YearsAtCompany, JobSatisfaction, etc.
- **7 features catégorielles** : Department, JobRole, OverTime, MaritalStatus, etc.

---

## Architecture MLOps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GitHub (code source + CI/CD)                                               │
│    push → CI (tests + smoke test) → CD (build image → déploiement VPS)     │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │ git fetch + make start
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Oracle Cloud VPS — Docker Compose                                          │
│                                                                             │
│  ┌──────────┐   model.joblib    ┌──────────┐   HTTP     ┌───────────────┐  │
│  │ trainer  │ ───────────────► │   api    │ ◄───────── │  streamlit    │  │
│  │ (oneshot)│   mlruns_vol      │  :8000   │            │    :8501      │  │
│  └──────────┘                  └──────────┘            └───────────────┘  │
│                                      ▲                                      │
│  ┌──────────────────────────────┐    │                                      │
│  │  airflow  :8080              │    │ modèles / mlruns                     │
│  │  ├─ model_retraining DAG     │────┘                                      │
│  │  └─ daily_predictions DAG   │                                           │
│  └──────────────────────────────┘                                          │
│                                                                             │
│  ┌──────────┐                                                               │
│  │ mlflow-ui│  :5000  (profil monitoring)                                  │
│  └──────────┘                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Volumes Docker partagés

| Volume | Contenu | Producteurs | Consommateurs |
|---|---|---|---|
| `model_vol` | `model.joblib` + snapshots `.joblib` / `.meta.json` | trainer, airflow | api (ro), airflow |
| `mlruns_vol` | base SQLite MLflow + artefacts | trainer, airflow | streamlit (ro), mlflow-ui, airflow |
| `airflow_logs` | logs Airflow + **SQLite Airflow** persistant | airflow | airflow (persistence inter-redémarrages) |

---

## Services Docker

| Service | Image | Rôle | Port | Profil |
|---|---|---|---|---|
| `trainer` | `Dockerfile.trainer` | Entraîne le modèle au démarrage et s'arrête (exit 0) | — | toujours |
| `api` | `Dockerfile.api` | FastAPI — `/predict`, `/models`, `/models/switch` | **8000** | toujours |
| `streamlit` | `Dockerfile.streamlit` | Dashboard MLOps 7 onglets | **8501** | toujours |
| `airflow` | `Dockerfile.airflow` | Orchestration DAGs (scheduler + webserver) | **8080** | `airflow` |
| `mlflow-ui` | `python:3.13-slim` | Interface de visualisation MLflow | **5000** | `monitoring` |

### Dépendances de démarrage

```
trainer ──► (service_completed_successfully)
              └──► api ──► (service_healthy)
                            └──► streamlit
```

L'API attend que le trainer ait terminé avec succès. Streamlit attend que l'API réponde sur `/health`.

---

## Installation locale

### Prérequis

- [uv](https://docs.astral.sh/uv/) (gestionnaire Python)
- Docker + Docker Compose

```bash
# Cloner le dépôt
git clone https://github.com/EthanVIEVILLE/td_orchestrationML_EthanVIEVILLE.git
cd td_orchestrationML_EthanVIEVILLE

# Installer les dépendances Python
make install
```

---

## Commandes Makefile

### Environnement

```bash
make install          # installe les dépendances via uv sync
make doctor           # vérifie que l'environnement est prêt (uv + Python)
make ci               # CI locale complète : ruff + pytest + pipeline end-to-end
```

### Entraînement (local, sans Docker)

```bash
make train                              # baseline LogisticRegression (C=1.0, max_iter=1000)
make train C=0.1 MAX_ITER=500           # paramètres personnalisés

make train_models                       # GridSearchCV — 5 modèles, CV=5
make train_models CV=2                  # CV réduit pour aller plus vite

make train_optuna                       # Optuna TPE — LightGBM, 50 trials
make train_optuna MODEL=XGBoost N_TRIALS=100

make evaluate                           # évalue model.joblib sur le test set
make evaluate SOURCE=registry           # évalue le champion @champion MLflow Registry
```

### API et dashboard (local, sans Docker)

```bash
make api              # FastAPI sur http://127.0.0.1:8000/docs (mode dev, reload auto)
make streamlit        # Dashboard sur http://127.0.0.1:8501
make predict_client   # envoie 3 requêtes de test à l'API locale
make mlflow_ui        # MLflow UI sur http://127.0.0.1:5000
```

### Tests

```bash
make test             # pytest (25 tests unitaires)
make ci               # ruff + pytest + pipeline complet (équivalent GitHub Actions)
```

### Docker Compose — pipeline complète

```bash
# ── Démarrage ─────────────────────────────────────────────────────────────────
make compose-up           # baseline LogisticRegression (~30 s) → API → Streamlit
make compose-full         # GridSearchCV 5 modèles CV=2 (~10 min) → API → Streamlit
make compose-optuna       # Optuna LightGBM 20 trials (~5 min) → API → Streamlit
make compose-monitoring   # + MLflow UI sur http://127.0.0.1:5000

# ── Arrêt et nettoyage ────────────────────────────────────────────────────────
make compose-down         # arrête les conteneurs (conserve les volumes)
make compose-clean        # arrête + supprime les volumes (repart from scratch)
make compose-logs         # logs en temps réel
```

### Stack complète (API + Airflow + MLflow UI)

```bash
make start            # toute la stack en arrière-plan (baseline + airflow + monitoring)
make stop             # arrête la stack (conserve les volumes)
make restart          # stop + start
```

URLs après `make start` :
- API : http://localhost:8000/docs
- Dashboard : http://localhost:8501
- Airflow UI : http://localhost:8080 *(airflow / airflow)*
- MLflow UI : http://localhost:5000

### Airflow — Orchestration

```bash
make airflow              # Airflow seul (sans API ni Streamlit)
make airflow-stack        # stack complète + Airflow (nécessaire pour daily_predictions)
make airflow-trigger      # déclenche model_retraining manuellement
make airflow-logs         # logs Airflow en temps réel
make airflow-down         # arrête Airflow (conserve les données)
make airflow-clean        # repart from scratch (supprime le volume airflow_logs)
```

---

## Pipeline Docker Compose pas à pas

### `make compose-up` (ou `make start`)

1. **trainer** démarre, génère 400 lignes synthétiques si `data/` est vide, entraîne LogisticRegression, écrit `model.joblib` dans `model_vol`, puis **s'arrête** (exit 0).
2. **api** démarre dès que trainer est terminé, charge `model.joblib`, expose les endpoints sur `:8000`.
3. **streamlit** démarre dès que l'API répond sur `/health`, se connecte à `http://api:8000`.
4. **airflow** *(profil `airflow`)* initialise sa DB SQLite dans le volume persistant `airflow_logs`, crée l'utilisateur admin, démarre scheduler + webserver.
5. **mlflow-ui** *(profil `monitoring`)* monte `mlruns_vol` en lecture et expose l'interface sur `:5000`.

### Modes d'entraînement du trainer

| Commande | `TRAIN_MODE` | Algorithme | Durée estimée |
|---|---|---|---|
| `make compose-up` / `make start` | `baseline` | LogisticRegression (max_iter=300) | ~30 s |
| `make compose-full` | `full` | GridSearchCV — 5 modèles, CV=2 | ~10 min |
| `make compose-optuna` | `optuna` | Optuna TPE — LightGBM, 20 trials | ~5 min |

---

## Orchestration Airflow

L'orchestration est gérée par Apache Airflow, qui tourne dans son propre conteneur (profil `airflow`). Il partage `model_vol` et `mlruns_vol` avec les autres services.

**Accès UI** : http://localhost:8080 — login : `airflow / airflow`

**Persistance** : la base SQLite d'Airflow est stockée dans le volume `airflow_logs` (`/opt/airflow/logs/airflow.db`), ce qui préserve l'historique des runs entre les redémarrages.

### DAG `model_retraining`

```
prepare_data → train → check_quality
```

| Propriété | Valeur |
|---|---|
| Schedule | `0 */5 * * *` (toutes les 5h) |
| Tâche 1 `prepare_data` | Vérifie / génère le dataset IBM HR |
| Tâche 2 `train` | Lance `train_models(cv=2)` — 5 modèles GridSearchCV, sélectionne le champion par ROC-AUC, pousse les métriques en XCom, tague les runs MLflow avec `triggered_by=airflow` |
| Tâche 3 `check_quality` | Lit le F1 du champion en XCom, échoue si `F1 < QUALITY_THRESHOLD` (default 0.35) — garde-fou qualité avant livraison |

**Trigger manuel** :
```bash
make airflow-trigger
# ou
docker compose --profile airflow exec airflow airflow dags trigger model_retraining
```

### DAG `daily_predictions`

```
send_predictions → log_summary
```

| Propriété | Valeur |
|---|---|
| Schedule | `30 */5 * * *` (30 min après le retrain) |
| Tâche 1 `send_predictions` | Échantillonne 20 employés du dataset, envoie chaque ligne à `POST /predict`, pousse `n_predictions` + `n_attrition` + `attrition_rate` en XCom |
| Tâche 2 `log_summary` | Lit les XCom, affiche le bilan, alerte si taux d'attrition > 30 % |

**Trigger manuel** :
```bash
docker compose --profile airflow exec airflow airflow dags trigger daily_predictions
```

### Tester le garde-fou qualité

Dans l'UI Airflow → **Admin → Variables** → ajouter `QUALITY_THRESHOLD = 0.99`, puis re-déclencher `model_retraining` → la tâche `check_quality` échoue intentionnellement, le modèle n'est pas livré.

---

## API FastAPI

Documentation interactive : http://localhost:8000/docs

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API (uptime, modèle chargé) |
| `GET` | `/model-info` | Modèle actif : type, display_name, métriques, params |
| `GET` | `/models` | Liste tous les modèles disponibles dans `models/` (avec métriques) |
| `POST` | `/models/switch` | Charge un modèle différent en mémoire sans redémarrage |
| `POST` | `/predict` | Prédiction d'attrition (30 features → `prediction` 0/1 + `probability`) |

### Exemple de requête `/predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "MonthlyIncome": 5000,
    "OverTime": "Yes",
    "JobSatisfaction": 1,
    "YearsAtCompany": 8,
    ...
  }'
```

```json
{
  "prediction": 1,
  "probability": 0.73,
  "model_type": "XGBClassifier"
}
```

### Sélecteur de modèle (switch live)

L'endpoint `/models/switch` permet de changer le modèle actif **sans redémarrer l'API**. Chaque nouveau modèle entraîné (par le trainer ou Airflow) est automatiquement visible dans `GET /models` via son fichier `.meta.json`.

---

## Dashboard Streamlit

Accessible sur http://localhost:8501 — 7 onglets :

| Onglet | Contenu |
|---|---|
| 🏠 **Contexte** | Problématique IBM HR, statistiques dataset, facteurs de risque, schéma architecture MLOps |
| 🔮 **Prédiction** | Formulaire employé (30 features) + jauge de risque + analyse des facteurs clés · mode batch CSV |
| 📊 **Modèles** | Runs MLflow, comparaison métriques (graphes par métrique ROC-AUC / F1 / Recall / Precision), meilleur run |
| 🖼️ **Artefacts** | Matrice de confusion + classification report par run MLflow, téléchargement |
| ⚗️ **Entraînement** | Lancer un entraînement personnalisé / GridSearchCV / Optuna avec logs en temps réel · historique des runs |
| 🔍 **Exploration** | Distributions par feature, corrélation avec l'attrition, statistiques descriptives |
| 🔄 **Orchestration** | Statut des DAGs Airflow, historique des runs model_retraining et daily_predictions, graphique d'évolution ROC-AUC / F1 depuis MLflow (runs manuels et Airflow distingués) |

### Sidebar — Sélecteur de modèle

Le sidebar liste tous les modèles entraînés disponibles (avec métriques ROC-AUC / F1).  
**↺ Charger ce modèle** switche instantanément le modèle actif dans l'API sans redémarrage.

---

## Tracking MLflow

Chaque entraînement (baseline, GridSearchCV, Optuna, Airflow) logue automatiquement :
- **Paramètres** : hyperparamètres du modèle sélectionné
- **Métriques** : `f1`, `roc_auc`, `precision`, `recall`
- **Artefacts** : matrice de confusion (PNG), classification report (TXT), pipeline sérialisé
- **Tags** : `triggered_by=airflow` + `airflow_run_id` pour les runs orchestrés

Les runs Airflow sont identifiés visuellement dans le dashboard Orchestration (🔄 vs 🖐 Manuel).

**Accès UI** : http://localhost:5000 *(nécessite le profil `monitoring`)*

---

## CI/CD GitHub Actions

### CI — `.github/workflows/ci.yml`

Déclenché sur chaque push et pull request vers `main`.

```
Job 1 : unit-tests
  └─ ruff check (style + imports)
  └─ pytest (25 tests unitaires)

Job 2 : pipeline-smoke-test  (nécessite job 1)
  └─ génère dataset synthétique
  └─ vérifie le pipeline de données
  └─ entraîne LogisticRegression (max-iter=300)
  └─ évalue le champion
  └─ démarre l'API + health check
  └─ client de prédiction (3 requêtes /predict)
```

### CD — `.github/workflows/cd.yml`

Déclenché sur push vers `main` ou tag `vX.Y.Z`.

```
Job 1 : build-push
  └─ build image Docker API (Dockerfile.api)
  └─ push sur GHCR (ghcr.io/<owner>/mlproject-api)
      Tags : latest + sha-<commit> + vX.Y.Z (sur tag sémantique)

Job 2 : deploy  (nécessite job 1, main uniquement)
  └─ SSH sur Oracle Cloud VPS
  └─ git fetch + git reset --hard FETCH_HEAD  (évite les conflits de merge)
  └─ make start  (rebuild + redémarrage de la stack complète)
```

**Secrets GitHub requis** :

| Secret | Description |
|---|---|
| `ORACLE_HOST` | IP publique du VPS |
| `ORACLE_USER` | Utilisateur SSH (ex: `ubuntu`) |
| `ORACLE_SSH_KEY` | Clé privée SSH (contenu du fichier `.pem`) |
| `GH_PAT` | Personal Access Token pour git fetch authentifié |

### Image Docker API sur GHCR

```bash
docker pull ghcr.io/ethanvieville/td_orchestrationml_ethanvieville/mlproject-api:latest

docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  ghcr.io/ethanvieville/td_orchestrationml_ethanvieville/mlproject-api:latest
```

> Le modèle (`model.joblib`) est monté en volume — il n'est pas inclus dans l'image. L'API retourne 503 sur `/predict` jusqu'à ce qu'un modèle soit disponible.

---

## Résultats — Comparaison multi-modèles

**Critère de sélection** : ROC-AUC (seuil-indépendant, robuste aux classes déséquilibrées)  
**Dataset** : IBM HR Attrition, 80/20 train/test stratifié, `class_weight="balanced"`

| Modèle | ROC-AUC | F1 (Yes) | Recall | Precision |
|---|---|---|---|---|
| **LogisticRegression** | **0.8124** | 0.4885 | 0.6809 | 0.3810 |
| RandomForest | 0.7955 | 0.4771 | 0.5532 | 0.4194 |
| LightGBM | 0.7735 | 0.4110 | 0.3191 | 0.5769 |
| MLP (PyTorch) | 0.7678 | 0.5321 | 0.6170 | 0.4688 |
| XGBoost | 0.7600 | 0.4421 | 0.4468 | 0.4375 |

> **Champion actuel** : LogisticRegression (C=0.1, class_weight=balanced) — meilleur ROC-AUC.  
> Le MLP PyTorch obtient le meilleur F1 (0.53) et Recall (0.62) — profil intéressant dans un contexte RH où rater un départ coûte cher.

---

## Structure du projet

```
.
├── data/
│   └── ibm_hr_attrition.csv              # Dataset IBM HR (non versionné)
│
├── models/                               # Modèles sauvegardés (non versionnés)
│   ├── model.joblib                      # Champion courant (chargé par l'API au démarrage)
│   ├── model.meta.json                   # Métadonnées du champion
│   └── model_<label>_<timestamp>.joblib  # Snapshots horodatés par entraînement
│
├── dags/
│   ├── retrain_dag.py                    # DAG ré-entraînement (5 modèles, toutes les 5h)
│   └── predictions_dag.py               # DAG prédictions (20 employés, toutes les 5h30)
│
├── docker/
│   ├── Dockerfile.api                    # Image serving (Python 3.13-slim, ~200 Mo)
│   ├── Dockerfile.trainer                # Image entraînement (avec torch CPU, ~2 Go)
│   ├── Dockerfile.streamlit              # Image dashboard (~300 Mo)
│   ├── Dockerfile.airflow                # Airflow 2.9 + dépendances ML
│   ├── requirements-api.txt              # FastAPI, sklearn, xgboost, lightgbm
│   ├── requirements-train.txt            # + torch, mlflow, optuna
│   ├── requirements-streamlit.txt        # + streamlit, plotly
│   ├── requirements-airflow.txt          # + airflow, torch, mlflow
│   ├── train-entrypoint.sh               # Dispatch baseline / full / optuna selon TRAIN_MODE
│   ├── airflow-entrypoint.sh             # Init DB + user + permissions volumes + standalone
│   └── mlflow_serve.py                   # Script gunicorn pour mlflow-ui
│
├── docker-compose.yml                    # Orchestration des 5 services (+ profils airflow/monitoring)
├── Makefile                              # ~40 commandes documentées
│
├── mlproject/
│   ├── config.py                         # DataConfig, MLflowConfig, DeployStage, MODEL_DIR
│   ├── data.py                           # Chargement CSV + split 80/20 stratifié
│   ├── features.py                       # Préprocessing (StandardScaler + OneHotEncoder)
│   ├── evaluation.py                     # Calcul métriques (f1, roc_auc, precision, recall)
│   ├── tracking.py                       # Interface MLflow centralisée (setup, log_*, log_model)
│   ├── model_registry.py                 # Sauvegarde .joblib + .meta.json horodatés
│   ├── models.py                         # TorchMLPClassifier (sklearn-compatible, PyTorch)
│   ├── train.py                          # Baseline LogisticRegression
│   ├── train_models.py                   # GridSearchCV — 5 familles de modèles
│   ├── train_optuna.py                   # Optimisation Optuna TPE (LightGBM par défaut)
│   ├── train_custom.py                   # Entraînement personnalisé (modèle + params libres)
│   ├── evaluate.py                       # Évaluation du champion sur le test set
│   └── api.py                            # FastAPI : endpoints + lifespan + switch modèle
│
├── streamlit/
│   └── app.py                            # Dashboard 7 onglets (Streamlit)
│
├── scripts/
│   └── predict_client.py                 # Client HTTP de test pour l'API
│
├── tests/
│   ├── conftest.py                       # Fixtures pytest (dataset synthétique auto-généré)
│   ├── generate_synthetic_data.py        # Génère 400 lignes IBM HR synthétiques
│   ├── check_pipeline.py                 # Vérifie le pipeline données → features
│   ├── test_data.py                      # Tests de chargement et split
│   ├── test_features.py                  # Tests du préprocessing
│   ├── test_evaluation.py                # Tests des métriques
│   └── test_api.py                       # Tests des endpoints FastAPI (client de test)
│
└── .github/
    └── workflows/
        ├── ci.yml                        # CI : ruff + pytest + smoke test pipeline
        └── cd.yml                        # CD : build image → GHCR → déploiement VPS SSH
```

---

## Avancement

| Séance | Objectif | Statut |
|---|---|---|
| S0 | Configuration dataset IBM HR + pipeline de données | ✅ |
| S5 | Baseline LogisticRegression + MLflow tracking | ✅ |
| S6/S7 | Comparaison multi-modèles (GridSearchCV 5 algos) + MLflow Registry | ✅ |
| S7bis | Optimisation Optuna TPE + évaluation standalone | ✅ |
| S12 | API FastAPI (`/predict`, `/health`, `/model-info`, `/models`, `/models/switch`) | ✅ |
| S15 | Tests pytest (25 tests) + CI GitHub Actions | ✅ |
| S19 | Dockerfile API + CD GitHub Actions → GHCR + déploiement Oracle Cloud | ✅ |
| S14 | Docker Compose (trainer → api → streamlit) — pipeline from scratch | ✅ |
| S14bis | Dashboard Streamlit 7 onglets (entraînement live, sélecteur de modèle, orchestration) | ✅ |
| S17 | DAG Airflow (retrain 5 modèles + predictions, XCom, garde-fou qualité, persistance DB) | ✅ |
