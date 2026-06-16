# Référence Makefile

Toutes les commandes disponibles — `make help` pour un résumé rapide.

## Setup

| Commande | Description |
|----------|-------------|
| `make install` | Installe toutes les dépendances (`uv sync --extra dev`) |
| `make doctor` | Vérifie uv, Python et le virtualenv |
| `make reset-env` | Recrée `.venv` et `uv.lock` from scratch |

## Pipeline ML

| Commande | Description | Paramètres |
|----------|-------------|------------|
| `make check_pipeline` | Valide le chargement des données et le préprocessing | — |
| `make train` | Baseline LogisticRegression + MLflow | `C=1.0` `MAX_ITER=1000` |
| `make train_models` | 5 modèles via GridSearchCV + MLflow Registry | `CV=5` |
| `make train_optuna` | Optimisation Optuna TPE + MLflow | `MODEL=LightGBM` `N_TRIALS=50` `CV=5` |
| `make evaluate` | Évalue le champion sauvegardé | `SOURCE=file\|registry` |

**Exemples :**

```bash
make train C=0.01 MAX_ITER=2000
make train_models CV=3
make train_optuna MODEL=XGBoost N_TRIALS=100
make evaluate SOURCE=registry
```

## API

| Commande | Description | Paramètres |
|----------|-------------|------------|
| `make api` | Lance FastAPI en mode dev (reload) | `PORT=8000` |
| `make predict_client` | Envoie 3 requêtes de test à l'API | `PORT=8000` |

## Tests & CI

| Commande | Description |
|----------|-------------|
| `make test` | Tests unitaires pytest |
| `make pipeline` | Pipeline complet : données → train → evaluate |
| `make ci` | CI locale : ruff + pytest + pipeline |

## MLflow

| Commande | Description |
|----------|-------------|
| `make mlflow_ui` | Interface MLflow → http://127.0.0.1:5000 |

## Documentation

| Commande | Description |
|----------|-------------|
| `make docs-serve` | Documentation locale → http://127.0.0.1:8001 |
| `make docs-build` | Génère le site statique dans `site/` |

---

## Variables globales

Toutes ces variables peuvent être surchargées depuis la ligne de commande :

```bash
make train_optuna MODEL=RandomForest N_TRIALS=30 CV=3
```

| Variable | Défaut | Utilisée par |
|----------|--------|-------------|
| `C` | `1.0` | `train` |
| `MAX_ITER` | `1000` | `train` |
| `CV` | `5` | `train_models`, `train_optuna` |
| `MODEL` | `LightGBM` | `train_optuna` |
| `N_TRIALS` | `50` | `train_optuna` |
| `SOURCE` | `file` | `evaluate` |
| `PORT` | `8000` | `api`, `predict_client` |
