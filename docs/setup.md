# Installation

## Prérequis

- **Python 3.13+**
- **uv** (gestionnaire de paquets) — [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- **Git**

## Installation de uv

=== "macOS / Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows (PowerShell)"
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

## Cloner et installer

```bash
git clone <url-du-repo>
cd td_orchestrationML_EthanVIEVILLE
make install
```

`make install` exécute `uv sync --extra dev` : il crée un virtualenv `.venv/`, installe toutes les dépendances (scikit-learn, MLflow, FastAPI, PyTorch, Optuna…) et les outils de développement (pytest, ruff, mypy, mkdocs).

## Dataset

Le CSV réel n'est pas versionné (trop lourd, données sensibles). Deux options :

**Option A — Dataset réel (recommandé)**

Télécharger `WA_Fn-UseC_-HR-Employee-Attrition.csv` depuis [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) et le placer ici :

```
data/ibm_hr_attrition.csv
```

**Option B — Dataset synthétique (CI / onboarding rapide)**

```bash
uv run python tests/generate_synthetic_data.py
```

Génère 400 lignes avec le même schéma que le dataset réel. Suffisant pour tester tout le pipeline.

## Configuration (.env)

Copier le template et ajuster si besoin :

```bash
cp .env.example .env
```

Variables disponibles :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlruns.db` | Backend MLflow |
| `MLFLOW_EXPERIMENT` | `ibm-hr-attrition` | Nom de l'expérience |
| `MODEL_NAME` | `attrition-classifier` | Nom dans le Registry |
| `DEPLOY_STAGE` | `challenger` | Alias initial du modèle |
| `API_URL` | `http://127.0.0.1:8000` | URL de l'API (client de test) |

## Vérification

```bash
make doctor          # vérifie uv + Python
make check_pipeline  # valide le chargement des données et le préprocessing
```
