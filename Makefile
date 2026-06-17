# ==============================================================================
# IBM HR Attrition — Makefile
# Environnement géré par uv (Python 3.13) depuis pyproject.toml
# ==============================================================================

SHELL    := /bin/sh
PYTHON   := uv run python
RUN      := uv run

# Paramètres overridables depuis la ligne de commande (ex: make train C=0.1)
C        ?= 1.0
MAX_ITER ?= 1000
CV       ?= 5
MODEL    ?= LightGBM
N_TRIALS ?= 50
SOURCE   ?= file
PORT     ?= 8000

IMAGE    ?= mlproject-api
TAG      ?= dev

YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help check-uv check-venv install sync lock reset-env doctor \
        train train_models train_optuna evaluate check_pipeline mlflow_ui \
        api predict_client pipeline ci test docs-serve docs-build \
        docker-build docker-run docker-push \
        compose-up compose-full compose-optuna compose-down compose-clean compose-monitoring compose-logs

# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ==============================================================================
# Setup
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d .venv || { \
		echo "$(RED)[ERREUR] Virtualenv manquant$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

install: check-uv ## Installe les dépendances (uv sync)
	@echo "$(YELLOW)>> Installation des dépendances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dépendances installées$(RESET)"

sync: install ## Alias de install

lock: check-uv ## Régénère uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Génération du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock généré$(RESET)"

reset-env: check-uv ## Réinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Réinitialisation...$(RESET)"
	rm -rf .venv uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recréé$(RESET)"

doctor: check-uv check-venv ## Vérifie l'environnement
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement prêt$(RESET)"

# ==============================================================================
# Pipeline ML
# ==============================================================================

check_pipeline: check-venv ## Vérifie que le pipeline de données fonctionne
	@echo "$(YELLOW)>> Vérification du pipeline...$(RESET)"
	$(PYTHON) tests/check_pipeline.py

train: check-venv ## Entraîne la baseline LogisticRegression (C=.. MAX_ITER=..)
	@echo "$(YELLOW)>> Entraînement baseline...$(RESET)"
	$(PYTHON) -m mlproject.train --c $(C) --max-iter $(MAX_ITER)

train_models: check-venv ## Compare 5 modèles via GridSearchCV + MLflow (CV=..)
	@echo "$(YELLOW)>> Comparaison multi-modèles...$(RESET)"
	$(PYTHON) -m mlproject.train_models --cv $(CV)

train_optuna: check-venv ## Optimisation Optuna TPE (MODEL=LightGBM N_TRIALS=50 CV=5)
	@echo "$(YELLOW)>> Optimisation Optuna ($(MODEL), $(N_TRIALS) trials)...$(RESET)"
	$(PYTHON) -m mlproject.train_optuna --model $(MODEL) --n-trials $(N_TRIALS) --cv $(CV)

evaluate: check-venv ## Évalue le champion sauvegardé sur le test set (SOURCE=file|registry)
	@echo "$(YELLOW)>> Évaluation du modèle (source=$(SOURCE))...$(RESET)"
	$(PYTHON) -m mlproject.evaluate --source $(SOURCE)

# ==============================================================================
# Tests & CI
# ==============================================================================

test: check-venv ## Exécute les tests pytest
	@echo "$(YELLOW)>> Tests unitaires...$(RESET)"
	$(RUN) pytest tests/ -v --tb=short

pipeline: check-venv ## Pipeline complet : données -> train baseline -> évaluation
	@echo "$(YELLOW)>> Pipeline complet...$(RESET)"
	$(PYTHON) tests/generate_synthetic_data.py
	$(PYTHON) tests/check_pipeline.py
	$(PYTHON) -m mlproject.train --max-iter 300
	$(PYTHON) -m mlproject.evaluate
	@echo "$(GREEN)[OK] Pipeline complet$(RESET)"

ci: check-venv ## CI locale : tests + pipeline complet (équivalent GitHub Actions)
	@echo "$(YELLOW)>> CI locale (tests + pipeline)...$(RESET)"
	$(RUN) ruff check mlproject/ tests/ scripts/
	$(RUN) pytest tests/ -v --tb=short
	$(MAKE) pipeline
	@echo "$(GREEN)[OK] CI locale réussie$(RESET)"

# ==============================================================================
# API
# ==============================================================================

api: check-venv ## Lance l'API FastAPI en mode dev (PORT=8000)
	@echo "$(YELLOW)>> Démarrage de l'API FastAPI...$(RESET)"
	@echo "   Docs  : http://127.0.0.1:$(PORT)/docs"
	@echo "   Santé : http://127.0.0.1:$(PORT)/health"
	$(RUN) uvicorn mlproject.api:app --reload --host 127.0.0.1 --port $(PORT)

predict_client: check-venv ## Envoie des requêtes de test à l'API locale (PORT=8000)
	@echo "$(YELLOW)>> Test client -> http://127.0.0.1:$(PORT)$(RESET)"
	$(PYTHON) scripts/predict_client.py --url http://127.0.0.1:$(PORT)

# ==============================================================================
# MLflow
# ==============================================================================

docs-serve: check-venv ## Sert la documentation en local sur http://127.0.0.1:8001
	@echo "$(YELLOW)>> Documentation -> http://127.0.0.1:8001$(RESET)"
	$(RUN) mkdocs serve --dev-addr 127.0.0.1:8001

docs-build: check-venv ## Génère la documentation statique dans site/
	@echo "$(YELLOW)>> Build de la documentation...$(RESET)"
	$(RUN) mkdocs build
	@echo "$(GREEN)[OK] Documentation générée dans site/$(RESET)"

# ==============================================================================
# MLflow
# ==============================================================================

# ==============================================================================
# Docker
# ==============================================================================

docker-build: ## Construit l'image Docker de l'API (IMAGE=mlproject-api TAG=dev)
	@echo "$(YELLOW)>> Build de l'image Docker $(IMAGE):$(TAG)...$(RESET)"
	docker build -f docker/Dockerfile.api -t $(IMAGE):$(TAG) .
	@echo "$(GREEN)[OK] Image construite : $(IMAGE):$(TAG)$(RESET)"

docker-run: ## Lance l'API dans un conteneur (PORT=8000, modèle local monté en lecture)
	@echo "$(YELLOW)>> Démarrage du conteneur $(IMAGE):$(TAG)...$(RESET)"
	@echo "   Docs  : http://127.0.0.1:$(PORT)/docs"
	@echo "   Santé : http://127.0.0.1:$(PORT)/health"
	docker run --rm \
		-p $(PORT):8000 \
		-v "$(CURDIR)/models:/app/models:ro" \
		-e MODEL_VERSION=$(TAG) \
		$(IMAGE):$(TAG)

docker-push: ## Pousse l'image sur le registre (IMAGE=ghcr.io/<owner>/mlproject-api TAG=...)
	@echo "$(YELLOW)>> Push de $(IMAGE):$(TAG)...$(RESET)"
	docker push $(IMAGE):$(TAG)
	@echo "$(GREEN)[OK] Image publiée$(RESET)"

# ==============================================================================
# Docker Compose — pipeline complète
# ==============================================================================

compose-up: ## Pipeline baseline : LogisticRegression (~30s) → API
	@echo "$(YELLOW)>> Pipeline Docker Compose — mode baseline$(RESET)"
	@echo "   API disponible sur : http://127.0.0.1:8000/docs"
	TRAIN_MODE=baseline docker compose up --build

compose-full: ## Pipeline GridSearchCV : 5 modèles CV=2 (~10 min) → API
	@echo "$(YELLOW)>> Pipeline Docker Compose — mode full (GridSearchCV)$(RESET)"
	@echo "   API disponible sur : http://127.0.0.1:8000/docs"
	TRAIN_MODE=full docker compose up --build

compose-optuna: ## Pipeline Optuna : LightGBM 20 trials (~5 min) → API
	@echo "$(YELLOW)>> Pipeline Docker Compose — mode optuna$(RESET)"
	@echo "   API disponible sur : http://127.0.0.1:8000/docs"
	TRAIN_MODE=optuna docker compose up --build

compose-down: ## Arrête les conteneurs (conserve les volumes model_vol / mlruns_vol)
	@echo "$(YELLOW)>> Arrêt des conteneurs...$(RESET)"
	docker compose down

compose-clean: ## Arrête et supprime conteneurs + volumes (repart from scratch)
	@echo "$(YELLOW)>> Nettoyage complet (conteneurs + volumes)...$(RESET)"
	docker compose down -v
	@echo "$(GREEN)[OK] Environnement Docker réinitialisé$(RESET)"

compose-monitoring: ## Pipeline + MLflow UI sur http://127.0.0.1:5000
	@echo "$(YELLOW)>> Pipeline + MLflow UI...$(RESET)"
	@echo "   MLflow UI : http://127.0.0.1:5000"
	@echo "   API       : http://127.0.0.1:8000/docs"
	docker compose --profile monitoring up --build

compose-logs: ## Affiche les logs en temps réel
	docker compose logs -f

# ==============================================================================
# MLflow
# ==============================================================================

mlflow_ui: ## Lance l'interface MLflow sur http://127.0.0.1:5000
	@echo "$(YELLOW)>> Démarrage MLflow UI...$(RESET)"
	@echo "   Ouvrir : http://127.0.0.1:5000"
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
