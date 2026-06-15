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

YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help check-uv check-venv install sync lock reset-env doctor \
        train train_models check_pipeline mlflow_ui

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

train_models: check-venv ## Compare 4 modèles via GridSearchCV + MLflow (CV=..)
	@echo "$(YELLOW)>> Comparaison multi-modèles...$(RESET)"
	$(PYTHON) -m mlproject.train_models --cv $(CV)

# ==============================================================================
# MLflow
# ==============================================================================

mlflow_ui: ## Lance l'interface MLflow sur http://127.0.0.1:5000
	@echo "$(YELLOW)>> Démarrage MLflow UI...$(RESET)"
	@echo "   Ouvrir : http://127.0.0.1:5000"
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlruns.db --host 127.0.0.1 --port 5000
