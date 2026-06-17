#!/bin/sh
# Point d'entrée du conteneur trainer.
# Sélectionne le script d'entraînement selon TRAIN_MODE (défaut : baseline).
#
# TRAIN_MODE=baseline  → Logistic Regression (max_iter=300)  ~30s
# TRAIN_MODE=full      → GridSearchCV 5 modèles (CV=2)       ~10 min
# TRAIN_MODE=optuna    → Optuna TPE LightGBM (20 trials)     ~5 min
set -e

# Génère le dataset synthétique si le vrai CSV est absent (no-op sinon)
python tests/generate_synthetic_data.py

case "$TRAIN_MODE" in
  full)
    echo "[trainer] Mode : GridSearchCV — 5 modèles, CV=2"
    python -m mlproject.train_models --cv 2
    ;;
  optuna)
    echo "[trainer] Mode : Optuna TPE — LightGBM, 20 trials"
    python -m mlproject.train_optuna --model LightGBM --n-trials 20
    ;;
  *)
    echo "[trainer] Mode : Baseline — LogisticRegression (max_iter=300)"
    python -m mlproject.train --max-iter 300
    ;;
esac
