"""Optimisation d'hyperparamètres via Optuna (algorithme TPE).

Alternative à GridSearchCV : au lieu d'explorer une grille fixe,
Optuna utilise l'algorithme TPE (Tree-structured Parzen Estimator) pour
cibler les zones prometteuses de l'espace de recherche.

Avantages vs GridSearchCV :
  - Espace continu (learning_rate entre 0.001 et 0.3, pas 2 valeurs fixes)
  - Converge plus vite : 50 trials TPE ~= 200 combinaisons grid search
  - Chaque trial exploite les résultats précédents (apprentissage bayésien)

Structure MLflow :
  Expérience : ibm-hr-attrition
  └── Run parent  : optuna_LightGBM_YYYYMMDD_HHMMSS
      ├── Run enfant : trial_000  (params + cv_roc_auc)
      ├── Run enfant : trial_001
      └── ...

Lancement :
  make train_optuna                              # LightGBM, 50 trials, CV=5
  make train_optuna MODEL=XGBoost N_TRIALS=100
  uv run python -m mlproject.train_optuna [--model LightGBM] [--n-trials 50] [--cv 5]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score as _roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlproject.tracking as tracking
from mlproject.config import MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.evaluation import compute_metrics
from mlproject.features import build_preprocessor
from mlproject.models import TorchMLPClassifier

# Même scorer que train_models.py — corrige le bug sklearn 1.9 sur les classifiers custom
_ROC_AUC = make_scorer(
    lambda y_true, y_proba: _roc_auc_score(
        y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    ),
    response_method="predict_proba",
)

AVAILABLE_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "MLP"]

# Optuna est verbeux par défaut — on n'affiche que les warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Espaces de recherche ─────────────────────────────────────────────────────

def _suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """Définit l'espace de recherche Optuna pour chaque modèle.

    Les espaces sont plus larges qu'une grille GridSearchCV classique :
    learning_rate en log-scale [0.001, 0.3] au lieu de [0.05, 0.1].
    """
    if model_name == "LogisticRegression":
        return {
            # log=True : exploration log-uniforme, adaptée aux ordres de grandeur
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
        }
    if model_name == "RandomForest":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 400),
            "max_depth":        trial.suggest_categorical("max_depth", [5, 10, 20, None]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 400),
            "max_depth":     trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        }
    if model_name == "MLP":
        return {
            "hidden_sizes": trial.suggest_categorical(
                "hidden_sizes", [(64, 32), (128, 64), (128, 64, 32)]
            ),
            "lr":      trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        }
    raise ValueError(f"Modèle inconnu : {model_name}")


# ── Construction de l'estimateur ─────────────────────────────────────────────

def _build_estimator(model_name: str, params: dict, scale_pos_weight: float):
    """Instancie l'estimateur sklearn à partir des hyperparamètres suggérés."""
    if model_name == "LogisticRegression":
        return LogisticRegression(
            class_weight="balanced", random_state=RANDOM_STATE,
            max_iter=2000, solver="lbfgs", **params,
        )
    if model_name == "RandomForest":
        return RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **params,
        )
    if model_name == "XGBoost":
        return XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0, **params,
        )
    if model_name == "LightGBM":
        return LGBMClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1, **params,
        )
    if model_name == "MLP":
        # hidden_sizes peut être une liste ou un tuple selon Optuna — on normalise
        params = {**params, "hidden_sizes": tuple(params["hidden_sizes"])}
        return TorchMLPClassifier(random_state=RANDOM_STATE, **params)
    raise ValueError(f"Modèle inconnu : {model_name}")


# ── Fonction objectif ─────────────────────────────────────────────────────────

def build_objective(
    model_name: str,
    preprocessor,
    X_train,
    y_train,
    cv: int,
    scale_pos_weight: float,
):
    """Retourne la fonction objectif pour Optuna.

    Chaque appel (= un trial Optuna) :
      1. Suggère des hyperparamètres via TPE
      2. Évalue par cross-validation stratifiée (métrique : ROC-AUC)
      3. Log params + score dans un run MLflow enfant

    Returns:
        Callable[optuna.Trial] -> float  (CV ROC-AUC à maximiser)
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params    = _suggest_params(trial, model_name)
        estimator = _build_estimator(model_name, params, scale_pos_weight)
        pipeline  = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])

        with tracking.start_run(f"trial_{trial.number:03d}", nested=True):
            # Sérialisation des tuples pour MLflow (ex: (64, 32) -> "64,32")
            loggable = {
                k: str(v) if isinstance(v, (tuple, list)) else v
                for k, v in params.items()
            }
            tracking.log_params({"trial": trial.number, **loggable})

            scores   = cross_val_score(
                pipeline, X_train, y_train,
                cv=cv_strategy, scoring=_ROC_AUC, n_jobs=-1,
            )
            cv_score = float(scores.mean())
            tracking.log_metrics({
                "cv_roc_auc":     cv_score,
                "cv_roc_auc_std": float(scores.std()),
            })

        return cv_score

    return objective


# ── Point d'entrée ───────────────────────────────────────────────────────────

def train_optuna(model_name: str = "LightGBM", n_trials: int = 50, cv: int = 5) -> None:
    """Lance une étude Optuna, log chaque trial dans MLflow, sauvegarde le champion."""
    if model_name not in AVAILABLE_MODELS:
        print(f"[ERREUR] Modèle inconnu : {model_name}")
        print(f"  Disponibles : {', '.join(AVAILABLE_MODELS)}")
        sys.exit(1)

    tracking.setup()

    print("=== Chargement des données ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    preprocessor     = build_preprocessor()
    run_name         = f"optuna_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n=== Optimisation Optuna ===")
    print(f"  Modèle   : {model_name}")
    print(f"  Trials   : {n_trials}  |  CV folds : {cv}")
    print(f"  Run parent MLflow : {run_name}\n")

    with tracking.start_run(run_name) as parent_run:
        tracking.log_params({
            "model":          model_name,
            "n_trials":       n_trials,
            "cv_folds":       cv,
            "optimizer":      "TPE",
            "dataset_size":   len(df),
            "attrition_rate": f"{y_train.mean():.2%}",
        })
        # Tags : permettent de filtrer/comparer les runs Optuna dans l'UI MLflow
        # (ex: filter tag "optimizer=optuna" pour voir tous les modèles optimisés)
        tracking.log_tags({
            "optimizer": "optuna",
            "model":     model_name,
        })

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        objective = build_objective(
            model_name, preprocessor, X_train, y_train, cv, scale_pos_weight,
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_trial  = study.best_trial
        best_params = best_trial.params
        best_cv     = best_trial.value

        print(f"\n=== Meilleur trial : #{best_trial.number} ===")
        print(f"  CV ROC-AUC : {best_cv:.4f}")
        print(f"  Params     : {best_params}")

        tracking.log_metrics({
            "best_cv_roc_auc":    best_cv,
            "n_completed_trials": len(study.trials),
        })
        loggable_best = {
            f"best_{k}": str(v) if isinstance(v, (tuple, list)) else v
            for k, v in best_params.items()
        }
        tracking.log_params(loggable_best)

        # ── Refit sur tout X_train avec les meilleurs hyperparamètres ────────
        print("\n=== Refit sur X_train complet ===")
        best_estimator = _build_estimator(model_name, best_params, scale_pos_weight)
        best_pipeline  = Pipeline([("preprocessor", preprocessor), ("classifier", best_estimator)])
        best_pipeline.fit(X_train, y_train)

        y_pred  = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        tracking.log_metrics(metrics, prefix="test_")
        tracking.log_confusion_matrix(y_test, y_pred)
        tracking.log_classification_report(y_test, y_pred)
        tracking.log_model(best_pipeline)

        print("\n=== Métriques test set ===")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"  F1 (Yes)  : {metrics['f1']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(best_pipeline, MODEL_DIR / "model.joblib")
        from mlproject.model_registry import save_model_snapshot
        key_items  = list(best_params.items())[:2]
        params_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in key_items
        )
        save_model_snapshot(
            best_pipeline,
            label=f"{model_name} optuna ({params_str})",
            params={k: str(v) for k, v in best_params.items()},
            metrics=metrics,
            run_id=parent_run.info.run_id,
            save_as_champion=True,
        )
        print(f"\n[OK] Modèle sauvegardé   -> {MODEL_DIR / 'model.joblib'}")

        tracking.register_champion(parent_run.info.run_id)
        print("[OK] MLflow UI           -> make mlflow_ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimisation Optuna — IBM HR Attrition")
    parser.add_argument(
        "--model",
        default="LightGBM",
        choices=AVAILABLE_MODELS,
        help="Famille de modèles à optimiser (défaut : LightGBM)",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Nombre de trials Optuna (défaut : 50)")
    parser.add_argument("--cv",       type=int, default=5,  help="Folds CV (défaut : 5)")
    args = parser.parse_args()
    train_optuna(model_name=args.model, n_trials=args.n_trials, cv=args.cv)
