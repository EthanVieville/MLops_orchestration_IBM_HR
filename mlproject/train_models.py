"""Comparaison multi-modèles avec GridSearchCV.

Tracking MLflow délégué à mlproject.tracking (aucun import mlflow direct).

Modèles comparés :
  1. LogisticRegression (sklearn)  — baseline rapide, très interprétable
  2. RandomForest       (sklearn)  — ensemble bagging, robuste
  3. XGBoost            (xgboost) — gradient boosting, fort sur tabular
  4. LightGBM           (lightgbm)— gradient boosting, plus rapide que XGB
  5. MLP                (PyTorch)  — réseau de neurones, profil différent

Structure MLflow :
  Expérience : ibm-hr-attrition
  └── Run parent  : train_models_YYYYMMDD_HHMMSS
      ├── Run enfant : LogisticRegression
      ├── Run enfant : RandomForest
      ├── Run enfant : XGBoost
      ├── Run enfant : LightGBM
      └── Run enfant : MLP

Critère de sélection du champion : ROC-AUC sur le test set.

Lancement :
  make train_models [CV=5]
  uv run python -m mlproject.train_models [--cv 5]

Visualisation :
  make mlflow_ui  ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

# LightGBM reçoit parfois des arrays numpy sans noms de colonnes pendant la CV
# (comportement interne de GridSearchCV) — warning cosmétique, sans impact
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import joblib
from sklearn.metrics import make_scorer, roc_auc_score as _roc_auc_score
from lightgbm import LGBMClassifier

# Scorer robuste pour tous les estimateurs (sklearn natifs et wrappers PyTorch).
# sklearn 1.9 ne sait pas extraire automatiquement [:, 1] du predict_proba 2D
# pour les classifiers custom — on le fait explicitement ici.
_ROC_AUC = make_scorer(
    lambda y_true, y_proba: _roc_auc_score(
        y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    ),
    response_method="predict_proba",
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import mlproject.tracking as tracking
from mlproject.config import MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.evaluation import compute_metrics
from mlproject.features import build_preprocessor
from mlproject.models import TorchMLPClassifier


# ── Structures de données ───────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Définition d'un modèle candidat et de sa grille d'hyperparamètres.

    Les clés de param_grid utilisent le préfixe "classifier__"
    pour traverser le Pipeline scikit-learn.
    """
    name: str
    estimator: Any
    param_grid: dict[str, list]


@dataclass
class FamilyResult:
    """Résultats après GridSearchCV pour une famille de modèles."""
    name: str
    best_params: dict      # sans le préfixe "classifier__"
    cv_roc_auc: float      # ROC-AUC moyen sur les folds CV
    test_metrics: dict     # métriques sur le test set
    pipeline: Pipeline     # meilleur pipeline refitté sur tout X_train
    run_id: str            # MLflow run ID pour le Registry


# ── Définition des modèles ──────────────────────────────────────────────────

def build_model_specs(scale_pos_weight: float) -> list[ModelSpec]:
    """Retourne les 5 familles de modèles avec leurs grilles d'hyperparamètres.

    Grilles compactes (2-4 valeurs par param) pour des temps d'exécution
    raisonnables (~3-5 min avec n_jobs=-1).

    Args:
        scale_pos_weight: ratio négatifs/positifs pour XGBoost (~5.2).
    """
    return [
        ModelSpec(
            name="LogisticRegression",
            estimator=LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=1000,
                solver="lbfgs",
            ),
            param_grid={
                # Inverse de la régularisation L2 : faible = plus contraint
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            },
        ),
        ModelSpec(
            name="RandomForest",
            estimator=RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_grid={
                "classifier__n_estimators":     [100, 200],
                "classifier__max_depth":        [5, 10, None],  # None = arbres complets
                "classifier__min_samples_leaf": [1, 5],
            },
        ),
        ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                # XGBoost utilise scale_pos_weight à la place de class_weight
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                verbosity=0,
            ),
            param_grid={
                "classifier__n_estimators":  [100, 200],
                "classifier__max_depth":     [3, 6],
                "classifier__learning_rate": [0.05, 0.1],
            },
        ),
        ModelSpec(
            name="LightGBM",
            estimator=LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbosity=-1,
            ),
            param_grid={
                "classifier__n_estimators":  [100, 200],
                "classifier__num_leaves":    [31, 63],    # complexité des arbres
                "classifier__learning_rate": [0.05, 0.1],
            },
        ),
        ModelSpec(
            name="MLP",
            # MLP PyTorch via wrapper sklearn — gère le déséquilibre via pos_weight interne
            estimator=TorchMLPClassifier(random_state=RANDOM_STATE),
            param_grid={
                # Architecture : nombre et taille des couches cachées
                "classifier__hidden_sizes": [(64, 32), (128, 64, 32)],
                # Learning rate Adam
                "classifier__lr":           [1e-3, 5e-4],
            },
        ),
    ]


# ── Entraînement d'une famille ──────────────────────────────────────────────

def run_family(
    spec: ModelSpec,
    preprocessor,
    X_train, y_train,
    X_test,  y_test,
    cv: int,
) -> FamilyResult:
    """GridSearchCV + logging MLflow pour une famille de modèles.

    1. GridSearchCV stratifié sur X_train (scoring = roc_auc)
    2. Évaluation du meilleur pipeline sur X_test
    3. Log via tracking : params, métriques, artefacts, modèle
    """
    with tracking.start_run(spec.name, nested=True) as run:
        tracking.log_tags({"optimizer": "gridsearch", "model": spec.name})

        pipeline    = Pipeline([("preprocessor", preprocessor), ("classifier", spec.estimator)])
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

        # Critère : roc_auc — robuste aux déséquilibres, indépendant du seuil
        grid = GridSearchCV(
            pipeline,
            param_grid=spec.param_grid,
            cv=cv_strategy,
            scoring=_ROC_AUC,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        best_params   = {k.replace("classifier__", ""): v for k, v in grid.best_params_.items()}
        cv_roc_auc    = float(grid.best_score_)

        y_pred  = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        tracking.log_params(best_params)
        tracking.log_metrics({"cv_roc_auc": cv_roc_auc})
        tracking.log_metrics(metrics, prefix="test_")
        tracking.log_confusion_matrix(y_test, y_pred)
        tracking.log_classification_report(y_test, y_pred)
        tracking.log_model(best_pipeline)

        print(
            f"  [{spec.name:<20}]  "
            f"CV={cv_roc_auc:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )
        print(f"    Best params : {best_params}")

        return FamilyResult(
            name         = spec.name,
            best_params  = best_params,
            cv_roc_auc   = cv_roc_auc,
            test_metrics = metrics,
            pipeline     = best_pipeline,
            run_id       = run.info.run_id,
        )


# ── Point d'entrée ──────────────────────────────────────────────────────────

def train_models(cv: int = 5) -> None:
    """Compare tous les modèles, log dans MLflow, sauvegarde le champion."""
    tracking.setup()

    print("=== Chargement des données ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"  scale_pos_weight (XGBoost) : {scale_pos_weight:.2f}")

    preprocessor = build_preprocessor()
    specs        = build_model_specs(scale_pos_weight)
    run_name     = f"train_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n=== Entraînement ({len(specs)} modèles, CV={cv}) ===")
    print(f"  Expérience : {tracking.MLFLOW.experiment}")
    print(f"  Run parent : {run_name}\n")

    results: list[FamilyResult] = []

    with tracking.start_run(run_name) as _:
        tracking.log_params({
            "cv_folds":      cv,
            "n_models":      len(specs),
            "dataset_size":  len(df),
            "attrition_rate": f"{y_train.mean():.2%}",
        })
        tracking.log_tags({"optimizer": "gridsearch"})

        for spec in specs:
            print(f"  >> {spec.name} ...")
            try:
                results.append(run_family(
                    spec, preprocessor,
                    X_train, y_train,
                    X_test,  y_test,
                    cv=cv,
                ))
            except Exception as exc:
                print(f"  [ERREUR] {spec.name} : {exc}")

    if not results:
        print("[ERREUR] Aucun modèle entraîné avec succès.")
        return

    # ── Sélection du champion ────────────────────────────────────────────────
    best = max(results, key=lambda r: r.test_metrics["roc_auc"])

    print("\n=== Comparatif final (trié par ROC-AUC test) ===")
    print(f"  {'Modèle':<22} {'ROC-AUC':>10} {'F1':>8} {'Recall':>8} {'Precision':>10}")
    print(f"  {'-'*62}")
    for r in sorted(results, key=lambda r: r.test_metrics["roc_auc"], reverse=True):
        m      = r.test_metrics
        marker = "  << CHAMPION" if r.name == best.name else ""
        print(
            f"  {r.name:<22}"
            f"  {m['roc_auc']:>8.4f}"
            f"  {m['f1']:>6.4f}"
            f"  {m['recall']:>6.4f}"
            f"  {m['precision']:>8.4f}"
            f"{marker}"
        )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best.pipeline, MODEL_DIR / "model.joblib")
    print(f"\n[OK] Champion sauvegardé -> {MODEL_DIR / 'model.joblib'}")

    tracking.register_champion(best.run_id)
    print("[OK] MLflow UI          -> make mlflow_ui")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison multi-modèles — IBM HR Attrition")
    parser.add_argument("--cv", type=int, default=5, help="Folds CV (défaut: 5)")
    args = parser.parse_args()
    train_models(cv=args.cv)
