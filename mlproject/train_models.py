"""Comparaison multi-modèles avec GridSearchCV et tracking MLflow.

Structure MLflow :
  Expérience : ibm-hr-attrition
  └── Run parent  : train_models_YYYYMMDD_HHMMSS
      ├── Run enfant : LogisticRegression
      ├── Run enfant : RandomForest
      ├── Run enfant : XGBoost
      └── Run enfant : LightGBM

Chaque run enfant log :
  - Hyperparamètres (meilleurs trouvés par GridSearchCV)
  - Métriques CV   : cv_roc_auc (critère de sélection)
  - Métriques test : test_f1, test_roc_auc, test_precision, test_recall
  - Artefacts      : confusion_matrix.png, classification_report.txt
  - Modèle sklearn : artefact "model/" chargeable via mlflow.sklearn.load_model()

Le meilleur modèle (ROC-AUC test) est enregistré dans le Model Registry
avec l'alias "@champion" et sauvegardé dans models/model.joblib.

Lancement :
  uv run python -m mlproject.train_models [--cv 5]
  make train_models [CV=5]

Visualisation :
  make mlflow_ui  -> http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field

# LightGBM reçoit parfois des numpy arrays sans noms de colonnes pendant la CV
# (comportement interne de GridSearchCV) — warning cosmétique, pas d'impact sur les résultats
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
from datetime import datetime
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mlproject.config import DATA, MLFLOW, MODEL_DIR, RANDOM_STATE, DeployStage
from mlproject.data import load_data, split
from mlproject.evaluation import (
    compute_metrics,
    log_classification_report,
    log_confusion_matrix,
)
from mlproject.features import build_preprocessor


# ── Structures de données ───────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """Définition d'un modèle candidat et de sa grille d'hyperparamètres.

    Les clés de param_grid doivent utiliser le préfixe "classifier__"
    car elles traversent le Pipeline scikit-learn.
    """
    name: str
    estimator: Any
    param_grid: dict[str, list]


@dataclass
class FamilyResult:
    """Résultats après GridSearchCV pour une famille de modèles."""
    name: str
    best_params: dict          # sans le préfixe "classifier__"
    cv_roc_auc: float          # ROC-AUC moyen sur les folds CV (critère de sélection)
    test_metrics: dict         # métriques finales sur le test set
    pipeline: Pipeline         # meilleur pipeline (refitté sur tout X_train)
    run_id: str                # MLflow run ID — utilisé pour l'enregistrement dans le Registry


# ── Définition des modèles ──────────────────────────────────────────────────

def build_model_specs(scale_pos_weight: float) -> list[ModelSpec]:
    """Retourne les 4 familles de modèles avec leurs grilles d'hyperparamètres.

    Grilles volontairement compactes (2-4 valeurs par paramètre) pour
    garder des temps d'exécution raisonnables (~2 min avec n_jobs=-1).

    Args:
        scale_pos_weight: ratio négatifs/positifs — utilisé par XGBoost
                          pour compenser le déséquilibre des classes.
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
                # C = inverse de la régularisation L2
                # Faible -> plus régularisé ; élevé -> moins contraint
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
                "classifier__n_estimators":   [100, 200],
                "classifier__max_depth":      [5, 10, None],  # None = arbres complets
                "classifier__min_samples_leaf": [1, 5],       # régularisation
            },
        ),
        ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                # XGBoost n'a pas class_weight — on utilise scale_pos_weight
                # = nb négatifs / nb positifs (~5.2 sur ce dataset)
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
                "classifier__num_leaves":    [31, 63],   # contrôle la complexité
                "classifier__learning_rate": [0.05, 0.1],
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
    """Lance un run MLflow enfant pour une famille de modèles.

    1. GridSearchCV stratifié sur X_train (scoring = roc_auc)
    2. Évaluation du meilleur pipeline sur X_test
    3. Log MLflow : params, métriques, artefacts, modèle sklearn
    """
    with mlflow.start_run(run_name=spec.name, nested=True) as run:
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   spec.estimator),
        ])

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

        # GridSearchCV optimise sur ROC-AUC — plus robuste que F1 pour
        # les datasets déséquilibrés car indépendant du seuil de décision
        grid = GridSearchCV(
            pipeline,
            param_grid=spec.param_grid,
            cv=cv_strategy,
            scoring="roc_auc",
            n_jobs=-1,   # parallélise les folds
            refit=True,  # refit le meilleur modèle sur tout X_train
            verbose=0,
        )
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        # Supprime le préfixe "classifier__" pour la lisibilité des logs
        best_params   = {
            k.replace("classifier__", ""): v
            for k, v in grid.best_params_.items()
        }
        cv_roc_auc = float(grid.best_score_)

        y_pred  = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        # ── Log MLflow ───────────────────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_roc_auc", cv_roc_auc)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        log_confusion_matrix(y_test, y_pred)
        log_classification_report(y_test, y_pred)

        # Artefact "model/" : chargeable via mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        mlflow.sklearn.log_model(best_pipeline, name="model")

        print(
            f"  [{spec.name:<20}]  "
            f"CV ROC-AUC={cv_roc_auc:.4f}  "
            f"Test F1={metrics['f1']:.4f}  "
            f"Test ROC-AUC={metrics['roc_auc']:.4f}"
        )
        print(f"    Best params : {best_params}")

        return FamilyResult(
            name        = spec.name,
            best_params = best_params,
            cv_roc_auc  = cv_roc_auc,
            test_metrics = metrics,
            pipeline    = best_pipeline,
            run_id      = run.info.run_id,
        )


# ── Point d'entrée principal ────────────────────────────────────────────────

def train_models(cv: int = 5) -> None:
    """Compare tous les modèles, log dans MLflow, sauvegarde le champion.

    Args:
        cv: nombre de folds pour la validation croisée stratifiée.
    """
    # ── Setup MLflow ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    mlflow.set_experiment(MLFLOW.experiment)

    print("=== Chargement des données ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"  Train : {X_train.shape} | Test : {X_test.shape}")

    # Ratio négatifs/positifs pour XGBoost
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"  scale_pos_weight (XGBoost) : {scale_pos_weight:.2f}")

    preprocessor = build_preprocessor()
    specs        = build_model_specs(scale_pos_weight)
    run_name     = f"train_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n=== Entraînement ({len(specs)} modèles, CV={cv}) ===")
    print(f"  Expérience MLflow : {MLFLOW.experiment}")
    print(f"  Run parent        : {run_name}\n")

    results: list[FamilyResult] = []

    # ── Run parent — contient les méta-données de la session ─────────────────
    with mlflow.start_run(run_name=run_name) as parent_run:
        mlflow.log_params({
            "cv_folds":      cv,
            "n_models":      len(specs),
            "dataset_size":  len(df),
            "train_size":    len(X_train),
            "test_size":     len(X_test),
            "attrition_rate": f"{y_train.mean():.2%}",
        })

        for spec in specs:
            print(f"  >> {spec.name} ...")
            try:
                result = run_family(
                    spec, preprocessor,
                    X_train, y_train,
                    X_test,  y_test,
                    cv=cv,
                )
                results.append(result)
            except Exception as exc:
                print(f"  [ERREUR] {spec.name} : {exc}")

    if not results:
        print("[ERREUR] Aucun modèle entraîné avec succès.")
        return

    # ── Sélection du champion ────────────────────────────────────────────────
    # Critère : ROC-AUC sur le test set (seuil-indépendant, fiable sur données déséquilibrées)
    best = max(results, key=lambda r: r.test_metrics["roc_auc"])

    print(f"\n=== Comparatif final ===")
    print(f"  {'Modèle':<22} {'ROC-AUC':>10} {'F1':>8} {'Recall':>8} {'Precision':>10}")
    print(f"  {'-'*60}")
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

    # ── Sauvegarde locale ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(best.pipeline, model_path)
    print(f"\n[OK] Champion sauvegardé      -> {model_path}")

    # ── Enregistrement dans le Model Registry ────────────────────────────────
    # L'alias "@champion" permet de charger le modèle via
    # mlflow.sklearn.load_model(f"models:/{MLFLOW.model_name}@champion")
    try:
        model_uri  = f"runs:/{best.run_id}/model"
        registered = mlflow.register_model(model_uri, MLFLOW.model_name)

        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            MLFLOW.model_name, DeployStage.CHAMPION.value, registered.version
        )
        print(f"[OK] Modèle enregistré        -> {MLFLOW.model_name} v{registered.version} (@champion)")
    except Exception as exc:
        # Le Registry nécessite un backend avec DB (sqlite:///mlruns.db)
        print(f"[WARN] Model Registry ignoré  : {exc}")

    print(f"\n[OK] MLflow UI -> make mlflow_ui  ({MLFLOW.tracking_uri})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison multi-modèles — IBM HR Attrition")
    parser.add_argument("--cv", type=int, default=5, help="Folds CV (défaut: 5)")
    args = parser.parse_args()

    # Import ici pour éviter une référence circulaire dans la docstring
    train_models(cv=args.cv)
