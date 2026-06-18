"""Entraînement d'un modèle unique avec des hyperparamètres personnalisés.

Utilisé par le dashboard Streamlit pour lancer un entraînement ad-hoc
sur n'importe lequel des 5 modèles avec les params choisis par l'utilisateur.

Exemples :
    python -m mlproject.train_custom --model LightGBM --n-estimators 200 --num-leaves 63 --learning-rate 0.05
    python -m mlproject.train_custom --model LogisticRegression --c 0.1 --max-iter 500
    python -m mlproject.train_custom --model RandomForest --n-estimators 300 --max-depth 10
    python -m mlproject.train_custom --model XGBoost --n-estimators 150 --max-depth 6 --learning-rate 0.1
    python -m mlproject.train_custom --model MLP --hidden-sizes 128,64,32 --lr 0.001 --dropout 0.3
"""
from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import mlproject.tracking as tracking
from mlproject.config import MODEL_DIR, RANDOM_STATE
from mlproject.data import load_data, split
from mlproject.evaluation import compute_metrics
from mlproject.features import build_preprocessor
from mlproject.model_registry import save_model_snapshot
from mlproject.models import TorchMLPClassifier


def _build_estimator(args) -> tuple:
    """Instancie l'estimateur et retourne (estimator, label, params)."""
    m = args.model

    if m == "LogisticRegression":
        params = {"C": args.c, "max_iter": args.max_iter}
        est    = LogisticRegression(
            class_weight="balanced", random_state=RANDOM_STATE, solver="lbfgs", **params,
        )
        label  = f"LogisticRegression (C={args.c}, max_iter={args.max_iter})"

    elif m == "RandomForest":
        max_depth = None if args.max_depth == 0 else args.max_depth
        params    = {
            "n_estimators":     args.n_estimators,
            "max_depth":        max_depth,
            "min_samples_leaf": args.min_samples_leaf,
        }
        est   = RandomForestClassifier(
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, **params,
        )
        depth_str = "None" if max_depth is None else str(max_depth)
        label = f"RandomForest (n_est={args.n_estimators}, depth={depth_str})"

    elif m == "XGBoost":
        params = {
            "n_estimators":  args.n_estimators,
            "max_depth":     args.max_depth,
            "learning_rate": args.learning_rate,
        }
        df_tmp              = load_data()
        _, _, y_tmp, _      = split(df_tmp)
        scale_pos_weight    = float((y_tmp == 0).sum() / (y_tmp == 1).sum())
        est = XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0, **params,
        )
        label = f"XGBoost (n_est={args.n_estimators}, depth={args.max_depth}, lr={args.learning_rate})"

    elif m == "LightGBM":
        params = {
            "n_estimators":  args.n_estimators,
            "num_leaves":    args.num_leaves,
            "learning_rate": args.learning_rate,
        }
        est   = LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1, **params)
        label = f"LightGBM (n_est={args.n_estimators}, leaves={args.num_leaves}, lr={args.learning_rate})"

    elif m == "MLP":
        hidden = tuple(int(x) for x in args.hidden_sizes.split(","))
        params = {
            "hidden_sizes": hidden,
            "lr":           args.lr,
            "dropout":      args.dropout,
            "epochs":       args.epochs,
        }
        est   = TorchMLPClassifier(random_state=RANDOM_STATE, **params)
        label = f"MLP (hidden={args.hidden_sizes}, lr={args.lr}, dropout={args.dropout}, epochs={args.epochs})"

    else:
        raise ValueError(f"Modèle inconnu : {m}")

    return est, label, params


def train_custom(args) -> dict:
    tracking.setup()

    print("=== Chargement des données ===")
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    print(f"  Train : {X_train.shape[0]} exemples | Test : {X_test.shape[0]} exemples")

    estimator, label, params = _build_estimator(args)
    preprocessor = build_preprocessor()

    with tracking.start_run(f"custom_{args.model}") as run:
        tracking.log_tags({"optimizer": "custom", "model": args.model})
        tracking.log_params({k: str(v) for k, v in params.items()})

        print(f"\n=== Entraînement : {label} ===")
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        pipeline.fit(X_train, y_train)
        print("  Entraînement terminé.")

        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        tracking.log_metrics(metrics, prefix="test_")
        tracking.log_confusion_matrix(y_test, y_pred)
        tracking.log_classification_report(y_test, y_pred)
        tracking.log_model(pipeline)

        print("\n=== Résultats test set ===")
        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"  F1 (Yes)  : {metrics['f1']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, MODEL_DIR / "model.joblib")
        save_model_snapshot(
            pipeline,
            label=label,
            params={k: str(v) for k, v in params.items()},
            metrics=metrics,
            run_id=run.info.run_id,
            save_as_champion=True,
        )

        print(f"\n[OK] Modèle sauvegardé  -> {MODEL_DIR / 'model.joblib'}")
        print(f"[OK] Run MLflow         -> {run.info.run_id[:8]}")
        tracking.register_champion(run.info.run_id)

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Entraînement personnalisé — IBM HR Attrition")
    p.add_argument("--model", required=True,
                   choices=["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "MLP"])

    # ── LogisticRegression ──────────────────────────────────────────────────
    p.add_argument("--c",               type=float, default=1.0,   help="Régularisation L2")
    p.add_argument("--max-iter",        type=int,   default=1000,  help="Itérations max")

    # ── RandomForest / arbres ───────────────────────────────────────────────
    p.add_argument("--n-estimators",    type=int,   default=100)
    p.add_argument("--max-depth",       type=int,   default=0,     help="0 = None (illimité)")
    p.add_argument("--min-samples-leaf",type=int,   default=1)

    # ── XGBoost / LightGBM ─────────────────────────────────────────────────
    p.add_argument("--learning-rate",   type=float, default=0.1)
    p.add_argument("--num-leaves",      type=int,   default=31,    help="LightGBM uniquement")

    # ── MLP PyTorch ─────────────────────────────────────────────────────────
    p.add_argument("--hidden-sizes",    type=str,   default="64,32",
                   help="Couches cachées séparées par virgule, ex: 128,64,32")
    p.add_argument("--lr",              type=float, default=1e-3,  help="Learning rate Adam")
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--epochs",          type=int,   default=150)

    train_custom(p.parse_args())
