"""DAG Airflow — Ré-entraînement planifié du modèle IBM HR Attrition.

Pipeline : préparation des données → entraînement → contrôle qualité

Planification : toutes les 5 heures (cron : "0 */5 * * *")
XCom          : task_train pousse f1 + roc_auc → task_check_quality les lit
Garde-fou     : check_quality échoue si F1 < QUALITY_THRESHOLD (DAG run = failed)

Usage :
    # Depuis l'UI Airflow : activer "model_retraining" → Trigger DAG
    # Tester le garde-fou : monter QUALITY_THRESHOLD à 0.99 → check_quality doit échouer
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Seuil de qualité minimum acceptable ─────────────────────────────────────
# F1 minimal sur la classe "Yes" (attrition). Basé sur nos résultats :
#   LogisticRegression baseline ≈ 0.49 | RandomForest ≈ 0.48
# On garde une marge de sécurité à 0.35 pour ne pas bloquer le pipeline
# si les données synthétiques donnent un score légèrement inférieur.
QUALITY_THRESHOLD: float = float(os.getenv("QUALITY_THRESHOLD", "0.35"))

# ── Paramètres par défaut des tâches ────────────────────────────────────────
default_args = {
    "owner": "ethan.vieville",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
    "email_on_retry": False,
}


# ── Tâches ───────────────────────────────────────────────────────────────────

def task_prepare_data(**context) -> dict:
    """S17-1 — Génère ou rafraîchit le dataset IBM HR dans data/.

    Si le CSV réel est déjà présent, ne fait rien (idempotent).
    Sinon, génère 400 lignes synthétiques avec le même schéma.
    """
    import pandas as pd
    from mlproject.config import DATA_PATH
    from tests.generate_synthetic_data import generate

    if DATA_PATH.exists():
        n = len(pd.read_csv(DATA_PATH))
        print(f"[OK] Dataset déjà présent : {DATA_PATH} ({n} lignes) — rien à faire.")
        return {"rows": n, "source": "existing"}

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"[OK] Dataset synthétique généré : {DATA_PATH} ({len(df)} lignes, {df.columns.size} colonnes)")
    return {"rows": len(df), "source": "synthetic"}


def task_train(**context) -> dict:
    """S17-2 — Compare 5 modèles (GridSearchCV CV=2) et garde le champion.

    Utilise train_models() pour entraîner LogisticRegression, RandomForest,
    XGBoost, LightGBM et MLP en parallèle, puis sélectionne le meilleur ROC-AUC.
    Sauvegarde model.joblib + model.meta.json dans MODEL_DIR.
    Logue chaque modèle dans MLflow et tague le run comme airflow.
    """
    import json
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlproject.config import MODEL_DIR
    from mlproject.train_models import train_models

    print("=== Début de l'entraînement Airflow (5 modèles, CV=2) ===")
    train_models(cv=2)  # entraîne les 5 modèles et sauvegarde le champion

    # Lire les métriques du champion depuis le meta.json
    meta_path = MODEL_DIR / "model.meta.json"
    meta      = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics   = meta.get("metrics", {})
    print(f"[Champion] {meta.get('display_name', '?')}")
    print(f"[Métriques] {metrics}")

    # Tagger les runs MLflow comme déclenchés par Airflow
    try:
        tracking_uri    = os.getenv("MLFLOW_TRACKING_URI")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT", "ibm-hr-attrition")
        mlflow.set_tracking_uri(tracking_uri)
        client     = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment:
            # Tagger les runs créés dans les 10 dernières minutes
            import time as _time
            recent_ms = int((_time.time() - 600) * 1000)
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"attributes.start_time > {recent_ms}",
                order_by=["start_time DESC"],
                max_results=10,
            )
            airflow_run_id = context["dag_run"].run_id
            for r in runs:
                client.set_tag(r.info.run_id, "triggered_by",   "airflow")
                client.set_tag(r.info.run_id, "airflow_run_id", airflow_run_id)
            print(f"[MLflow] {len(runs)} run(s) tagués triggered_by=airflow")
    except Exception as exc:
        print(f"[WARN] Impossible de tagger les runs MLflow : {exc}")

    # Pousser les métriques du champion pour check_quality
    ti = context["ti"]
    ti.xcom_push(key="f1",        value=float(metrics.get("f1",        0.0)))
    ti.xcom_push(key="roc_auc",   value=float(metrics.get("roc_auc",   0.0)))
    ti.xcom_push(key="recall",    value=float(metrics.get("recall",    0.0)))
    ti.xcom_push(key="precision", value=float(metrics.get("precision", 0.0)))

    print(f"\n[XCom] f1={metrics.get('f1',0):.4f}  roc_auc={metrics.get('roc_auc',0):.4f}")
    return metrics


def task_check_quality(**context) -> None:
    """S17-3 — Vérifie que le modèle atteint le seuil de qualité minimum.

    Lit f1 depuis XCom (poussé par task_train).
    Lève ValueError si F1 < QUALITY_THRESHOLD → le DAG run passe en "failed"
    et le modèle n'est PAS considéré comme livrable.
    """
    ti  = context["ti"]
    f1       = ti.xcom_pull(task_ids="train", key="f1")
    roc_auc  = ti.xcom_pull(task_ids="train", key="roc_auc")
    recall   = ti.xcom_pull(task_ids="train", key="recall")
    precision = ti.xcom_pull(task_ids="train", key="precision")

    print("=== Rapport de qualité ===")
    print(f"  F1 (Yes)      : {f1:.4f}   (seuil : {QUALITY_THRESHOLD})")
    print(f"  ROC-AUC       : {roc_auc:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  Precision     : {precision:.4f}")

    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualité insuffisante : F1={f1:.3f} < seuil {QUALITY_THRESHOLD} — "
            "modèle non livré. Vérifiez les données ou ajustez les hyperparamètres."
        )

    print(f"\n[OK] Qualité validée : F1={f1:.3f} >= {QUALITY_THRESHOLD}")
    print("[OK] Modèle prêt pour la production.")


# ── Définition du DAG ────────────────────────────────────────────────────────

with DAG(
    dag_id="model_retraining",
    description=(
        "Ré-entraînement planifié IBM HR Attrition — "
        "préparation données → 5 modèles GridSearchCV → contrôle qualité"
    ),
    schedule="0 */5 * * *",        # toutes les 5 heures (0h, 5h, 10h, 15h, 20h)
    start_date=datetime(2024, 1, 1),
    catchup=False,                 # ne rejoue pas les runs manqués
    default_args=default_args,
    max_active_runs=1,             # évite les runs parallèles sur le même modèle
    is_paused_upon_creation=False,  # actif dès le premier démarrage (pas de pause manuelle)
    tags=["mlops", "ibm-hr", "retraining"],
    doc_md="""
## DAG model_retraining

Pipeline de ré-entraînement automatique du modèle IBM HR Attrition.

### Tâches
| Tâche | Rôle |
|-------|------|
| `prepare_data` | Génère des données synthétiques si le CSV est absent |
| `train` | Entraîne la baseline LogisticRegression, logue dans MLflow |
| `check_quality` | Vérifie F1 ≥ QUALITY_THRESHOLD (défaut 0.35) via XCom |

### Planification
Toutes les 5 heures : 0h, 5h, 10h, 15h, 20h.
Le DAG `daily_predictions` se déclenche 30 min après (0h30, 5h30...) pour laisser le temps au retrain de finir.

### Tester le garde-fou
Mettre `QUALITY_THRESHOLD=0.99` dans les variables Airflow (Admin → Variables)
puis re-déclencher le DAG → `check_quality` doit échouer.
    """,
) as dag:

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
        doc_md="Génère le dataset IBM HR synthétique si absent.",
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=task_train,
        doc_md="Entraîne LogisticRegression et pousse f1/roc_auc en XCom.",
    )

    check = PythonOperator(
        task_id="check_quality",
        python_callable=task_check_quality,
        doc_md="Valide F1 ≥ seuil, échoue si insuffisant.",
    )

    # S17-5 — Ordre d'exécution
    prepare >> train_task >> check
