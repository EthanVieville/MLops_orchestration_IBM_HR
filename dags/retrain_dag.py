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
    """S17-2 — Entraîne la baseline LogisticRegression et pousse les métriques en XCom.

    Sauvegarde model.joblib + model.meta.json dans MODEL_DIR.
    Logue le run dans MLflow (URI configurée via MLFLOW_TRACKING_URI).
    """
    from mlproject.train import train

    print("=== Début de l'entraînement Airflow ===")
    metrics = train()

    # Pousser toutes les métriques pour que check_quality puisse les lire
    ti = context["ti"]
    ti.xcom_push(key="f1",        value=metrics["f1"])
    ti.xcom_push(key="roc_auc",   value=metrics["roc_auc"])
    ti.xcom_push(key="recall",    value=metrics["recall"])
    ti.xcom_push(key="precision", value=metrics["precision"])

    print(f"\n[XCom] f1={metrics['f1']:.4f}  roc_auc={metrics['roc_auc']:.4f}")
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
        "préparation données → entraînement → contrôle qualité"
    ),
    schedule="0 */5 * * *",        # toutes les 5 heures (0h, 5h, 10h, 15h, 20h)
    start_date=datetime(2024, 1, 1),
    catchup=False,                 # ne rejoue pas les runs manqués
    default_args=default_args,
    max_active_runs=1,             # évite les runs parallèles sur le même modèle
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
