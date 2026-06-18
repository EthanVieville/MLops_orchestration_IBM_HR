"""DAG Airflow — Prédictions quotidiennes via l'API IBM HR Attrition.

Chaque jour à 10h, échantillonne N employés du dataset et envoie leurs
caractéristiques à l'endpoint POST /predict de l'API. Simule un trafic de
production et alimente la boucle de feedback.

Pré-requis : l'API doit être joignable (make api ou make compose-up).

XCom : n_predictions + n_attrition poussés par send_predictions
Planification : tous les jours à 10h (cron : "0 10 * * *")
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL       = os.getenv("API_URL", "http://api:8000")
N_PREDICTIONS = int(os.getenv("N_PREDICTIONS", "20"))

default_args = {
    "owner": "ethan.vieville",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False,
    "email_on_retry": False,
}


# ── Tâches ───────────────────────────────────────────────────────────────────

def task_send_predictions(**context) -> dict:
    """S17-6/7 — Échantillonne N employés et envoie leurs prédictions à l'API.

    Retire la colonne cible avant l'envoi (on ne doit pas la soumettre).
    Convertit chaque ligne en JSON natif (pas de types numpy).
    Pousse n_predictions + n_attrition en XCom pour traçabilité.
    """
    import json

    import httpx
    import pandas as pd

    from mlproject.config import (
        CATEGORICAL_FEATURES,
        DATA_PATH,
        NUMERIC_FEATURES,
    )

    # ── Chargement des données ──────────────────────────────────────────────
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}\n"
            "Lancez d'abord le DAG model_retraining (tâche prepare_data)."
        )

    df = pd.read_csv(DATA_PATH)
    feature_cols = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    features = df[feature_cols]

    # S17-6 — Échantillonnage
    sample = features.sample(n=min(N_PREDICTIONS, len(features)), random_state=context["ts_nodash"])
    print(f"[INFO] {len(sample)} employés échantillonnés sur {len(features)} disponibles")

    # ── Envoi des prédictions ───────────────────────────────────────────────
    results: list[dict] = []
    errors:  list[str]  = []

    # S17-7 — Envoi via httpx
    with httpx.Client(base_url=API_URL, timeout=10.0) as client:
        # Vérification santé avant l'envoi
        health = client.get("/health")
        health.raise_for_status()
        print(f"[OK] API joignable sur {API_URL} — statut : {health.json()}")

        for idx, (_, row) in enumerate(sample.iterrows()):
            # json.loads(row.to_json()) convertit les types numpy → types Python natifs
            payload = json.loads(row.to_json())
            try:
                r = client.post("/predict", json=payload)
                r.raise_for_status()
                result = r.json()
                results.append(result)
                label = "QUITTE" if result["prediction"] == 1 else "reste "
                print(f"  [{idx+1:02d}] {label} | P(attrition)={result['probability']:.3f}")
            except httpx.HTTPStatusError as e:
                errors.append(f"Ligne {idx}: {e.response.status_code} — {e.response.text[:100]}")

    # ── Bilan ───────────────────────────────────────────────────────────────
    n_attrition = sum(r["prediction"] for r in results)
    n_success   = len(results)
    rate        = n_attrition / n_success if n_success > 0 else 0.0

    print(f"\n=== Bilan des prédictions ===")
    print(f"  Envoyées  : {n_success} / {len(sample)}")
    print(f"  Attrition : {n_attrition} ({rate:.1%})")
    if errors:
        print(f"  Erreurs   : {len(errors)}")
        for e in errors:
            print(f"    - {e}")

    if errors and not results:
        raise RuntimeError(f"Toutes les prédictions ont échoué : {errors[0]}")

    # Pousser les statistiques en XCom
    ti = context["ti"]
    ti.xcom_push(key="n_predictions", value=n_success)
    ti.xcom_push(key="n_attrition",   value=n_attrition)
    ti.xcom_push(key="attrition_rate", value=round(rate, 4))

    return {"n_predictions": n_success, "n_attrition": n_attrition, "rate": rate}


def task_log_summary(**context) -> None:
    """Lit les XCom de send_predictions et affiche un résumé structuré."""
    ti            = context["ti"]
    n_predictions = ti.xcom_pull(task_ids="send_predictions", key="n_predictions")
    n_attrition   = ti.xcom_pull(task_ids="send_predictions", key="n_attrition")
    attrition_rate = ti.xcom_pull(task_ids="send_predictions", key="attrition_rate")

    run_date = context["ds"]
    print(f"=== Résumé quotidien — {run_date} ===")
    print(f"  Prédictions envoyées : {n_predictions}")
    print(f"  Risques d'attrition  : {n_attrition} ({attrition_rate:.1%})")
    print(f"  API cible            : {API_URL}")

    if n_predictions and attrition_rate > 0.3:
        print("\n  [ALERTE] Taux d'attrition élevé (> 30%) — action RH recommandée.")


# ── Définition du DAG ────────────────────────────────────────────────────────

with DAG(
    dag_id="daily_predictions",
    description=(
        "Prédictions quotidiennes — envoie N employés à l'API /predict "
        "pour simuler le trafic de production"
    ),
    schedule="0 10 * * *",        # S17-8 : tous les jours à 10h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
    tags=["mlops", "ibm-hr", "predictions", "daily"],
    doc_md="""
## DAG daily_predictions

Envoie quotidiennement un échantillon d'employés à l'API de prédiction.

### Pré-requis
L'API doit être joignable : `make api` ou `make compose-up` (Docker).

### Variables configurables
| Variable env | Défaut | Description |
|---|---|---|
| `API_URL` | `http://api:8000` | URL de base de l'API |
| `N_PREDICTIONS` | `20` | Nombre d'employés envoyés par run |

### XCom exposés par `send_predictions`
- `n_predictions` — nombre de prédictions réussies
- `n_attrition` — nombre de cas attrition détectés
- `attrition_rate` — taux (float 0.0–1.0)
    """,
) as dag:

    send = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
        doc_md="Échantillonne N employés et envoie leurs features à POST /predict.",
    )

    summary = PythonOperator(
        task_id="log_summary",
        python_callable=task_log_summary,
        doc_md="Affiche le bilan quotidien depuis les XCom de send_predictions.",
    )

    send >> summary
