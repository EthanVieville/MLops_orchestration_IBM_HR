"""Client de test pour l'API FastAPI — IBM HR Attrition.

Envoie quelques payloads à une instance locale de l'API (`make api`)
et affiche les réponses de /health, /predict et /model-info.

Les payloads sont échantillonnés depuis le dataset IBM HR :
leurs colonnes correspondent exactement au schéma Features de l'API.

Lancement :
    make predict_client          # utilise l'URL par défaut (http://127.0.0.1:8000)
    uv run python scripts/predict_client.py
    uv run python scripts/predict_client.py --url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import logging

import httpx

from mlproject.config import API_URL, TARGET
from mlproject.data import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

N_SAMPLES = 3


def build_payloads(n: int = N_SAMPLES) -> list[dict]:
    """Construit n payloads de test à partir du dataset.

    On retire la colonne cible et on convertit chaque ligne en dict JSON natif.
    random_state=42 garantit des résultats reproductibles entre appels.
    """
    features = load_data().drop(columns=[TARGET])
    sample   = features.sample(n=n, random_state=42)
    return [json.loads(row.to_json()) for _, row in sample.iterrows()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=API_URL,
        help="URL de base de l'API (défaut: %(default)s)",
    )
    args = parser.parse_args()

    payloads = build_payloads()
    logger.info("Envoi de %d requêtes vers %s", len(payloads), args.url)

    with httpx.Client(base_url=args.url, timeout=10.0) as client:

        # GET /health
        health = client.get("/health")
        logger.info("GET /health       -> %s %s", health.status_code, health.json())

        # POST /predict — un appel par payload
        for i, payload in enumerate(payloads):
            response = client.post("/predict", json=payload)
            logger.info(
                "POST /predict (#%d) -> %s %s",
                i, response.status_code, response.json(),
            )

        # GET /model-info
        info = client.get("/model-info")
        logger.info("GET /model-info   -> %s %s", info.status_code, info.json())


if __name__ == "__main__":
    main()
