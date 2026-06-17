"""Sauvegarde des modèles avec métadonnées (.meta.json).

Chaque modèle entraîné produit deux fichiers dans MODEL_DIR :
  model_{label}_{timestamp}.joblib     — pipeline sklearn sérialisé
  model_{label}_{timestamp}.meta.json  — type, params, métriques, run_id

Le fichier model.joblib reste le champion (rétrocompatibilité API).
Un model.meta.json est créé en parallèle quand save_as_champion=True,
permettant à l'API de l'afficher avec ses vraies métadonnées au démarrage.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from mlproject.config import MODEL_DIR


def save_model_snapshot(
    pipeline: Any,
    label: str,
    params: dict,
    metrics: dict,
    run_id: str = "",
    *,
    save_as_champion: bool = False,
) -> Path:
    """Sauvegarde le pipeline + un .meta.json horodaté dans MODEL_DIR.

    Args:
        pipeline:         Pipeline sklearn entraîné.
        label:            Nom lisible du modèle (ex. "LightGBM (n_est=200, lr=0.1)").
        params:           Hyperparamètres retenus (dict str→any).
        metrics:          Métriques test (roc_auc, f1, …).
        run_id:           ID du run MLflow associé.
        save_as_champion: Si True, écrit aussi model.meta.json (champion courant).

    Returns:
        Chemin du fichier .joblib créé.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe      = re.sub(r"[^a-zA-Z0-9_]", "_", label)[:60]
    stem      = f"model_{safe}_{ts}"

    joblib_path = MODEL_DIR / f"{stem}.joblib"
    meta_path   = MODEL_DIR / f"{stem}.meta.json"

    model_type  = type(pipeline.steps[-1][1]).__name__
    key_metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    parts: list[str] = []
    if (roc := key_metrics.get("roc_auc")):
        parts.append(f"ROC-AUC={roc:.3f}")
    if (f1 := key_metrics.get("f1")):
        parts.append(f"F1={f1:.3f}")
    display = f"{label} | {' '.join(parts)}" if parts else label

    meta = {
        "filename":     f"{stem}.joblib",
        "model_type":   model_type,
        "display_name": display,
        "label":        label,
        "params":       {k: str(v) for k, v in params.items()},
        "metrics":      key_metrics,
        "run_id":       run_id,
        "trained_at":   datetime.now().isoformat(),
    }

    joblib.dump(pipeline, joblib_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_as_champion:
        (MODEL_DIR / "model.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return joblib_path
