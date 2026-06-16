"""Fixtures pytest partagées — IBM HR Attrition.

Organisation :
  - pytest_configure : hook qui génère le dataset synthétique si le vrai
    CSV est absent (CI/CD, premier clone). Exécuté avant toute collecte.
  - df              : DataFrame complet (load_data())
  - trained_pipeline: Pipeline LogisticRegression entraîné rapidement
                      (utilisé par les tests API et d'évaluation)
"""
from __future__ import annotations

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlproject.config import DATA_PATH, RANDOM_STATE
from mlproject.features import build_preprocessor


# ── Hook pytest — génération du dataset synthétique ──────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Génère le dataset synthétique si le vrai CSV n'est pas présent.

    Ce hook s'exécute avant toute collecte de tests, garantissant que
    load_data() fonctionne même en CI (où le CSV réel n'est pas versionné).
    """
    if not DATA_PATH.exists():
        # Import local pour éviter une dépendance circulaire au niveau module
        from tests.generate_synthetic_data import generate
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DATA_PATH, index=False)
        print(f"\n[conftest] Dataset synthétique créé : {DATA_PATH}")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def df():
    """DataFrame IBM HR complet (réel ou synthétique selon l'environnement)."""
    from mlproject.data import load_data
    return load_data()


@pytest.fixture(scope="session")
def trained_pipeline(df):
    """Pipeline LogisticRegression entraîné rapidement sur l'ensemble de données.

    Utilisé par les tests qui ont besoin d'un modèle valide sans lancer
    un entraînement complet (API /predict, évaluation).
    max_iter=200 suffit pour converger sur 400 lignes synthétiques.
    """
    from mlproject.data import split

    X_train, _, y_train, _ = split(df)
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=200,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture()
def saved_model(trained_pipeline, tmp_path):
    """Sauvegarde le pipeline dans un répertoire temporaire.

    Retourne le Path du répertoire (équivalent MODEL_DIR dans les tests API).
    """
    model_file = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, model_file)
    return tmp_path
