# Prédiction de l'attrition des employés — IBM HR Analytics

Pipeline MLOps de classification binaire pour prédire si un employé va quitter l'entreprise.

**Auteur** : VIEVILLE Ethan

---

## Problématique

> **Un employé va-t-il quitter l'entreprise ?** (`Attrition = Yes/No`)

Anticiper les départs permet aux RH de prioriser les actions de rétention sur les profils à risque, réduisant les coûts de recrutement et la perte de compétences.

- `1` — l'employé quitte l'entreprise
- `0` — l'employé reste

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- 1 470 employés, 31 colonnes exploitables (4 colonnes constantes supprimées)
- Target : `Attrition` — 83.9% No / 16.1% Yes (dataset déséquilibré)
- 23 features numériques, 7 features catégorielles

---

## Installation

```bash
make install
```

## Commandes

```bash
make install                            # installe les dépendances
make train                              # entraîne la baseline (C=1.0, MAX_ITER=1000)
make train C=0.1                        # avec régularisation plus forte
make check_pipeline                     # vérifie le pipeline de données
make doctor                             # diagnostique l'environnement
uv run python tests/check_pipeline.py  # check pipeline sans make
```

---

## Résultats baseline (Logistic Regression)

| Métrique | Score |
|---|---|
| ROC-AUC | 0.8032 |
| F1 (Yes) | 0.4511 |
| F1 weighted | 0.7775 |
| Recall (Yes) | 0.6383 |

> Dataset déséquilibré (16% Yes) — `class_weight="balanced"` activé. ROC-AUC solide, F1 sur la classe minoritaire à améliorer avec XGBoost/LightGBM (S6/S7).

---

## Structure actuelle

```
.
├── data/
│   └── ibm_hr_attrition.csv       # Dataset IBM (non versionné)
├── models/
│   └── model.joblib               # Modèle entraîné (non versionné)
├── mlproject/
│   ├── config.py                  # Configuration centrale (colonnes, paths, MLflow)
│   ├── data.py                    # Chargement & split 80/20 stratifié
│   ├── features.py                # Préprocessing (StandardScaler + OneHotEncoder)
│   └── train.py                   # Baseline Logistic Regression
├── tests/
│   └── check_pipeline.py          # Validation du pipeline de données
├── .env                           # Variables d'environnement (non versionné)
├── .env.example                   # Template .env
├── Makefile
└── pyproject.toml
```

---

## Avancement

| Séance | Objectif | Statut |
|---|---|---|
| S0 | Configuration dataset IBM + pipeline de données | ✅ |
| S5 | Baseline Logistic Regression | ✅ |
| S6 | Optimisation Optuna | 🔲 |
| S7 | GridSearchCV + SHAP | 🔲 |
| S8 | Dockerfile entraînement | 🔲 |
| S12 | API FastAPI | 🔲 |
| S14 | Docker Compose | 🔲 |
| S14bis | Frontend Streamlit | 🔲 |
| S17 | DAG Airflow | 🔲 |
