# Prédiction de l'attrition des employés — IBM HR Analytics

Pipeline MLOps complet de classification binaire pour prédire si un employé va quitter l'entreprise.

**Auteur** : Ethan Viéville

---

## Problématique

> **Un employé va-t-il quitter l'entreprise ?** (`Attrition = Yes/No`)

Anticiper les départs permet aux RH de prioriser les actions de rétention sur les profils à risque, réduisant ainsi les coûts de recrutement et la perte de compétences.

- `1` — l'employé quitte l'entreprise
- `0` — l'employé reste

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance** — [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

- 1 470 employés, 35 colonnes
- Target : `Attrition` (Yes / No → 1 / 0)
- Mix de features numériques (âge, salaire, ancienneté, satisfaction...) et catégorielles (département, rôle, niveau d'éducation...)

---

## Stack technique prévue

| Couche | Outil |
|---|---|
| ML | scikit-learn, XGBoost, LightGBM |
| Tracking | MLflow |
| Optimisation | Optuna |
| Explainabilité | SHAP |
| API | FastAPI |
| Frontend | Streamlit |
| Conteneurisation | Docker + Docker Compose |
| Orchestration | Apache Airflow |
| Package manager | uv |

---

## Installation

```bash
# Depuis Git Bash
make install
```

Ou directement :

```bash
uv sync --extra dev
```

---

## Structure

```
.
├── data/                        # Dataset CSV (non versionné)
├── models/                      # Modèles entraînés (non versionnés)
├── mlproject/
│   ├── config.py                # Configuration centrale (colonnes, paths, MLflow)
│   ├── data.py                  # Chargement & split 80/20 stratifié
│   ├── features.py              # Préprocessing (StandardScaler + OneHotEncoder)
│   ├── evaluation.py            # Métriques & SHAP summary plot
│   ├── train.py                 # Baseline logistic regression + MLflow   (S5)
│   ├── train_optuna.py          # Optimisation Optuna                     (S6)
│   ├── train_models.py          # GridSearchCV + SHAP                     (S7)
│   └── api.py                   # API FastAPI /predict                    (S12)
├── frontend/app.py              # Interface Streamlit                     (S14bis)
├── docker/                      # Dockerfiles train / api / frontend
├── docker-compose.yml           # Stack complète                          (S14)
├── dags/retrain_dag.py          # DAG Airflow retraining                  (S17)
└── Makefile                     # Commandes du projet
```

---
