# Prédiction de l'attrition des employés — IBM HR Analytics

![CI](https://github.com/EthanVIEVILLE/td_orchestrationML_EthanVIEVILLE/actions/workflows/ci.yml/badge.svg)

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
make train                              # baseline LogisticRegression (C=1.0)
make train C=0.1                        # avec régularisation plus forte
make train_models                       # compare 5 modèles via GridSearchCV + MLflow
make train_models CV=3                  # run rapide (3 folds au lieu de 5)
make train_optuna                       # optimisation Optuna TPE (LightGBM, 50 trials)
make train_optuna MODEL=XGBoost N_TRIALS=100   # autre modèle, plus de trials
make evaluate                           # évalue model.joblib sur le test set
make evaluate SOURCE=registry           # évalue le @champion MLflow Registry
make api                                # lance l'API FastAPI (http://127.0.0.1:8000/docs)
make api PORT=8080                      # sur un port personnalisé
make predict_client                     # envoie 3 requêtes de test à l'API locale
make test                               # tests unitaires pytest
make pipeline                           # pipeline complet : data -> train -> evaluate
make ci                                 # CI locale : ruff + pytest + pipeline
make mlflow_ui                          # ouvre l'UI MLflow sur http://127.0.0.1:5000
make check_pipeline                     # vérifie le pipeline de données
make doctor                             # diagnostique l'environnement
```

---

## Résultats — Comparaison multi-modèles (GridSearchCV, CV=5)

Critère de sélection : **ROC-AUC** (seuil-indépendant, robuste aux classes déséquilibrées)

| Modèle | ROC-AUC | F1 (Yes) | Recall | Precision |
|---|---|---|---|---|
| **LogisticRegression** | **0.8124** | 0.4885 | 0.6809 | 0.3810 |
| RandomForest | 0.7955 | 0.4771 | 0.5532 | 0.4194 |
| LightGBM | 0.7735 | 0.4110 | 0.3191 | 0.5769 |
| MLP (PyTorch) | 0.7678 | 0.5321 | 0.6170 | 0.4688 |
| XGBoost | 0.7600 | 0.4421 | 0.4468 | 0.4375 |

> Champion actuel : **LogisticRegression** (C=0.1, class_weight=balanced) — enregistré dans le Model Registry MLflow sous l'alias `@champion`.  
> Le MLP PyTorch obtient le meilleur F1 (0.53) et un Recall élevé (0.62) grâce au `pos_weight` interne — bon profil pour le contexte RH où rater un départ coûte cher.

---

## Structure actuelle

```
.
├── data/
│   └── ibm_hr_attrition.csv       # Dataset IBM (non versionné)
├── models/
│   └── model.joblib               # Meilleur modèle (non versionné)
├── mlproject/
│   ├── config.py                  # DataConfig, MLflowConfig, DeployStage + aliases
│   ├── data.py                    # Chargement & split 80/20 stratifié
│   ├── features.py                # Préprocessing (StandardScaler + OneHotEncoder)
│   ├── evaluation.py              # Calcul des métriques (f1, roc_auc, precision, recall)
│   ├── tracking.py                # Interface centralisée MLflow (aucun autre fichier n'importe mlflow)
│   ├── models.py                  # TorchMLPClassifier — MLP PyTorch compatible sklearn Pipeline
│   ├── train.py                   # Baseline Logistic Regression + MLflow
│   ├── train_models.py            # GridSearchCV sur 5 modèles + Model Registry
│   ├── train_optuna.py            # Optimisation Optuna TPE + MLflow (un run par trial)
│   ├── evaluate.py                # Évaluation du champion sauvegardé sur le test set
│   └── api.py                     # API FastAPI : /health, /predict, /model-info
├── scripts/
│   └── predict_client.py          # Client HTTP de test pour l'API
├── tests/
│   ├── conftest.py                # Fixtures pytest + génération dataset synthétique
│   ├── generate_synthetic_data.py # Génère ibm_hr_attrition.csv synthétique (CI)
│   ├── check_pipeline.py          # Validation rapide du pipeline de données (standalone)
│   ├── test_data.py               # Tests chargement et split du dataset
│   ├── test_features.py           # Tests pipeline de préprocessing
│   ├── test_evaluation.py         # Tests calcul des métriques
│   └── test_api.py                # Tests endpoints FastAPI
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions : tests unitaires + smoke test pipeline
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
| S5 | Baseline Logistic Regression + MLflow tracking | ✅ |
| S6/S7 | Comparaison multi-modèles (GridSearchCV) + MLflow Registry | ✅ |
| S7bis | Optimisation Optuna TPE + évaluation standalone | ✅ |
| S8 | Dockerfile entraînement | 🔲 |
| S12 | API FastAPI (/predict, /health, /model-info) | ✅ |
| S15 | Tests pytest + CI GitHub Actions | ✅ |
| S12 | API FastAPI | 🔲 |
| S14 | Docker Compose | 🔲 |
| S14bis | Frontend Streamlit | 🔲 |
| S17 | DAG Airflow | 🔲 |
