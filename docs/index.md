# IBM HR Attrition — Pipeline MLOps

Projet fil rouge ESGI — prédiction du risque de départ des employés.

---

## Problématique

> **Un employé va-t-il quitter l'entreprise ?**

Anticiper les départs permet aux équipes RH de prioriser leurs actions de rétention sur les profils à risque, réduisant les coûts de recrutement et la perte de compétences.

| Classe | Signification |
|--------|--------------|
| **0** | L'employé **reste** |
| **1** | L'employé **quitte** (attrition) |

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- **1 470 employés**, 35 colonnes brutes (4 colonnes constantes supprimées)
- **30 features exploitables** : 23 numériques + 7 catégorielles
- **Déséquilibre** : 83.9 % No / 16.1 % Yes → les modèles utilisent `class_weight="balanced"`

---

## Architecture du pipeline

```
CSV brut
  └─► data.py          Chargement, nettoyage, split 80/20 stratifié
        └─► features.py    StandardScaler + OneHotEncoder (ColumnTransformer)
              └─► train*.py   Entraînement + GridSearchCV / Optuna
                    └─► tracking.py  MLflow (métriques, artefacts, Model Registry)
                          └─► api.py      FastAPI /predict
```

---

## Résultats

| Modèle | ROC-AUC | F1 (Yes) | Recall |
|--------|---------|---------|--------|
| **LogisticRegression** | **0.8124** | 0.4885 | 0.6809 |
| RandomForest | 0.7955 | 0.4771 | 0.5532 |
| LightGBM | 0.7735 | 0.4110 | 0.3191 |
| MLP (PyTorch) | 0.7678 | 0.5321 | 0.6170 |
| XGBoost | 0.7600 | 0.4421 | 0.4468 |

!!! tip "Critère de sélection"
    Le **ROC-AUC** est le critère principal car il est indépendant du seuil de décision et robuste aux classes déséquilibrées. Le **Recall** est surveillé en priorité : rater un départ coûte plus cher que surveiller un employé qui reste.

---

## Avancement

| Étape | Objectif | Statut |
|-------|----------|--------|
| S0 | Configuration dataset + pipeline de données | ✅ |
| S5 | Baseline LogisticRegression + MLflow | ✅ |
| S6/S7 | Comparaison multi-modèles (GridSearchCV) + Registry | ✅ |
| S7bis | Optimisation Optuna TPE + évaluation standalone | ✅ |
| S12 | API FastAPI (/predict, /health, /model-info) | ✅ |
| S15 | Tests pytest + CI GitHub Actions | ✅ |
| S8 | Dockerfile entraînement | 🔲 |
| S14 | Docker Compose | 🔲 |
| S14bis | Frontend Streamlit | 🔲 |
| S17 | DAG Airflow | 🔲 |
