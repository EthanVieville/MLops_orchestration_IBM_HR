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
make check                          # vérifie le pipeline de données
make doctor                         # diagnostique l'environnement
uv run python tests/check_pipeline.py  # même chose, sans make
```

---

## Structure actuelle

```
.
├── data/
│   └── ibm_hr_attrition.csv       # Dataset IBM (non versionné)
├── mlproject/
│   ├── config.py                  # Configuration centrale (colonnes, paths, MLflow)
│   ├── data.py                    # Chargement & split 80/20 stratifié
│   └── features.py                # Préprocessing (StandardScaler + OneHotEncoder)
├── tests/
│   └── check_pipeline.py          # Validation du pipeline de données
├── .env                           # Variables d'environnement (non versionné)
├── .env.example                   # Template .env
├── Makefile
└── pyproject.toml
```

---
