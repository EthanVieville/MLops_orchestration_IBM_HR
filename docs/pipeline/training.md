# Entraînement

Trois modes d'entraînement disponibles, du plus simple au plus avancé.

---

## 1. Baseline — Logistic Regression

```bash
make train               # C=1.0, max_iter=1000
make train C=0.1         # régularisation plus forte
```

`mlproject/train.py` — entraîne une LogisticRegression avec `class_weight="balanced"` et log le run dans MLflow.

**Quand l'utiliser :** vérification rapide du pipeline, établissement d'une référence de performance.

---

## 2. Comparaison multi-modèles — GridSearchCV

```bash
make train_models          # 5 modèles, CV=5
make train_models CV=3     # plus rapide
```

`mlproject/train_models.py` — compare 5 familles de modèles via `GridSearchCV` + `StratifiedKFold`.

### Modèles comparés

| Modèle | Librairie | Gestion déséquilibre |
|--------|-----------|----------------------|
| LogisticRegression | scikit-learn | `class_weight="balanced"` |
| RandomForest | scikit-learn | `class_weight="balanced"` |
| XGBoost | xgboost | `scale_pos_weight` |
| LightGBM | lightgbm | `class_weight="balanced"` |
| MLP | PyTorch | `pos_weight` (BCEWithLogitsLoss) |

### Structure MLflow

```
ibm-hr-attrition/
└── train_models_20260615_143022      ← run parent
    ├── LogisticRegression            ← run enfant (params + métriques + artefacts)
    ├── RandomForest
    ├── XGBoost
    ├── LightGBM
    └── MLP
```

À la fin, le **champion** (meilleur ROC-AUC test) est sauvegardé dans `models/model.joblib` et enregistré dans le MLflow Model Registry sous l'alias `@champion`.

---

## 3. Optimisation Optuna — TPE Bayesian Search

```bash
make train_optuna                          # LightGBM, 50 trials
make train_optuna MODEL=XGBoost N_TRIALS=100
```

`mlproject/train_optuna.py` — utilise l'algorithme **TPE** (Tree-structured Parzen Estimator) pour explorer intelligemment l'espace des hyperparamètres.

### Optuna vs GridSearchCV

| | GridSearchCV | Optuna TPE |
|--|-------------|------------|
| Espace de recherche | Grille fixe | Continu (log-scale possible) |
| Stratégie | Exhaustive | Bayésienne (exploite les résultats précédents) |
| Efficacité | O(n×m) combinaisons | Converge en ~50 trials |
| Exemple learning_rate | `[0.05, 0.1]` | float dans `[0.001, 0.3]` |

### Structure MLflow

```
ibm-hr-attrition/
└── optuna_LightGBM_20260615_...    ← run parent (tags: optimizer=optuna, model=LightGBM)
    ├── trial_000                   ← params + cv_roc_auc + cv_roc_auc_std
    ├── trial_001
    └── ...
```

!!! tip "Filtrer dans MLflow UI"
    Les runs Optuna sont tagués `optimizer=optuna` et `model=LightGBM`.
    Les runs GridSearch sont tagués `optimizer=gridsearch`.
    Utilise le filtre par tag dans l'UI pour comparer les deux approches sur un même modèle.

---

## 4. Évaluation du champion

```bash
make evaluate                    # charge models/model.joblib
make evaluate SOURCE=registry    # charge @champion depuis MLflow Registry
```

`mlproject/evaluate.py` — évalue le modèle sauvegardé sur le test set et log un run MLflow dédié (prefix `eval_`).

---

## MLP PyTorch

Le MLP est un wrapper scikit-learn (`BaseEstimator + ClassifierMixin`) qui encapsule un réseau PyTorch, utilisable directement dans un `Pipeline` et dans `GridSearchCV`/`cross_val_score`.

```
Architecture : Input → [Linear → ReLU → Dropout] × N → Linear(1)
Loss         : BCEWithLogitsLoss avec pos_weight = n_neg / n_pos
Optimiseur   : Adam
```

Paramètres configurables : `hidden_sizes`, `lr`, `epochs`, `batch_size`, `dropout`.
