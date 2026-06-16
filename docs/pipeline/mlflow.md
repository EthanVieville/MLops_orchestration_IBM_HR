# MLflow — Tracking & Registry

Toutes les interactions MLflow passent par `mlproject/tracking.py`. Aucun autre fichier n'importe `mlflow` directement.

---

## Démarrer l'UI

```bash
make mlflow_ui   # → http://127.0.0.1:5000
```

---

## Configuration

Définie dans `.env` (lue automatiquement via `python-dotenv`) :

```env
MLFLOW_TRACKING_URI=sqlite:///mlruns.db   # backend SQLite local
MLFLOW_EXPERIMENT=ibm-hr-attrition        # nom de l'expérience
MODEL_NAME=attrition-classifier           # nom dans le Registry
DEPLOY_STAGE=challenger
```

---

## Ce qui est tracké

### Paramètres (`log_params`)

Selon le script d'entraînement : hyperparamètres du modèle, nombre de folds CV, taille du dataset, etc.

### Métriques (`log_metrics`)

| Métrique | Préfixe | Description |
|----------|---------|-------------|
| `roc_auc` | `test_` | ROC-AUC sur le test set |
| `f1` | `test_` | F1-score (classe positive) |
| `f1_weighted` | `test_` | F1 pondéré |
| `precision` | `test_` | Précision (classe positive) |
| `recall` | `test_` | Recall (classe positive) |
| `cv_roc_auc` | — | ROC-AUC moyen sur les folds CV |

### Artefacts

- `confusion_matrix.png` — matrice de confusion (PNG)
- `classification_report.txt` — rapport complet par classe
- `model/` — pipeline sklearn sérialisé (chargeable via `mlflow.sklearn.load_model`)

### Tags

| Tag | Valeur | Appliqué sur |
|-----|--------|-------------|
| `optimizer` | `gridsearch` / `optuna` | runs parent |
| `model` | `LightGBM`, `XGBoost`, etc. | runs parent + enfants (train_models) |

---

## Model Registry

Le meilleur modèle de chaque run est enregistré automatiquement sous l'alias `@champion` :

```python
# Chargement du champion depuis le Registry
model = mlflow.sklearn.load_model("models:/attrition-classifier@champion")
```

```bash
make evaluate SOURCE=registry   # utilise cet alias
```

---

## API de tracking (`mlproject/tracking.py`)

| Fonction | Rôle |
|----------|------|
| `setup(experiment)` | Configure l'URI et l'expérience |
| `start_run(name, nested)` | Context manager ouvrant un run |
| `log_params(params)` | Log un dict de paramètres |
| `log_metrics(metrics, prefix)` | Log un dict de métriques (filtre NaN/Inf) |
| `log_tags(tags)` | Log un dict de tags |
| `log_model(pipeline)` | Log le pipeline comme artefact |
| `log_confusion_matrix(y_true, y_pred)` | Génère et log le PNG |
| `log_classification_report(y_true, y_pred)` | Log le rapport texte |
| `register_champion(run_id)` | Enregistre dans le Registry avec `@champion` |
