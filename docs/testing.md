# Tests & CI/CD

## Tests unitaires (pytest)

```bash
make test    # lance pytest
```

### Couverture

| Fichier | Tests | Ce qui est vérifié |
|---------|-------|--------------------|
| `test_data.py` | 7 | Shape, encodage target, colonnes supprimées, ratio split, stratification, pas d'overlap |
| `test_features.py` | 4 | Type ColumnTransformer, pas de NaN, shape train/test cohérente |
| `test_evaluation.py` | 5 | Clés retournées, valeurs dans [0,1], float, cas oracle, sur pipeline réel |
| `test_api.py` | 8 | `/health`, `/model-info` (avec/sans modèle), `/predict` (200/503/422), lifespan |

### Dataset synthétique en CI

Le vrai CSV (`data/ibm_hr_attrition.csv`) n'est pas versionné. Le `conftest.py` génère automatiquement un dataset synthétique de 400 lignes **avant** toute collecte de tests via le hook `pytest_configure` :

```python
# tests/conftest.py
def pytest_configure(config):
    if not DATA_PATH.exists():
        generate().to_csv(DATA_PATH, index=False)
```

Même schéma que le dataset réel (35 colonnes, mêmes types, mêmes valeurs possibles), avec ~16 % d'attrition pour reproduire le déséquilibre.

---

## CI/CD — GitHub Actions

Le workflow `.github/workflows/ci.yml` se déclenche à chaque `push` et `pull_request` sur `main`.

### Job 1 : Tests unitaires

```
checkout → uv (cache) → uv sync --extra dev → ruff check → pytest
```

- **ruff** vérifie le style et les imports de `mlproject/`, `tests/`, `scripts/`
- **pytest** génère automatiquement le dataset synthétique et lance les 24 tests

### Job 2 : Smoke test pipeline (si Job 1 passe)

```
checkout → uv (cache) → uv sync
  → generate_synthetic_data.py
  → check_pipeline.py
  → train --max-iter 300
  → evaluate
  → uvicorn + curl /health + /model-info
  → predict_client.py
```

Ce job valide que le pipeline complet fonctionne de bout en bout : de la génération des données jusqu'à la prédiction via l'API.

### Cache uv

```yaml
- uses: astral-sh/setup-uv@v4
  with:
    enable-cache: true
    cache-dependency-glob: "uv.lock"
```

Le virtualenv est mis en cache entre les runs — les installations ne se répètent que si `uv.lock` change.

---

## CI locale

```bash
make ci     # ruff + pytest + pipeline complet
```

Equivalent local du workflow GitHub Actions. À lancer avant de pousser pour éviter les allers-retours :

```bash
make ci && git push
```

| Commande | Rôle |
|----------|------|
| `make test` | pytest seul (rapide) |
| `make pipeline` | data → train → evaluate (sans tests) |
| `make ci` | ruff + pytest + pipeline (tout) |
