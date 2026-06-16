# Données & Features

## Chargement (`mlproject/data.py`)

```python
from mlproject.data import load_data, split

df = load_data()              # 1 470 lignes × 31 colonnes
X_train, X_test, y_train, y_test = split(df)  # 80/20 stratifié
```

### Transformations appliquées

| Étape | Détail |
|-------|--------|
| Lecture CSV | `data/ibm_hr_attrition.csv` |
| Suppression colonnes constantes | `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours` |
| Encodage target | `"Yes"` → `1`, `"No"` → `0` |
| Split stratifié | 80 % train / 20 % test, `random_state=42` |

!!! info "Pourquoi stratifié ?"
    Avec seulement 16.1 % de positifs, un split aléatoire pourrait produire un test set sans aucun positif. Le split stratifié garantit la même proportion dans les deux ensembles.

---

## Features

### 23 features numériques

| Feature | Description | Plage |
|---------|-------------|-------|
| `Age` | Âge | 18–65 |
| `DailyRate` | Taux journalier | 102–1499 |
| `DistanceFromHome` | Distance domicile-travail (km) | 1–29 |
| `Education` | Niveau d'éducation (1=Below College … 5=Doctor) | 1–5 |
| `EnvironmentSatisfaction` | Satisfaction environnement | 1–4 |
| `HourlyRate` | Taux horaire | 30–100 |
| `JobInvolvement` | Implication au travail | 1–4 |
| `JobLevel` | Niveau hiérarchique | 1–5 |
| `JobSatisfaction` | Satisfaction au travail | 1–4 |
| `MonthlyIncome` | Revenu mensuel (USD) | 1009–19999 |
| `MonthlyRate` | Taux mensuel | 2094–26999 |
| `NumCompaniesWorked` | Entreprises précédentes | 0–9 |
| `PercentSalaryHike` | Augmentation salariale (%) | 11–25 |
| `PerformanceRating` | Évaluation annuelle | 3–4 |
| `RelationshipSatisfaction` | Satisfaction relationnelle | 1–4 |
| `StockOptionLevel` | Options sur actions | 0–3 |
| `TotalWorkingYears` | Expérience totale (ans) | 0–40 |
| `TrainingTimesLastYear` | Formations l'an dernier | 0–6 |
| `WorkLifeBalance` | Équilibre pro/perso | 1–4 |
| `YearsAtCompany` | Ancienneté dans l'entreprise | 0–40 |
| `YearsInCurrentRole` | Années dans le poste actuel | 0–18 |
| `YearsSinceLastPromotion` | Depuis la dernière promotion | 0–15 |
| `YearsWithCurrManager` | Avec le manager actuel | 0–17 |

### 7 features catégorielles

| Feature | Valeurs possibles |
|---------|------------------|
| `BusinessTravel` | `Non-Travel`, `Travel_Rarely`, `Travel_Frequently` |
| `Department` | `Sales`, `Research & Development`, `Human Resources` |
| `EducationField` | `Life Sciences`, `Medical`, `Marketing`, `Technical Degree`, `Human Resources`, `Other` |
| `Gender` | `Male`, `Female` |
| `JobRole` | `Sales Executive`, `Research Scientist`, `Laboratory Technician`, `Manufacturing Director`, `Healthcare Representative`, `Manager`, `Sales Representative`, `Research Director`, `Human Resources` |
| `MaritalStatus` | `Single`, `Married`, `Divorced` |
| `OverTime` | `Yes`, `No` |

---

## Préprocessing (`mlproject/features.py`)

```python
from mlproject.features import build_preprocessor

preprocessor = build_preprocessor()
# Retourne un ColumnTransformer :
#   - StandardScaler  → features numériques
#   - OneHotEncoder   → features catégorielles (drop='first')
```

Le `ColumnTransformer` est encapsulé dans chaque `Pipeline` sklearn — il est ajusté (`fit`) uniquement sur `X_train` pour éviter toute fuite de données vers le test set.
