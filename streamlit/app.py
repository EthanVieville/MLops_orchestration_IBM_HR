"""IBM HR Attrition — Dashboard MLOps (Streamlit)

Six onglets :
  1. Contexte       — problématique IBM, stats dataset, facteurs de risque
  2. Prédiction     — formulaire employé unique + batch CSV → API /predict
  3. Modèles        — runs MLflow, comparaison métriques, meilleur run
  4. Artefacts      — matrice de confusion + classification report par run
  5. Entraînement   — statut modèle, lancer un entraînement local ou voir cmds Docker
  6. Exploration    — distributions, corrélations, stats descriptives sur le dataset
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IBM HR Attrition · MLOps",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_API_URL      = os.getenv("API_URL",             "http://127.0.0.1:8000")
DEFAULT_MLF_URI      = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
DEFAULT_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT",   "ibm-hr-attrition")
DEFAULT_AIRFLOW_URL  = os.getenv("AIRFLOW_URL",         "http://airflow:8080")
QUALITY_THRESHOLD    = float(os.getenv("QUALITY_THRESHOLD", "0.35"))

NUMERIC_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education",
    "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "BusinessTravel":  ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department":      ["Sales", "Research & Development", "Human Resources"],
    "EducationField":  ["Life Sciences", "Medical", "Marketing",
                        "Technical Degree", "Human Resources", "Other"],
    "Gender":          ["Male", "Female"],
    "JobRole":         [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources",
    ],
    "MaritalStatus":   ["Single", "Married", "Divorced"],
    "OverTime":        ["Yes", "No"],
}

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("api_url",     DEFAULT_API_URL),
    ("mlf_uri",     DEFAULT_MLF_URI),
    ("airflow_url", DEFAULT_AIRFLOW_URL),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=20)
def api_health(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=10)
def api_model_info(url: str) -> dict:
    try:
        r = requests.get(f"{url}/model-info", timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


@st.cache_data(ttl=5)
def api_list_models(url: str) -> list[dict]:
    try:
        r = requests.get(f"{url}/models", timeout=3)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def api_predict(url: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{url}/predict", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"Erreur API {r.status_code} : {r.text[:300]}")
        return None
    except Exception as exc:
        st.error(f"Connexion impossible : {exc}")
        return None


@st.cache_data(ttl=60)
def mlflow_runs(tracking_uri: str, experiment: str) -> pd.DataFrame:
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        return mlflow.search_runs(
            experiment_names=[experiment],
            order_by=["metrics.test_roc_auc DESC"],
        )
    except Exception:
        return pd.DataFrame()


def mlflow_artifact(tracking_uri: str, run_id: str, artifact_path: str) -> bytes | None:
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=tracking_uri)
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run_id, artifact_path, tmp)
            return Path(local).read_bytes()
    except Exception:
        return None


@st.cache_data(ttl=30)
def airflow_health(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=30)
def airflow_dag_runs(url: str, dag_id: str, limit: int = 20) -> list[dict]:
    try:
        r = requests.get(
            f"{url}/api/v1/dags/{dag_id}/dagRuns",
            params={"order_by": "-start_date", "limit": limit},
            auth=("airflow", "airflow"),
            timeout=5,
        )
        return r.json().get("dag_runs", []) if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=30)
def airflow_xcom(url: str, dag_id: str, dag_run_id: str, task_id: str, key: str) -> Any:
    import json as _json
    try:
        r = requests.get(
            f"{url}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
            f"/taskInstances/{task_id}/xcomEntries/{key}",
            auth=("airflow", "airflow"),
            timeout=3,
        )
        if r.status_code != 200:
            return None
        value = r.json().get("value")
        if isinstance(value, str):
            try:
                return _json.loads(value)
            except Exception:
                return value
        return value
    except Exception:
        return None


def run_duration(start: str, end: str) -> str:
    if not start or not end:
        return "–"
    try:
        dt_s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        dt_e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        secs = int((dt_e - dt_s).total_seconds())
        return f"{secs // 60}m {secs % 60}s"
    except Exception:
        return "–"


def fmt_date(iso: str) -> str:
    return iso[:16].replace("T", " ") if iso else "–"


def state_icon(state: str) -> str:
    return {"success": "✅", "failed": "❌", "running": "⏳"}.get(state, "⏸️")


def risk_gauge(prob: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 1.8))
    for left, right, color in [(0, 0.33, "#2ecc71"), (0.33, 0.66, "#f39c12"), (0.66, 1.0, "#e74c3c")]:
        ax.barh(0, right - left, left=left, color=color, height=0.5, alpha=0.85)
    ax.barh(0, 0.012, left=max(prob - 0.006, 0), color="black", height=0.8)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"], fontsize=9)
    ax.set_yticks([])
    ax.set_title("Jauge de risque", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 👔 IBM HR Attrition")
    st.caption("Dashboard MLOps · ESGI · Ethan Viéville")
    st.divider()

    api_url = st.text_input("🔗 URL API", value=st.session_state.api_url)
    st.session_state.api_url = api_url

    if api_health(api_url):
        st.success("API ✅ connectée")
        info = api_model_info(api_url)

        if info.get("loaded"):
            # ── Modèle actif ─────────────────────────────────────────────────
            st.markdown("**Modèle actif**")
            st.info(info.get("display_name", info.get("model_type", "?")))
            mets = info.get("metrics", {})
            if mets:
                col_r, col_f = st.columns(2)
                if (roc := mets.get("roc_auc")):
                    col_r.metric("ROC-AUC", f"{roc:.3f}")
                if (f1 := mets.get("f1")):
                    col_f.metric("F1", f"{f1:.3f}")
        else:
            st.warning("Modèle non chargé")

        # ── Sélecteur de modèle ───────────────────────────────────────────────
        available_models = api_list_models(api_url)
        if available_models:
            st.divider()
            st.markdown("**Changer de modèle**")

            current_file = info.get("filename", "model.joblib")
            model_labels = [m["display_name"] for m in available_models]
            current_idx  = next(
                (i for i, m in enumerate(available_models) if m["filename"] == current_file),
                0,
            )

            chosen_label = st.selectbox(
                "Modèles disponibles",
                model_labels,
                index=current_idx,
                label_visibility="collapsed",
            )
            chosen_model = next(m for m in available_models if m["display_name"] == chosen_label)

            if st.button("↺ Charger ce modèle", use_container_width=True,
                         disabled=(chosen_model["filename"] == current_file)):
                try:
                    r = requests.post(
                        f"{api_url}/models/switch",
                        json={"filename": chosen_model["filename"]},
                        timeout=5,
                    )
                    if r.status_code == 200:
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Erreur {r.status_code} : {r.text[:150]}")
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.error("API ❌ inaccessible")

    st.divider()
    mlf_uri = st.text_input("📊 MLflow URI", value=st.session_state.mlf_uri)
    st.session_state.mlf_uri = mlf_uri
    mlf_exp = st.text_input("Expérience MLflow", value=DEFAULT_EXPERIMENT)

    if st.button("🔄 Vider le cache"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # ── Accès rapide — URLs déduites de l'hôte courant ───────────────────────
    st.markdown("**🔗 Accès rapide**")
    try:
        _host = st.context.headers.get("host", "localhost").split(":")[0]
    except Exception:
        _host = "localhost"

    st.link_button("📖 API Docs", f"http://{_host}:8000/docs", use_container_width=True)
    st.link_button("🔄 Airflow UI", f"http://{_host}:8080", use_container_width=True)
    st.link_button("📊 MLflow UI", f"http://{_host}:5000", use_container_width=True)
    st.caption("MLflow UI : `make compose-monitoring`")

    st.divider()
    st.caption("Stack : scikit-learn · LightGBM · XGBoost · PyTorch · MLflow · FastAPI · Streamlit · Docker")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Contexte",
    "🔮 Prédiction",
    "📊 Modèles",
    "🖼️ Artefacts",
    "⚗️ Entraînement",
    "🔍 Exploration",
    "🔄 Orchestration",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Contexte
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Prédiction du risque d'attrition des employés")
    st.subheader("IBM HR Analytics Employee Attrition Dataset")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employés", "1 470")
    c2.metric("Taux d'attrition", "16.1 %")
    c3.metric("Features", "30  (23 num + 7 cat)")
    c4.metric("Tâche ML", "Classification binaire")

    st.divider()

    col_desc, col_pie = st.columns([3, 2])

    with col_desc:
        st.markdown("""
### Contexte métier

IBM souhaite identifier en amont les employés susceptibles de **quitter l'entreprise**,
afin de mettre en place des actions préventives ciblées (augmentation, formation, flexibilité…).

**Problème :**  Classification binaire
- `0` → l'employé **reste**
- `1` → l'employé **part** (attrition)

**Déséquilibre des classes :** 83.9 % restent / 16.1 % quittent
→ les modèles utilisent `class_weight="balanced"` pour ne pas ignorer les départs.

**Métriques prioritaires :**
| Métrique | Pourquoi |
|---|---|
| **ROC-AUC** | Discrimine bien les 2 classes quelle que soit le seuil |
| **F1 (classe Yes)** | Équilibre précision et rappel sur la classe minoritaire |
| Accuracy seule | ❌ Trompeuse — 84 % en prédisant "reste" tout le temps |

**5 modèles comparés :** LogisticRegression · RandomForest · XGBoost · LightGBM · PyTorch MLP
        """)

    with col_pie:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [83.9, 16.1],
            labels=["Reste (No)", "Part (Yes)"],
            colors=["#2ecc71", "#e74c3c"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2),
        )
        ax.set_title("Distribution de l'attrition", fontweight="bold")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.subheader("Facteurs de risque clés (littérature RH)")

    risk_factors = [
        ("🔴 OverTime = Yes",        "Heures supp. régulières · multiplie le risque par ~3"),
        ("🔴 MonthlyIncome bas",     "Faible salaire → cherche mieux ailleurs"),
        ("🟡 YearsAtCompany < 3",    "Période critique : les 3 premières années sont décisives"),
        ("🟡 JobSatisfaction ≤ 2",   "Insatisfaction au travail → risque fortement accru"),
        ("🟡 DistanceFromHome > 15", "Trajet long et usant · 15+ km augmente le risque"),
        ("🟡 StockOptionLevel = 0",  "Pas d'options → aucun levier de fidélisation financier"),
        ("🟠 MaritalStatus = Single","Célibataires plus mobiles professionnellement"),
        ("🟠 Travel_Frequently",     "Déplacements pro fréquents · facteur d'épuisement"),
        ("🟠 EnvironmentSatisfaction ≤ 2", "Mauvais cadre de travail"),
    ]

    cols_rf = st.columns(3)
    for i, (factor, desc) in enumerate(risk_factors):
        with cols_rf[i % 3]:
            st.info(f"**{factor}**\n\n{desc}")

    st.divider()
    st.subheader("Architecture MLOps")
    st.code("""
CSV (IBM HR) ──► Feature Engineering ──► 5 modèles ML ──► MLflow Tracking
                                                │
                                                ▼
                                       Model Registry @champion
                                                │
                                                ▼
                                        FastAPI /predict ◄──── Ce Dashboard Streamlit
    """, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Prédiction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("🔮 Prédiction du risque d'attrition")

    mode = st.radio("Mode", ["👤 Employé unique", "📋 Batch CSV"], horizontal=True)
    st.divider()

    # ── Employé unique ────────────────────────────────────────────────────────
    if mode == "👤 Employé unique":

        with st.expander("👤 Profil personnel", expanded=True):
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            age       = r1c1.slider("Âge", 18, 65, 35)
            gender    = r1c2.selectbox("Genre", CATEGORICAL_OPTIONS["Gender"])
            marital   = r1c3.selectbox("Situation familiale", CATEGORICAL_OPTIONS["MaritalStatus"])
            education = r1c4.slider("Éducation (1=Bac-2 … 5=Doctorat)", 1, 5, 3)
            edu_field = st.selectbox("Domaine d'études", CATEGORICAL_OPTIONS["EducationField"])

        with st.expander("💼 Poste et carrière", expanded=True):
            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            job_role   = r2c1.selectbox("Intitulé de poste", CATEGORICAL_OPTIONS["JobRole"])
            department = r2c2.selectbox("Département", CATEGORICAL_OPTIONS["Department"])
            job_level  = r2c3.slider("Niveau hiérarchique (1–5)", 1, 5, 2)
            overtime   = r2c4.selectbox("Heures supplémentaires", CATEGORICAL_OPTIONS["OverTime"])
            job_inv    = r2c1.slider("Implication au travail (1–4)", 1, 4, 3)
            job_sat    = r2c2.slider("Satisfaction au travail (1–4)", 1, 4, 3)
            perf_rat   = r2c3.slider("Évaluation annuelle (1–4)", 1, 4, 3)
            num_comp   = r2c4.slider("Nb entreprises précédentes", 0, 9, 2)

        with st.expander("📅 Ancienneté & Formations", expanded=True):
            r3c1, r3c2, r3c3, r3c4 = st.columns(4)
            total_years   = r3c1.slider("Années d'expérience totale", 0, 40, 10)
            years_company = r3c2.slider("Années dans l'entreprise", 0, 40, 5)
            years_role    = r3c3.slider("Années dans le rôle actuel", 0, 18, 3)
            years_promo   = r3c4.slider("Années depuis dernière promo", 0, 15, 1)
            years_manager = r3c1.slider("Années avec ce manager", 0, 17, 3)
            training      = r3c2.slider("Formations l'an dernier", 0, 6, 3)

        with st.expander("💰 Rémunération", expanded=True):
            r4c1, r4c2, r4c3, r4c4 = st.columns(4)
            monthly_income = r4c1.number_input("Revenu mensuel (USD)", 0, 20000, 5000, step=100)
            monthly_rate   = r4c2.number_input("Taux mensuel", 0, 27000, 10000, step=100)
            daily_rate     = r4c3.number_input("Taux journalier", 0, 1500, 800, step=10)
            hourly_rate    = r4c4.number_input("Taux horaire", 0, 100, 60, step=1)
            pct_hike       = r4c1.slider("Augmentation salariale (%)", 0, 25, 15)
            stock_opt      = r4c2.slider("Options sur actions (0–3)", 0, 3, 1)

        with st.expander("🌍 Environnement de travail", expanded=False):
            r5c1, r5c2, r5c3, r5c4 = st.columns(4)
            env_sat  = r5c1.slider("Satisfaction environnement (1–4)", 1, 4, 3)
            rel_sat  = r5c2.slider("Satisfaction relations (1–4)", 1, 4, 3)
            wlb      = r5c3.slider("Work-life balance (1–4)", 1, 4, 3)
            distance = r5c4.slider("Distance domicile-travail (km)", 1, 29, 5)
            travel   = st.selectbox("Déplacements professionnels", CATEGORICAL_OPTIONS["BusinessTravel"])

        payload = {
            "Age": age, "DailyRate": daily_rate, "DistanceFromHome": distance,
            "Education": education, "EnvironmentSatisfaction": env_sat,
            "HourlyRate": hourly_rate, "JobInvolvement": job_inv,
            "JobLevel": job_level, "JobSatisfaction": job_sat,
            "MonthlyIncome": monthly_income, "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_comp, "PercentSalaryHike": pct_hike,
            "PerformanceRating": perf_rat, "RelationshipSatisfaction": rel_sat,
            "StockOptionLevel": stock_opt, "TotalWorkingYears": total_years,
            "TrainingTimesLastYear": training, "WorkLifeBalance": wlb,
            "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo, "YearsWithCurrManager": years_manager,
            "BusinessTravel": travel, "Department": department,
            "EducationField": edu_field, "Gender": gender,
            "JobRole": job_role, "MaritalStatus": marital, "OverTime": overtime,
        }

        if st.button("🔮 Prédire", type="primary"):
            with st.spinner("Appel à l'API en cours…"):
                result = api_predict(api_url, payload)

            if result is not None:
                prob = result["probability"]
                pred = result["prediction"]
                st.divider()

                res_left, res_right = st.columns([2, 3])

                with res_left:
                    if pred == 1:
                        st.error("## ⚠️ Risque d'attrition détecté")
                    else:
                        st.success("## ✅ Employé stable")

                    delta_str = f"{prob - 0.5:+.1%} vs seuil 50 %"
                    st.metric("Probabilité de départ", f"{prob:.1%}", delta=delta_str, delta_color="inverse")
                    risk_gauge(prob)

                with res_right:
                    st.subheader("Analyse des facteurs de risque")
                    warnings_list: list[tuple[str, str]] = []

                    if overtime == "Yes":
                        warnings_list.append(("🔴", "Heures supplémentaires régulières", "fort signal de départ"))
                    if job_sat <= 2:
                        warnings_list.append(("🔴", "Satisfaction au travail faible", f"score {job_sat}/4"))
                    if monthly_income < 3000:
                        warnings_list.append(("🔴", "Revenu faible", f"${monthly_income:,}/mois"))
                    if distance > 20:
                        warnings_list.append(("🟡", "Distance domicile-travail élevée", f"{distance} km"))
                    if marital == "Single":
                        warnings_list.append(("🟡", "Célibataire", "plus grande mobilité professionnelle"))
                    if years_company < 2:
                        warnings_list.append(("🟡", "Faible ancienneté", f"{years_company} an(s) — période critique"))
                    if travel == "Travel_Frequently":
                        warnings_list.append(("🟡", "Déplacements fréquents", "facteur d'épuisement"))
                    if stock_opt == 0:
                        warnings_list.append(("🟡", "Pas d'options sur actions", "levier de fidélisation absent"))
                    if env_sat <= 2:
                        warnings_list.append(("🟡", "Environnement peu satisfaisant", f"score {env_sat}/4"))
                    if wlb <= 2:
                        warnings_list.append(("🟡", "Work-life balance dégradé", f"score {wlb}/4"))

                    if warnings_list:
                        for icon, factor, detail in warnings_list:
                            st.markdown(f"{icon} **{factor}** — {detail}")
                    else:
                        st.success("Aucun facteur de risque majeur identifié.")

                    st.divider()
                    st.caption(f"Seuil de décision : 0.50 · Probabilité brute : {prob:.4f}")

    # ── Batch CSV ─────────────────────────────────────────────────────────────
    else:
        st.subheader("Prédiction en masse")
        st.info("Uploadez un CSV avec les 30 features IBM HR (colonnes identiques au dataset).")

        template_cols = NUMERIC_FEATURES + list(CATEGORICAL_OPTIONS.keys())
        st.download_button(
            "⬇️ Télécharger le template CSV",
            data=pd.DataFrame(columns=template_cols).to_csv(index=False),
            file_name="template_employees.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("📁 Charger le CSV", type=["csv"])
        if uploaded:
            df_batch = pd.read_csv(uploaded)
            st.info(f"{len(df_batch)} employé(s) chargé(s)")
            st.dataframe(df_batch.head(), use_container_width=True)

            if st.button("🔮 Prédire pour tous", type="primary"):
                results = []
                bar = st.progress(0, text="Prédiction en cours…")
                for i, row in df_batch.iterrows():
                    r = api_predict(api_url, row.to_dict())
                    if r:
                        results.append({
                            "Employé #": i,
                            "Résultat": "⚠️ Part" if r["prediction"] else "✅ Reste",
                            "Probabilité (%)": round(r["probability"] * 100, 1),
                            "Niveau de risque": (
                                "Élevé" if r["probability"] > 0.66 else
                                "Moyen" if r["probability"] > 0.33 else "Faible"
                            ),
                        })
                    bar.progress((i + 1) / len(df_batch))
                bar.empty()

                if results:
                    df_res = pd.DataFrame(results)
                    n_total = len(df_res)
                    n_depart = (df_res["Résultat"].str.contains("Part")).sum()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total", n_total)
                    m2.metric("Départs prédits", n_depart)
                    m3.metric("Taux prédit", f"{n_depart / n_total:.1%}")

                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Télécharger les résultats",
                        data=df_res.to_csv(index=False),
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

                    # Pie chart des niveaux
                    risk_counts = df_res["Niveau de risque"].value_counts()
                    fig, ax = plt.subplots(figsize=(4, 4))
                    colors_map = {"Élevé": "#e74c3c", "Moyen": "#f39c12", "Faible": "#2ecc71"}
                    ax.pie(
                        risk_counts.values,
                        labels=risk_counts.index,
                        colors=[colors_map.get(l, "#95a5a6") for l in risk_counts.index],
                        autopct="%1.0f%%",
                        wedgeprops=dict(edgecolor="white", linewidth=2),
                    )
                    ax.set_title("Répartition par niveau de risque", fontweight="bold")
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Modèles MLflow
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("📊 Comparaison des modèles — MLflow")

    with st.spinner("Chargement des runs MLflow…"):
        df_runs = mlflow_runs(mlf_uri, mlf_exp)

    if df_runs.empty:
        st.warning("""
**Aucun run trouvé.** Raisons possibles :
- Aucun entraînement lancé → `make train` ou `make compose-up`
- URI MLflow incorrecte (vérifier la barre latérale)
- Le fichier `mlruns.db` est inaccessible depuis ce conteneur
        """)
        st.stop()

    metric_cols = [c for c in df_runs.columns if c.startswith("metrics.")]
    run_name_col = "tags.mlflow.runName" if "tags.mlflow.runName" in df_runs.columns else "run_id"

    # Table
    display_df = df_runs[[run_name_col, "run_id", "start_time", "status"] + metric_cols].copy()
    display_df["start_time"] = pd.to_datetime(display_df["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
    display_df["run_id"]     = display_df["run_id"].str[:8]
    display_df.columns = [
        c.replace("metrics.", "").replace("tags.mlflow.", "")
        for c in display_df.columns
    ]
    st.subheader(f"📋 {len(df_runs)} runs dans l'expérience **{mlf_exp}**")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # Metric chart
    available = {
        c.replace("metrics.", ""): c
        for c in metric_cols
        if any(m in c for m in ["roc_auc", "f1", "precision", "recall"])
    }

    if available:
        st.subheader("📈 Comparaison visuelle des métriques")
        selected = st.multiselect(
            "Métriques à afficher",
            list(available.keys()),
            default=list(available.keys())[:4],
        )

        if selected:
            run_labels = df_runs[run_name_col].fillna(df_runs["run_id"].str[:8]).tolist()
            COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
            n_met  = len(selected)
            n_cols = min(2, n_met)
            n_rows = (n_met + n_cols - 1) // n_cols
            row_h  = max(3.0, len(run_labels) * 0.45 + 1.2)

            fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * row_h))

            # Normalise en liste plate quelle que soit la forme retournée par subplots
            if n_met == 1:
                axes_flat = [axes_grid]
            elif n_rows == 1:
                axes_flat = list(axes_grid)
            else:
                axes_flat = [ax for row in axes_grid for ax in row]

            y_pos = np.arange(len(run_labels))

            for i, met in enumerate(selected):
                ax   = axes_flat[i]
                vals = df_runs[available[met]].fillna(0).values
                bars = ax.barh(y_pos, vals, color=COLORS[i % len(COLORS)], alpha=0.85, edgecolor="white")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(run_labels, fontsize=9)
                ax.set_xlim(0, 1.12)
                ax.set_xlabel("Score")
                ax.set_title(met, fontweight="bold", fontsize=11)
                ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.grid(axis="x", alpha=0.3)
                for bar, val in zip(bars, vals):
                    if val > 0:
                        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

            for j in range(n_met, len(axes_flat)):
                axes_flat[j].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.divider()

    # Best run
    st.subheader("🏆 Meilleur run (ROC-AUC)")
    auc_col = next(
        (c for c in ["metrics.test_roc_auc", "metrics.roc_auc"] if c in df_runs.columns),
        None,
    )
    if auc_col and not df_runs[auc_col].isna().all():
        best = df_runs.loc[df_runs[auc_col].idxmax()]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Run", str(best.get(run_name_col, "?"))[:25])
        b2.metric("ROC-AUC", f"{best[auc_col]:.4f}")
        f1_col = next(
            (c for c in ["metrics.test_f1", "metrics.f1"] if c in df_runs.columns), None
        )
        if f1_col:
            b3.metric("F1", f"{best[f1_col]:.4f}" if pd.notna(best[f1_col]) else "–")
        b4.metric("Run ID", best["run_id"][:8])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Artefacts
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("🖼️ Artefacts MLflow par run")

    with st.spinner("Chargement des runs…"):
        df_runs_art = mlflow_runs(mlf_uri, mlf_exp)

    if df_runs_art.empty:
        st.warning("Aucun run trouvé. Lancez d'abord un entraînement.")
    else:
        rn_col = "tags.mlflow.runName" if "tags.mlflow.runName" in df_runs_art.columns else "run_id"
        run_options = {
            f"{str(row.get(rn_col, 'run'))[:30]}  ({row['run_id'][:8]})": row["run_id"]
            for _, row in df_runs_art.iterrows()
        }
        selected_label  = st.selectbox("Sélectionner un run", list(run_options.keys()))
        selected_run_id = run_options[selected_label]

        st.divider()
        col_cm, col_rep = st.columns(2)

        with col_cm:
            st.subheader("Matrice de confusion")
            cm_bytes = mlflow_artifact(mlf_uri, selected_run_id, "confusion_matrix.png")
            if cm_bytes:
                st.image(cm_bytes, use_container_width=True)
                st.download_button(
                    "⬇️ Télécharger",
                    data=cm_bytes,
                    file_name=f"confusion_matrix_{selected_run_id[:8]}.png",
                    mime="image/png",
                )
            else:
                st.info("Non disponible pour ce run.")

        with col_rep:
            st.subheader("Classification Report")
            rep_bytes = mlflow_artifact(mlf_uri, selected_run_id, "classification_report.txt")
            if rep_bytes:
                st.text(rep_bytes.decode("utf-8"))
                st.download_button(
                    "⬇️ Télécharger",
                    data=rep_bytes,
                    file_name=f"report_{selected_run_id[:8]}.txt",
                    mime="text/plain",
                )
            else:
                st.info("Non disponible pour ce run.")

        st.divider()
        st.subheader("Détail du run sélectionné")
        run_row = df_runs_art[df_runs_art["run_id"] == selected_run_id].iloc[0]

        m_cols = [c for c in df_runs_art.columns if c.startswith("metrics.")]
        p_cols = [c for c in df_runs_art.columns if c.startswith("params.")]

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Métriques**")
            for c in m_cols:
                if pd.notna(run_row[c]):
                    st.metric(c.replace("metrics.", ""), f"{run_row[c]:.4f}")
        with d2:
            st.markdown("**Paramètres**")
            params_data = {c.replace("params.", ""): run_row[c] for c in p_cols if pd.notna(run_row[c])}
            if params_data:
                st.json(params_data)
            else:
                st.caption("Aucun paramètre enregistré pour ce run.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Entraînement
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.title("⚗️ Entraînement")

    # ── Modèle actif ──────────────────────────────────────────────────────────
    t5_info = api_model_info(api_url)
    if t5_info.get("loaded"):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Modèle actif", t5_info.get("model_type", "–"))
        mets5 = t5_info.get("metrics", {})
        mc2.metric("ROC-AUC", f"{mets5['roc_auc']:.3f}" if mets5.get("roc_auc") else "–")
        mc3.metric("F1",      f"{mets5['f1']:.3f}"      if mets5.get("f1")      else "–")
        mc4.metric("Recall",  f"{mets5['recall']:.3f}"  if mets5.get("recall")  else "–")
    else:
        st.warning("API non disponible.")

    st.divider()

    in_docker = Path("/.dockerenv").exists()

    if in_docker:
        st.info("""
**Mode Docker** — l'entraînement s'exécute dans le conteneur `trainer`.
```bash
docker compose run --rm trainer                      # baseline
TRAIN_MODE=full   docker compose run --rm trainer   # GridSearchCV
TRAIN_MODE=optuna docker compose run --rm trainer   # Optuna
```
        """)

    else:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        MODELS_5 = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "MLP"]

        def _run_training(cmd: list[str]) -> None:
            st.code(" ".join(cmd), language="bash")
            with st.status("⏳ Entraînement en cours…", expanded=True) as status:
                log_area = st.empty()
                lines_buf: list[str] = []
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        cwd=str(PROJECT_ROOT),
                    )
                    for line in proc.stdout:
                        lines_buf.append(line.rstrip())
                        log_area.code("\n".join(lines_buf[-50:]))
                    proc.wait()
                    if proc.returncode == 0:
                        status.update(label="✅ Terminé avec succès !", state="complete")
                        st.cache_data.clear()
                    else:
                        status.update(label=f"❌ Erreur (code {proc.returncode})", state="error")
                except FileNotFoundError:
                    status.update(label="❌ Python introuvable", state="error")
                    st.error("Lance `make streamlit` depuis l'environnement uv.")

        tab_custom, tab_grid, tab_optuna_t = st.tabs([
            "🎯 Personnalisé", "🔬 GridSearchCV (5 modèles)", "⚡ Optuna TPE",
        ])

        # ── 🎯 Personnalisé ───────────────────────────────────────────────────
        with tab_custom:
            st.subheader("Choisis ton modèle et tes hyperparamètres")

            model_choice = st.selectbox("Modèle", MODELS_5, key="custom_model")
            st.divider()

            cmd_custom = [sys.executable, "-m", "mlproject.train_custom", "--model", model_choice]

            if model_choice == "LogisticRegression":
                c1, c2 = st.columns(2)
                lr_c    = c1.number_input("C (régularisation L2)", 0.001, 100.0, 1.0, step=0.1, format="%.3f")
                lr_iter = c2.slider("max_iter", 100, 5000, 1000, step=100)
                cmd_custom += ["--c", str(lr_c), "--max-iter", str(lr_iter)]
                st.caption("C petit → plus régularisé · C grand → moins contraint. Augmenter max_iter si avertissement de convergence.")

            elif model_choice == "RandomForest":
                c1, c2, c3 = st.columns(3)
                rf_n    = c1.slider("n_estimators", 50, 500, 100, step=50)
                rf_d    = c2.slider("max_depth  (0 = illimité)", 0, 30, 10)
                rf_leaf = c3.slider("min_samples_leaf", 1, 20, 1)
                cmd_custom += ["--n-estimators", str(rf_n), "--max-depth", str(rf_d), "--min-samples-leaf", str(rf_leaf)]
                st.caption("Augmenter n_estimators améliore la robustesse · max_depth faible réduit l'overfitting.")

            elif model_choice == "XGBoost":
                c1, c2, c3 = st.columns(3)
                xgb_n  = c1.slider("n_estimators", 50, 500, 100, step=50)
                xgb_d  = c2.slider("max_depth", 2, 10, 6)
                xgb_lr = c3.slider("learning_rate", 0.01, 0.30, 0.10, step=0.01, format="%.2f")
                cmd_custom += ["--n-estimators", str(xgb_n), "--max-depth", str(xgb_d), "--learning-rate", str(xgb_lr)]
                st.caption("XGBoost utilise scale_pos_weight automatiquement pour gérer le déséquilibre 84/16.")

            elif model_choice == "LightGBM":
                c1, c2, c3 = st.columns(3)
                lgb_n  = c1.slider("n_estimators", 50, 500, 100, step=50)
                lgb_lv = c2.slider("num_leaves", 15, 127, 31)
                lgb_lr = c3.slider("learning_rate", 0.01, 0.30, 0.10, step=0.01, format="%.2f")
                cmd_custom += ["--n-estimators", str(lgb_n), "--num-leaves", str(lgb_lv), "--learning-rate", str(lgb_lr)]
                st.caption("num_leaves contrôle la complexité des arbres (↑ = plus expressif mais risque d'overfitting).")

            elif model_choice == "MLP":
                arch_options = {
                    "64 → 32":          "64,32",
                    "128 → 64":         "128,64",
                    "128 → 64 → 32":    "128,64,32",
                    "256 → 128 → 64":   "256,128,64",
                }
                c1, c2, c3, c4 = st.columns(4)
                mlp_arch   = c1.selectbox("Architecture", list(arch_options.keys()))
                mlp_lr     = c2.number_input("learning_rate", 1e-4, 1e-2, 1e-3, step=1e-4, format="%.4f")
                mlp_drop   = c3.slider("dropout", 0.0, 0.6, 0.3, step=0.05, format="%.2f")
                mlp_epochs = c4.slider("epochs", 50, 500, 150, step=50)
                cmd_custom += [
                    "--hidden-sizes", arch_options[mlp_arch],
                    "--lr", str(mlp_lr),
                    "--dropout", str(mlp_drop),
                    "--epochs", str(mlp_epochs),
                ]
                st.caption("MLP PyTorch · BCEWithLogitsLoss + pos_weight · Adam optimizer · Dropout régularisation.")

            st.divider()
            if st.button("▶️ Lancer l'entraînement", type="primary", key="btn_custom"):
                _run_training(cmd_custom)

        # ── 🔬 GridSearchCV ───────────────────────────────────────────────────
        with tab_grid:
            st.subheader("Comparaison exhaustive — 5 modèles en parallèle")
            st.markdown("""
Explore des grilles d'hyperparamètres prédéfinies pour les **5 modèles** et sélectionne automatiquement le champion (meilleur ROC-AUC test).

| Modèle | Grille explorée |
|---|---|
| LogisticRegression | C ∈ {0.01, 0.1, 1, 10} |
| RandomForest | n_est × max_depth × min_samples_leaf (12 combinaisons) |
| XGBoost | n_est × max_depth × learning_rate (8 combinaisons) |
| LightGBM | n_est × num_leaves × learning_rate (8 combinaisons) |
| MLP PyTorch | hidden_sizes × lr (4 combinaisons) |
            """)

            c1g, c2g = st.columns(2)
            cv_folds = c1g.slider("Folds de cross-validation", 2, 10, 2,
                                  help="CV=2 → ~10 min · CV=5 → ~25 min · CV=10 → ~50 min")
            c2g.metric("Durée estimée", f"~{cv_folds * 5} min")
            st.caption("Tous les modèles sont entraînés en parallèle (n_jobs=-1).")

            if st.button("▶️ Lancer GridSearchCV", type="primary", key="btn_grid"):
                _run_training([sys.executable, "-m", "mlproject.train_models", "--cv", str(cv_folds)])

        # ── ⚡ Optuna TPE ─────────────────────────────────────────────────────
        with tab_optuna_t:
            st.subheader("Optimisation bayésienne — un modèle, espaces continus")
            st.markdown("""
Optuna explore l'espace de recherche avec l'algorithme **TPE** (Tree-structured Parzen Estimator) :
chaque trial exploite les résultats précédents pour cibler les zones prometteuses.

**Avantage vs GridSearchCV** : espaces continus (ex. `learning_rate ∈ [0.001, 0.3]`),
convergence plus rapide, moins de trials suffit pour trouver un bon optimum.
            """)

            c1o, c2o, c3o = st.columns(3)
            opt_model  = c1o.selectbox("Modèle à optimiser", MODELS_5, key="optuna_model")
            opt_trials = c2o.slider("Nombre de trials", 5, 200, 20)
            opt_cv     = c3o.slider("Folds CV", 2, 10, 3)

            duration_map = {
                "LogisticRegression": 0.2, "RandomForest": 2.0,
                "XGBoost": 1.0, "LightGBM": 0.7, "MLP": 4.0,
            }
            est_min = round(opt_trials * opt_cv * duration_map.get(opt_model, 1.0) / 60, 1)
            st.info(f"Durée estimée : **~{est_min} min** pour {opt_model} × {opt_trials} trials × CV={opt_cv}")

            if st.button("▶️ Lancer Optuna", type="primary", key="btn_optuna"):
                _run_training([
                    sys.executable, "-m", "mlproject.train_optuna",
                    "--model",    opt_model,
                    "--n-trials", str(opt_trials),
                    "--cv",       str(opt_cv),
                ])

    st.divider()

    # ── Historique des runs (toujours visible) ────────────────────────────────
    st.subheader("📋 Historique des runs MLflow")
    with st.spinner("Chargement…"):
        df_hist = mlflow_runs(mlf_uri, mlf_exp)

    if not df_hist.empty:
        rn_hist    = "tags.mlflow.runName" if "tags.mlflow.runName" in df_hist.columns else "run_id"
        color_map  = {"custom": "🎯", "gridsearch": "🔬", "optuna": "⚡"}
        m_hist     = [c for c in df_hist.columns if c.startswith("metrics.")]

        hist_df = df_hist[[rn_hist, "run_id", "start_time", "status"] + m_hist].copy()
        if "tags.optimizer" in df_hist.columns:
            hist_df.insert(0, "type", df_hist["tags.optimizer"].map(color_map).fillna("📝"))
        hist_df["run_id"]     = hist_df["run_id"].str[:8]
        hist_df["start_time"] = pd.to_datetime(hist_df["start_time"]).dt.strftime("%Y-%m-%d %H:%M")
        hist_df.columns       = [c.replace("metrics.", "").replace("tags.mlflow.", "") for c in hist_df.columns]
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Aucun run enregistré. Lance un entraînement ci-dessus.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Exploration des données
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.title("🔍 Exploration du dataset IBM HR")

    DATA_CANDIDATES = [
        Path("data/ibm_hr_attrition.csv"),
        Path("/app/data/ibm_hr_attrition.csv"),
        Path(__file__).resolve().parents[1] / "data" / "ibm_hr_attrition.csv",
    ]

    df_data: pd.DataFrame | None = None
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            df_data = pd.read_csv(candidate)
            st.success(f"Dataset chargé : `{candidate}` ({len(df_data)} lignes)")
            break

    if df_data is None:
        st.info("Dataset non trouvé automatiquement — uploadez-le ici :")
        uploaded_csv = st.file_uploader("📁 ibm_hr_attrition.csv", type=["csv"])
        if uploaded_csv:
            df_data = pd.read_csv(uploaded_csv)
            st.success(f"{len(df_data)} lignes chargées")

    if df_data is not None:
        # Nettoyage
        drop_const = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
        df_data = df_data.drop(columns=[c for c in drop_const if c in df_data.columns])
        if "Attrition" in df_data.columns:
            df_data["Attrition_num"] = (df_data["Attrition"] == "Yes").astype(int)

        st.divider()

        # Vue d'ensemble
        st.subheader("Vue d'ensemble")
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Lignes", len(df_data))
        v2.metric("Colonnes", len(df_data.columns))
        v3.metric("Valeurs manquantes", int(df_data.isnull().sum().sum()))
        if "Attrition" in df_data.columns:
            v4.metric("Taux d'attrition", f"{(df_data['Attrition']=='Yes').mean():.1%}")

        with st.expander("Aperçu des données brutes"):
            st.dataframe(df_data.head(15), use_container_width=True)

        st.divider()

        # Distribution par feature
        st.subheader("Distribution par feature")
        all_features = NUMERIC_FEATURES + list(CATEGORICAL_OPTIONS.keys())
        available_features = [f for f in all_features if f in df_data.columns]
        feature = st.selectbox("Feature à analyser", available_features)

        if "Attrition" in df_data.columns and feature in df_data.columns:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            if feature in NUMERIC_FEATURES:
                for val, color, lbl in [("Yes", "#e74c3c", "Quitte"), ("No", "#2ecc71", "Reste")]:
                    subset = df_data[df_data["Attrition"] == val][feature].dropna()
                    axes[0].hist(subset, bins=20, alpha=0.65, color=color, label=lbl, edgecolor="white")
                axes[0].set_title(f"Distribution de {feature} par attrition")
                axes[0].set_xlabel(feature)
                axes[0].set_ylabel("Fréquence")
                axes[0].legend()

                d_yes = df_data[df_data["Attrition"] == "Yes"][feature].dropna()
                d_no  = df_data[df_data["Attrition"] == "No"][feature].dropna()
                axes[1].boxplot(
                    [d_no, d_yes],
                    patch_artist=True,
                    boxprops=dict(facecolor="#aed6f1"),
                    medianprops=dict(color="navy", linewidth=2),
                )
                axes[1].set_xticklabels(["Reste (No)", "Quitte (Yes)"])
                axes[1].set_title(f"Boxplot {feature} par attrition")
                axes[1].set_ylabel(feature)

            else:
                ct = df_data.groupby([feature, "Attrition"]).size().unstack(fill_value=0)
                ct_pct = ct.div(ct.sum(axis=1), axis=0)

                ct.plot(kind="bar", ax=axes[0], color=["#2ecc71", "#e74c3c"], alpha=0.85, edgecolor="white")
                axes[0].set_title(f"Effectifs par {feature}")
                axes[0].set_xlabel("")
                axes[0].set_ylabel("Nombre")
                plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")

                ct_pct.plot(kind="bar", ax=axes[1], color=["#2ecc71", "#e74c3c"], alpha=0.85, edgecolor="white")
                axes[1].set_title(f"Taux d'attrition par {feature}")
                axes[1].set_xlabel("")
                axes[1].set_ylabel("Proportion")
                axes[1].set_ylim(0, 1)
                plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # Corrélation avec attrition
        st.subheader("Corrélation des features numériques avec l'attrition")
        if "Attrition_num" in df_data.columns:
            num_avail = [f for f in NUMERIC_FEATURES if f in df_data.columns]
            corr = (
                df_data[num_avail + ["Attrition_num"]]
                .dropna()
                .corr()["Attrition_num"]
                .drop("Attrition_num")
                .sort_values()
            )
            fig, ax = plt.subplots(figsize=(9, 6))
            bar_colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr.values]
            ax.barh(corr.index, corr.values, color=bar_colors, alpha=0.85, edgecolor="white")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Corrélation de Pearson")
            ax.set_title("Corrélation avec l'attrition\nRouge = corrélé positivement, Vert = corrélé négativement")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()

        # Stats descriptives
        st.subheader("Statistiques descriptives")
        with st.expander("Features numériques"):
            num_avail_stats = [f for f in NUMERIC_FEATURES if f in df_data.columns]
            desc = df_data[num_avail_stats].describe().T.round(2)
            desc["missing"] = df_data[num_avail_stats].isnull().sum()
            st.dataframe(desc, use_container_width=True)

        with st.expander("Features catégorielles & taux d'attrition"):
            cat_rows = []
            for col in CATEGORICAL_OPTIONS:
                if col not in df_data.columns:
                    continue
                for val, cnt in df_data[col].value_counts().items():
                    rate = 0.0
                    if "Attrition" in df_data.columns:
                        rate = (df_data[df_data[col] == val]["Attrition"] == "Yes").mean()
                    cat_rows.append({
                        "Feature": col, "Valeur": val,
                        "Effectif": cnt,
                        "% total": f"{cnt/len(df_data):.1%}",
                        "Taux attrition": f"{rate:.1%}",
                    })
            if cat_rows:
                st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Orchestration Airflow
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    t7_left, t7_right = st.columns([5, 1])
    t7_left.title("🔄 Orchestration Airflow")
    if t7_right.button("↺ Actualiser", key="refresh_orches", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    af_url = st.text_input(
        "🔗 URL Airflow (interne Docker)",
        value=st.session_state.airflow_url,
        key="airflow_url_input",
        help="URL utilisée par Streamlit pour appeler l'API REST Airflow (communication inter-conteneurs).",
    )
    st.session_state.airflow_url = af_url

    with st.expander("🔧 Debug — configuration active"):
        st.code(
            f"Airflow API  : {af_url}\n"
            f"MLflow URI   : {mlf_uri}\n"
            f"Expérience   : {mlf_exp}",
            language="text",
        )
        st.caption("Si le MLflow URI pointe vers un fichier local introuvable, changez-le dans la barre latérale.")

    af_ok = airflow_health(af_url)

    if af_ok:
        st.success(f"Airflow ✅ connecté sur `{af_url}`")
    else:
        st.error(f"Airflow ❌ inaccessible — lancez `make airflow` ou `make airflow-stack`")
        st.code(
            "make airflow          # Airflow seul (port 8080)\n"
            "make airflow-stack    # Stack complète : API + Airflow + Streamlit",
            language="bash",
        )

    st.divider()

    # ── Statut des deux DAGs ──────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    def _dag_status_card(col, dag_id: str, label: str, schedule: str) -> None:
        with col:
            st.markdown(f"### {label}")
            st.caption(f"Schedule : `{schedule}`")
            if not af_ok:
                st.warning("Airflow non disponible")
                return
            runs = airflow_dag_runs(af_url, dag_id, limit=1)
            if not runs:
                st.info("Aucun run enregistré")
                return
            last = runs[0]
            state = last.get("state", "?")
            st.metric("Dernier run", f"{state_icon(state)} {state}")
            st.caption(f"Démarré : {fmt_date(last.get('start_date', ''))}")
            if last.get("end_date"):
                st.caption(f"Durée : {run_duration(last.get('start_date',''), last.get('end_date',''))}")

    _dag_status_card(c1, "model_retraining",  "🔁 model_retraining",  "0 */5 * * *")
    _dag_status_card(c2, "daily_predictions", "📨 daily_predictions", "30 */5 * * *")

    if not af_ok:
        st.stop()

    st.divider()

    # ── Historique des réentraînements ────────────────────────────────────────
    st.subheader("🔁 Historique des réentraînements")

    retrain_runs = airflow_dag_runs(af_url, "model_retraining", limit=20)

    if retrain_runs:
        n_ok  = sum(1 for r in retrain_runs if r.get("state") == "success")
        n_err = sum(1 for r in retrain_runs if r.get("state") == "failed")
        n_run = sum(1 for r in retrain_runs if r.get("state") == "running")

        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Runs totaux", len(retrain_runs))
        rm2.metric("✅ Succès",   n_ok)
        rm3.metric("❌ Échecs",   n_err)
        rm4.metric("⏳ En cours", n_run)

        retrain_rows = [
            {
                "État":    state_icon(r.get("state", "")),
                "Statut":  r.get("state", "?"),
                "Démarré": fmt_date(r.get("start_date", "")),
                "Durée":   run_duration(r.get("start_date", ""), r.get("end_date", "")),
                "Run ID":  r.get("dag_run_id", "?")[-22:],
            }
            for r in retrain_runs
        ]
        st.dataframe(pd.DataFrame(retrain_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Aucun run pour model_retraining — déclenchez `make airflow-trigger`.")

    # ── Évolution F1 via MLflow ───────────────────────────────────────────────
    st.subheader("📈 Évolution du F1 (MLflow)")

    with st.spinner("Chargement des métriques MLflow…"):
        df_mlf7 = mlflow_runs(mlf_uri, mlf_exp)

    if not df_mlf7.empty:
        f1_col  = next((c for c in ["metrics.f1", "metrics.test_f1"]      if c in df_mlf7.columns), None)
        auc_col = next((c for c in ["metrics.roc_auc","metrics.test_roc_auc"] if c in df_mlf7.columns), None)

        if f1_col:
            df_tl = df_mlf7[["start_time", f1_col] + ([auc_col] if auc_col else [])].dropna(subset=[f1_col]).copy()
            df_tl = df_tl.sort_values("start_time")
            df_tl["start_time"] = pd.to_datetime(df_tl["start_time"])

            if len(df_tl) >= 2:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(df_tl["start_time"], df_tl[f1_col],
                        marker="o", color="#3498db", linewidth=2, label="F1")
                if auc_col and auc_col in df_tl.columns:
                    ax.plot(df_tl["start_time"], df_tl[auc_col],
                            marker="s", color="#2ecc71", linewidth=2, linestyle="--", label="ROC-AUC")
                ax.axhline(QUALITY_THRESHOLD, color="#e74c3c", linestyle=":", linewidth=1.5,
                           label=f"Seuil qualité ({QUALITY_THRESHOLD})")
                ax.set_ylabel("Score")
                ax.set_title("Métriques des réentraînements automatiques dans le temps")
                ax.legend()
                ax.grid(alpha=0.3)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Minimum 2 runs nécessaires pour tracer la courbe.")
        else:
            st.info("Aucune métrique F1 trouvée dans MLflow.")
    else:
        st.info("Aucun run MLflow disponible.")

    st.divider()

    # ── Historique des prédictions ────────────────────────────────────────────
    st.subheader("📨 Historique des prédictions automatiques")

    pred_runs = airflow_dag_runs(af_url, "daily_predictions", limit=20)

    if pred_runs:
        pred_rows = []
        for run in pred_runs:
            state  = run.get("state", "?")
            run_id = run.get("dag_run_id", "")

            n_pred = n_attr = rate_val = None
            if state == "success" and run_id:
                n_pred   = airflow_xcom(af_url, "daily_predictions", run_id, "send_predictions", "n_predictions")
                n_attr   = airflow_xcom(af_url, "daily_predictions", run_id, "send_predictions", "n_attrition")
                rate_val = airflow_xcom(af_url, "daily_predictions", run_id, "send_predictions", "attrition_rate")

            pred_rows.append({
                "État":         state_icon(state),
                "Statut":       state,
                "Démarré":      fmt_date(run.get("start_date", "")),
                "Durée":        run_duration(run.get("start_date", ""), run.get("end_date", "")),
                "Prédictions":  n_pred  if n_pred  is not None else "–",
                "Attrition":    n_attr  if n_attr  is not None else "–",
                "Taux":         f"{float(rate_val):.1%}" if rate_val is not None else "–",
            })

        n_ok_p = sum(1 for r in pred_runs if r.get("state") == "success")
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Runs totaux",  len(pred_runs))
        pm2.metric("✅ Succès",    n_ok_p)
        pm3.metric("❌ Échecs",    len(pred_runs) - n_ok_p)

        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

        # ── Courbe du taux d'attrition ────────────────────────────────────────
        rate_points = [
            (r["Démarré"], float(r["Taux"].replace("%", "")) / 100)
            for r in pred_rows
            if r["Taux"] != "–"
        ]

        if len(rate_points) >= 2:
            st.subheader("📊 Taux d'attrition prédit dans le temps")
            dates_r, rates_r = zip(*sorted(rate_points))
            pct = [v * 100 for v in rates_r]
            x   = range(len(dates_r))

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(x, pct, marker="o", color="#e74c3c", linewidth=2)
            ax.fill_between(x, pct, alpha=0.12, color="#e74c3c")
            ax.axhline(30, color="#f39c12", linestyle="--", linewidth=1.5, label="Seuil alerte 30 %")
            ax.set_xticks(list(x))
            ax.set_xticklabels(dates_r, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("Taux d'attrition prédit (%)")
            ax.set_title("Évolution du taux d'attrition par run de prédictions")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("Aucun run pour daily_predictions.")

    st.divider()
    st.caption("Données Airflow rechargées toutes les 30 s · auth : airflow / airflow · MLflow rafraîchi toutes les 60 s")
