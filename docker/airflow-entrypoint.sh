#!/usr/bin/env bash
# Entrypoint Airflow — s'exécute en root pour corriger les permissions des volumes,
# puis bascule sur l'utilisateur airflow (UID 50000) pour démarrer les services.
set -e

echo "[Airflow] Correction des permissions des volumes..."
mkdir -p /opt/airflow/logs /opt/airflow/project/data \
         /opt/airflow/project/models /opt/airflow/project/mlruns
chown -R 50000:0 /opt/airflow/logs /opt/airflow/project/mlruns 2>/dev/null || true
chmod -R 775     /opt/airflow/logs /opt/airflow/project/mlruns 2>/dev/null || true

echo "[Airflow] Démarrage en tant qu'utilisateur airflow..."
exec su -s /bin/bash airflow -c '
    set -e

    echo "[Airflow] Initialisation de la base de données..."
    airflow db migrate

    echo "[Airflow] Création de l'\''utilisateur admin (airflow / airflow)..."
    airflow users create \
        --username airflow \
        --password airflow \
        --firstname Airflow \
        --lastname Admin \
        --role Admin \
        --email airflow@ibm-hr.local 2>/dev/null \
        && echo "[Airflow] Utilisateur créé." \
        || echo "[Airflow] Utilisateur déjà existant — skip."

    echo "[Airflow] Démarrage (scheduler + webserver + triggerer)..."
    exec airflow standalone
'
