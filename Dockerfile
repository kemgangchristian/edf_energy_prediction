# Dockerfile — API EDF Prédiction Consommation Électrique
# Image de base légère Python 3.11 slim
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="Projet MSPR EPSI"
LABEL description="API FastAPI de prédiction de consommation électrique EDF"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_TAG="2021-01-01_2023-12-31" \
    MINIO_ENDPOINT="minio:9000" \
    MINIO_ACCESS_KEY="edf_admin" \
    MINIO_SECRET_KEY="edf_password_2024"

# Répertoire de travail
WORKDIR /app

# Copie des dépendances en premier (cache Docker optimisé)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY config.py        ./config.py
COPY minio_client.py  ./minio_client.py
COPY main.py          ./main.py

# Port exposé
EXPOSE 8000

# Healthcheck Docker intégré
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Lancement de l'API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]