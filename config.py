# config.py — Configuration centralisée du projet EDF
# Ce fichier est importé par tous les scripts du projet.

# ─────────────────────────────────────────────
# MINIO
# ─────────────────────────────────────────────
MINIO_ENDPOINT   = "localhost:9000"
MINIO_ACCESS_KEY = "edf_admin"
MINIO_SECRET_KEY = "edf_password_2024"
MINIO_SECURE     = False   # True si HTTPS en production

# Buckets
BUCKET_RAW       = "edf-raw"        # données brutes RTE
BUCKET_PROCESSED = "edf-processed"  # données prétraitées journalières
BUCKET_MODELS    = "edf-models"     # modèles ML entraînés (.pkl)
BUCKET_REPORTS   = "edf-reports"    # rapports, métriques, logs

# ─────────────────────────────────────────────
# DONNÉES RTE
# ─────────────────────────────────────────────
RTE_API_URL  = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
RTE_DATASET  = "eco2mix-national-cons-def"
DATE_START   = "2021-01-01"
DATE_END     = "2023-12-31"

# ─────────────────────────────────────────────
# MODÈLES ML
# ─────────────────────────────────────────────
TARGET_COL   = "consommation_moy"
FEATURE_COLS = [
    "temperature_moy",
    "mois_sin", "mois_cos",
    "jour_sin", "jour_cos",
    "is_weekend",
    "saison",
    "trimestre",
    "jour_annee",
]
TEST_SIZE    = 0.2
RANDOM_STATE = 42