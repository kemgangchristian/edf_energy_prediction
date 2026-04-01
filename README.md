# Prédiction de la Consommation Électrique — EDF
> Projet MSPR EPSI — Certification Chef de Projet Expert en Intelligence Artificielle (RNCP36582)  
> Bloc 3 : Déploiement & Maintenabilité · Bloc 4 : Management Agile

---

## Présentation

Ce projet développe une solution IA complète pour **prédire la consommation électrique journalière** des abonnés EDF à l'échelle nationale, en s'appuyant sur les données open data RTE éco2mix.

Il couvre l'intégralité du cycle de vie d'une solution IA en production :
collecte des données → prétraitement → entraînement de modèles ML → déploiement Docker → monitoring → documentation.

---

## Architecture du projet

```
edf_energy_prediction/
│
├── docker-compose.yml       # MinIO (data lake S3)
├── config.py                # Configuration centralisée (MinIO, dates, features)
├── pipeline.py              # Pipeline principal : collecte → traitement → MinIO
├── minio_client.py          # Client MinIO réutilisable (upload/download CSV, JSON, pickle)
│
├── data/
│   ├── collect_data.py      # Collecte API RTE + simulation hors-ligne
│   ├── raw/                 # Données brutes (géré par MinIO en prod)
│   └── processed/           # Données traitées (géré par MinIO en prod)
│
├── models/                  # Scripts d'entraînement des modèles ML
├── api/                     # API FastAPI de prédiction
├── monitoring/              # Monitoring des performances en production
├── tests/                   # Tests unitaires et d'intégration
└── docs/                    # Documentation technique et runbook
```

---

## Stack technique

| Composant | Technologie |
|---|---|
| Data Lake | MinIO (compatible S3) |
| Conteneurisation | Docker + Docker Compose |
| Langage | Python 3.10+ |
| Modèles ML | scikit-learn (Random Forest, KNN, Decision Tree, RBF Neural Net) |
| API | FastAPI |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana (Phase 4) |

---

## Prérequis

- Docker Desktop installé et lancé
- Python 3.10 ou supérieur
- pip

```bash
pip install minio pandas numpy requests scikit-learn fastapi uvicorn
```

---

## Installation et lancement

### 1. Cloner le projet

```bash
git clone https://github.com/kemgangchristian/edf_energy_prediction.git
cd edf_energy_prediction
```

### 2. Lancer MinIO (data lake)

```bash
# Docker Desktop doit être lancé
docker compose up -d
```

MinIO démarre sur deux ports :
- `http://localhost:9000` → API S3 (utilisée par Python)
- `http://localhost:9001` → Console web (interface graphique)

**Identifiants :**
```
Login    : edf_admin
Password : edf_password_2024
```

Vérifier que MinIO est bien lancé :
```bash
docker compose ps
```

### 3. Lancer le pipeline de données

```bash
# Période par défaut : 2021-01-01 → 2023-12-31
python pipeline.py

# Période personnalisée
python pipeline.py 2022-01-01 2022-12-31

# Forcer le re-traitement même si déjà présent sur MinIO
python pipeline.py --force
```

Le pipeline exécute automatiquement :
1. Collecte des données via l'API RTE éco2mix (bascule sur données simulées si hors connexion)
2. Upload des données brutes → bucket `edf-raw`
3. Prétraitement (features temporelles, encodage cyclique, agrégation journalière)
4. Upload des données traitées → bucket `edf-processed`
5. Génération du rapport d'exploration → bucket `edf-reports`

---

## Buckets MinIO

| Bucket | Contenu |
|---|---|
| `edf-raw` | Données brutes RTE demi-horaires (CSV) |
| `edf-processed` | Données journalières prétraitées — prêtes pour l'entraînement (CSV) |
| `edf-models` | Modèles ML entraînés sérialisés (pickle) |
| `edf-reports` | Rapports d'exploration, métriques, logs de monitoring (JSON) |

---

## Features utilisées par les modèles

| Feature | Description |
|---|---|
| `temperature_moy` | Température moyenne journalière (°C) |
| `mois_sin` / `mois_cos` | Encodage cyclique du mois |
| `jour_sin` / `jour_cos` | Encodage cyclique du jour de la semaine |
| `is_weekend` | 1 si week-end, 0 sinon |
| `saison` | 0=Hiver, 1=Printemps, 2=Été, 3=Automne |
| `trimestre` | Trimestre de l'année (1-4) |
| `jour_annee` | Jour de l'année (1-365) |

**Variable cible :** `consommation_moy` — consommation moyenne journalière en MW

---

## Phases du projet

### Phase 1 — Collecte & Exploration ✅
- Collecte API RTE éco2mix (open data)
- Prétraitement et feature engineering
- Sauvegarde sur MinIO

### Phase 2 — Modèles ML (à venir)
- Random Forest
- K-Nearest Neighbors (KNN)
- Arbre de décision
- Réseau de neurones RBF
- Métriques : R², RMSE, MAPE, temps d'entraînement
- Sauvegarde des modèles sur MinIO (`edf-models`)

### Phase 3 — API & Déploiement Docker (à venir)
- API FastAPI de prédiction
- Dockerfile + pipeline CI/CD GitHub Actions
- Simulation de montée en charge

### Phase 4 — Monitoring & Documentation (à venir)
- Détection de data drift / model drift
- Runbook d'exploitation
- Plan d'accompagnement au changement

---

## Commandes utiles

```bash
# Démarrer MinIO
docker compose up -d

# Arrêter MinIO
docker compose down

# Voir les logs MinIO
docker compose logs -f minio

# Re-lancer le pipeline sur une nouvelle période
python pipeline.py 2024-01-01 2024-06-30

# Vérifier les fichiers présents sur MinIO (console web)
# http://localhost:9001
```

---

## Configuration

Tous les paramètres sont centralisés dans `config.py` :

```python
# Connexion MinIO
MINIO_ENDPOINT   = "localhost:9000"
MINIO_ACCESS_KEY = "edf_admin"
MINIO_SECRET_KEY = "edf_password_2024"

# Période de données
DATE_START = "2021-01-01"
DATE_END   = "2023-12-31"
```

---

## Source des données

Les données de consommation électrique sont issues du site open data RTE éco2mix :  
[https://www.rte-france.com/eco2mix/la-consommation-delectricite-en-france](https://www.rte-france.com/eco2mix/la-consommation-delectricite-en-france)

---

## Auteurs

Projet réalisé dans le cadre de la MSPR EPSI 2025-2026  
Certification RNCP36582 — Chef de Projet Expert en Intelligence Artificielle