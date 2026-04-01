"""
pipeline.py — Pipeline principal EDF
Collecte données RTE → Prétraitement → Sauvegarde MinIO

Usage :
    python pipeline.py                    # période par défaut (config.py)
    python pipeline.py 2022-01-01 2022-12-31
    python pipeline.py --force            # re-traite même si déjà présent sur MinIO
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATE_START, DATE_END,
    BUCKET_RAW, BUCKET_PROCESSED, BUCKET_REPORTS,
)
from minio_client import (
    get_client, init_buckets,
    upload_dataframe, upload_json,
    download_dataframe, object_exists,
)


# ─────────────────────────────────────────────
# COLLECTE
# ─────────────────────────────────────────────

def fetch_rte_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Tente de collecter les données RTE via l'API OpenData.
    Bascule automatiquement sur données simulées si l'API est inaccessible.
    """
    try:
        import requests
        url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/records"
        params = {
            "where": f"date_heure >= '{start_date}T00:00:00Z' AND date_heure <= '{end_date}T23:59:59Z'",
            "limit": 10000,
            "order_by": "date_heure ASC",
            "select": "date_heure,consommation,prevision_j1,prevision_j,taux_co2",
            "timezone": "Europe/Paris",
        }
        print(f"[COLLECTE] Appel API RTE éco2mix ({start_date} → {end_date})...")
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            records = response.json().get("results", [])
            if records:
                df = pd.DataFrame(records)
                print(f"[COLLECTE] ✓ {len(df)} lignes reçues de l'API RTE.")
                return df

        print("[COLLECTE] API RTE inaccessible — bascule sur données simulées.")

    except Exception as e:
        print(f"[COLLECTE] Erreur API : {e} — bascule sur données simulées.")

    return _generate_simulated(start_date, end_date)


def _generate_simulated(start_date: str, end_date: str) -> pd.DataFrame:
    """Génère des données de consommation simulées réalistes (modèle physique simplifié)."""
    print("[COLLECTE] Génération données simulées (modèle saisonnier + température)...")

    dates = pd.date_range(start_date, end_date, freq="30min")
    np.random.seed(42)

    month_factor = {
        1: 1.35, 2: 1.30, 3: 1.10, 4: 0.95, 5: 0.90, 6: 0.88,
        7: 0.92, 8: 0.88, 9: 0.93, 10: 1.05, 11: 1.20, 12: 1.32,
    }

    records = []
    for dt in dates:
        base = 45000 * month_factor.get(dt.month, 1.0)
        h = dt.hour
        if 8 <= h <= 12 or 17 <= h <= 20:
            base *= 1.18
        elif 0 <= h <= 5:
            base *= 0.72
        if dt.weekday() >= 5:
            base *= 0.88

        day_of_year = dt.timetuple().tm_yday
        temp = 15 - 10 * np.cos(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3)
        if temp < 10:
            base *= 1 + (10 - temp) * 0.012
        elif temp > 25:
            base *= 1 + (temp - 25) * 0.008

        conso = base * (1 + np.random.normal(0, 0.03))
        records.append({
            "date_heure": dt.strftime("%Y-%m-%dT%H:%M:%S+02:00"),
            "consommation": round(conso),
            "prevision_j1": round(conso * (1 + np.random.normal(0, 0.015))),
            "prevision_j": round(conso * (1 + np.random.normal(0, 0.020))),
            "taux_co2": round(np.random.uniform(40, 120), 1),
            "temperature_simulee": round(temp, 1),
        })

    df = pd.DataFrame(records)
    print(f"[COLLECTE] ✓ {len(df)} lignes simulées.")
    return df


# ─────────────────────────────────────────────
# PRÉTRAITEMENT
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraitement complet du DataFrame brut RTE :
    1. Conversion types + nettoyage
    2. Extraction features temporelles
    3. Encodage cyclique (mois, jour)
    4. Agrégation journalière
    """
    print("\n[PRÉTRAITEMENT] Démarrage...")
    df = df.copy()

    # Conversion date
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True, errors="coerce")
    df = df.dropna(subset=["date_heure"])
    df["date_heure"] = df["date_heure"].dt.tz_convert("Europe/Paris")

    # Colonnes numériques
    for col in ["consommation", "prevision_j1", "prevision_j", "taux_co2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["consommation"])
    df = df[df["consommation"] > 0]

    # Features temporelles
    df["date"]        = df["date_heure"].dt.date
    df["mois"]        = df["date_heure"].dt.month
    df["jour_semaine"]= df["date_heure"].dt.dayofweek
    df["is_weekend"]  = (df["jour_semaine"] >= 5).astype(int)
    df["trimestre"]   = df["date_heure"].dt.quarter
    df["jour_annee"]  = df["date_heure"].dt.dayofyear

    # Température simulée si absente
    if "temperature_simulee" not in df.columns:
        df["temperature_simulee"] = (
            15 - 10 * np.cos(2 * np.pi * df["jour_annee"] / 365)
            + np.random.normal(0, 2, len(df))
        ).round(1)

    # Agrégation journalière
    daily = df.groupby("date").agg(
        consommation_moy  = ("consommation", "mean"),
        consommation_max  = ("consommation", "max"),
        consommation_min  = ("consommation", "min"),
        prevision_j1_moy  = ("prevision_j1", "mean"),
        temperature_moy   = ("temperature_simulee", "mean"),
        mois              = ("mois", "first"),
        jour_semaine      = ("jour_semaine", "first"),
        is_weekend        = ("is_weekend", "first"),
        trimestre         = ("trimestre", "first"),
        jour_annee        = ("jour_annee", "first"),
    ).reset_index()

    # Encodage cyclique
    daily["mois_sin"] = np.sin(2 * np.pi * daily["mois"] / 12)
    daily["mois_cos"] = np.cos(2 * np.pi * daily["mois"] / 12)
    daily["jour_sin"] = np.sin(2 * np.pi * daily["jour_semaine"] / 7)
    daily["jour_cos"] = np.cos(2 * np.pi * daily["jour_semaine"] / 7)

    # Saison
    saison_map = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
    daily["saison"] = daily["mois"].map(saison_map)

    daily = daily.sort_values("date").reset_index(drop=True)
    print(f"[PRÉTRAITEMENT] ✓ {len(daily)} jours | {len(daily.columns)} colonnes")
    return daily


# ─────────────────────────────────────────────
# RAPPORT D'EXPLORATION
# ─────────────────────────────────────────────

def build_report(df: pd.DataFrame, start_date: str, end_date: str) -> dict:
    """Construit un rapport JSON d'exploration des données."""
    conso = df["consommation_moy"]
    saison_labels = {0: "hiver", 1: "printemps", 2: "ete", 3: "automne"}

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "periode_start": start_date,
            "periode_end": end_date,
            "nb_jours": int(len(df)),
            "colonnes": list(df.columns),
        },
        "consommation_mw": {
            "moyenne": round(float(conso.mean()), 1),
            "min": round(float(conso.min()), 1),
            "max": round(float(conso.max()), 1),
            "ecart_type": round(float(conso.std()), 1),
        },
        "par_saison": {
            label: round(float(df[df["saison"] == s]["consommation_moy"].mean()), 1)
            for s, label in saison_labels.items()
        },
        "semaine_vs_weekend": {
            "semaine": round(float(df[df["is_weekend"] == 0]["consommation_moy"].mean()), 1),
            "weekend": round(float(df[df["is_weekend"] == 1]["consommation_moy"].mean()), 1),
        },
        "correlation_temp_conso": round(float(df["temperature_moy"].corr(df["consommation_moy"])), 4),
        "valeurs_manquantes": int(df.isnull().sum().sum()),
    }


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_pipeline(start_date: str, end_date: str, force: bool = False):
    """
    Exécute le pipeline complet :
    1. Connexion MinIO + init buckets
    2. Collecte RTE (API ou simulé)
    3. Upload données brutes → edf-raw
    4. Prétraitement
    5. Upload données traitées → edf-processed
    6. Génération rapport + upload → edf-reports
    """
    tag = f"{start_date}_{end_date}"
    raw_object      = f"rte/raw_{tag}.csv"
    processed_object= f"daily/rte_daily_{tag}.csv"
    report_object   = f"exploration/report_{tag}.json"

    print("\n" + "=" * 60)
    print("  PIPELINE EDF — COLLECTE & TRAITEMENT")
    print(f"  Période : {start_date} → {end_date}")
    print("=" * 60)

    # 1. Connexion MinIO
    print("\n[MINIO] Connexion...")
    client = get_client()
    init_buckets(client)

    # 2. Vérification si déjà traité (idempotence)
    if not force and object_exists(client, BUCKET_PROCESSED, processed_object):
        print(f"\n[INFO] Données déjà présentes sur MinIO : s3://{BUCKET_PROCESSED}/{processed_object}")
        print("[INFO] Utilisez --force pour re-traiter. Pipeline interrompu.")
        return

    # 3. Collecte
    print("\n[ÉTAPE 1/4] Collecte des données RTE...")
    df_raw = fetch_rte_data(start_date, end_date)

    # 4. Upload brut
    print("\n[ÉTAPE 2/4] Sauvegarde données brutes sur MinIO...")
    upload_dataframe(client, BUCKET_RAW, raw_object, df_raw)

    # 5. Prétraitement
    print("\n[ÉTAPE 3/4] Prétraitement...")
    df_daily = preprocess(df_raw)

    # 6. Upload traité
    print("\n[ÉTAPE 4/4] Sauvegarde données traitées + rapport sur MinIO...")
    upload_dataframe(client, BUCKET_PROCESSED, processed_object, df_daily)

    # 7. Rapport
    report = build_report(df_daily, start_date, end_date)
    upload_json(client, BUCKET_REPORTS, report_object, report)

    # 8. Résumé
    print("\n" + "=" * 60)
    print("  PIPELINE TERMINÉ ✓")
    print(f"  Brut      → s3://{BUCKET_RAW}/{raw_object}")
    print(f"  Traité    → s3://{BUCKET_PROCESSED}/{processed_object}")
    print(f"  Rapport   → s3://{BUCKET_REPORTS}/{report_object}")
    print(f"\n  Jours traités   : {report['meta']['nb_jours']}")
    print(f"  Conso moyenne   : {report['consommation_mw']['moyenne']:,.0f} MW")
    print(f"  Corr. temp/conso: {report['correlation_temp_conso']}")
    print("=" * 60)

    return df_daily, report


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline EDF : collecte RTE → prétraitement → MinIO"
    )
    parser.add_argument("start", nargs="?", default=DATE_START, help="Date début YYYY-MM-DD")
    parser.add_argument("end",   nargs="?", default=DATE_END,   help="Date fin   YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Re-traite même si déjà présent sur MinIO")

    args = parser.parse_args()
    run_pipeline(args.start, args.end, force=args.force)