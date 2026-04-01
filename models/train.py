"""
models/train.py — Entraînement des 4 modèles ML
Projet EDF — Prédiction consommation électrique

Modèles :
  1. Random Forest
  2. K-Nearest Neighbors (KNN)
  3. Arbre de Décision
  4. Réseau de neurones RBF (MLPRegressor)

Métriques calculées : R², RMSE, MAPE, temps d'entraînement
Sauvegarde : modèles pickle + rapport JSON → MinIO
"""

import sys
import os
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    BUCKET_PROCESSED, BUCKET_MODELS, BUCKET_REPORTS,
    FEATURE_COLS, TARGET_COL, TEST_SIZE, RANDOM_STATE,
)
from minio_client import (
    get_client, download_dataframe,
    upload_model, upload_json, list_objects,
)


# ─────────────────────────────────────────────
# DÉFINITION DES MODÈLES
# ─────────────────────────────────────────────

def get_models() -> dict:
    """
    Retourne les 4 modèles avec leurs hyperparamètres.
    Chaque modèle est encapsulé dans un Pipeline scikit-learn
    avec un StandardScaler pour normaliser les features.
    Note : Random Forest et Decision Tree sont robustes au scaling,
    mais on l'applique uniformément pour la cohérence du pipeline.
    """
    return {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=7,
                weights="distance",
                metric="euclidean",
                n_jobs=-1,
            )),
        ]),

        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("model", DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )),
        ]),

        "rbf_neural_net": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ─────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_time: float) -> dict:
    """
    Calcule les métriques de performance demandées par le cahier des charges :
    - R²     : coefficient de détermination (1.0 = parfait)
    - RMSE   : Root Mean Squared Error (MW)
    - MAPE   : Mean Absolute Percentage Error (%)
    - Temps  : durée d'entraînement (secondes)
    """
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100

    return {
        "r2":         round(r2, 6),
        "rmse_mw":    round(rmse, 2),
        "mape_pct":   round(mape, 4),
        "train_time_s": round(train_time, 3),
    }


# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────

def train_all(df: pd.DataFrame) -> dict:
    """
    Entraîne les 4 modèles sur les données journalières prétraitées.
    Retourne un dictionnaire avec modèles entraînés + métriques.
    """
    print("\n[PRÉPARATION] Séparation features / cible...")

    # Vérification que toutes les features sont présentes
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes dans le dataset : {missing}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Split train / test chronologique (pas de shuffle — respect de l'ordre temporel)
    split_idx = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train : {len(X_train)} jours  |  Test : {len(X_test)} jours")
    print(f"  Features ({len(FEATURE_COLS)}) : {FEATURE_COLS}")
    print(f"  Cible : {TARGET_COL}")

    models = get_models()
    results = {}

    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT DES MODÈLES")
    print("=" * 60)

    for name, pipeline in models.items():
        print(f"\n[{name.upper()}]")

        # Entraînement
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - t0

        # Prédictions
        y_pred_train = pipeline.predict(X_train)
        y_pred_test  = pipeline.predict(X_test)

        # Métriques
        train_metrics = compute_metrics(y_train, y_pred_train, train_time)
        test_metrics  = compute_metrics(y_test,  y_pred_test,  train_time)

        print(f"  Temps entraînement : {train_time:.3f}s")
        print(f"  R²   (train/test)  : {train_metrics['r2']:.4f}  /  {test_metrics['r2']:.4f}")
        print(f"  RMSE (train/test)  : {train_metrics['rmse_mw']:,.0f} MW  /  {test_metrics['rmse_mw']:,.0f} MW")
        print(f"  MAPE (train/test)  : {train_metrics['mape_pct']:.2f}%  /  {test_metrics['mape_pct']:.2f}%")

        results[name] = {
            "pipeline":      pipeline,
            "train_metrics": train_metrics,
            "test_metrics":  test_metrics,
            "y_test":        y_test.tolist(),
            "y_pred_test":   y_pred_test.tolist(),
        }

    return results, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# SÉLECTION DU MEILLEUR MODÈLE
# ─────────────────────────────────────────────

def select_best(results: dict) -> str:
    """
    Sélectionne le meilleur modèle selon le R² sur le jeu de test.
    En cas d'égalité, privilégie le RMSE le plus faible.
    """
    best_name = max(
        results.keys(),
        key=lambda n: (
            results[n]["test_metrics"]["r2"],
            -results[n]["test_metrics"]["rmse_mw"],
        )
    )
    return best_name


# ─────────────────────────────────────────────
# RAPPORT COMPARATIF
# ─────────────────────────────────────────────

def build_comparison_report(results: dict, best_name: str, tag: str) -> dict:
    """Construit le rapport JSON de comparaison des modèles."""
    summary = {}
    for name, res in results.items():
        summary[name] = {
            "train": res["train_metrics"],
            "test":  res["test_metrics"],
        }

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "dataset_tag":  tag,
            "feature_cols": FEATURE_COLS,
            "target_col":   TARGET_COL,
            "test_size":    TEST_SIZE,
        },
        "metriques": summary,
        "meilleur_modele": {
            "nom":      best_name,
            "r2_test":  results[best_name]["test_metrics"]["r2"],
            "rmse_test":results[best_name]["test_metrics"]["rmse_mw"],
            "mape_test":results[best_name]["test_metrics"]["mape_pct"],
            "justification": (
                f"{best_name} obtient le meilleur R² sur le jeu de test "
                f"({results[best_name]['test_metrics']['r2']:.4f}) avec un RMSE de "
                f"{results[best_name]['test_metrics']['rmse_mw']:,.0f} MW, "
                "ce qui en fait le modèle le plus stable et précis pour la production."
            ),
        },
    }


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_training(tag: str = "2021-01-01_2023-12-31"):
    """
    Pipeline complet :
    1. Chargement données depuis MinIO (edf-processed)
    2. Entraînement des 4 modèles
    3. Upload modèles → edf-models
    4. Upload rapport comparatif → edf-reports
    """
    print("\n" + "=" * 60)
    print("  PHASE 2 — ENTRAÎNEMENT DES MODÈLES ML")
    print(f"  Dataset : {tag}")
    print("=" * 60)

    # 1. Connexion MinIO
    client = get_client()

    # 2. Chargement des données prétraitées
    print("\n[MINIO] Chargement des données prétraitées...")
    object_name = f"daily/rte_daily_{tag}.csv"
    df = download_dataframe(client, BUCKET_PROCESSED, object_name)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  → {len(df)} jours chargés.")

    # 3. Entraînement
    results, X_train, X_test, y_train, y_test = train_all(df)

    # 4. Sélection du meilleur modèle
    best_name = select_best(results)
    print(f"\n[RÉSULTAT] Meilleur modèle : {best_name.upper()}")

    # 5. Sauvegarde des modèles sur MinIO
    print("\n[MINIO] Sauvegarde des modèles...")
    for name, res in results.items():
        object_name = f"{tag}/{name}.pkl"
        upload_model(client, BUCKET_MODELS, object_name, res["pipeline"])

    # Sauvegarde du meilleur modèle séparément pour l'API
    upload_model(
        client, BUCKET_MODELS,
        f"{tag}/best_model.pkl",
        results[best_name]["pipeline"],
    )

    # 6. Rapport comparatif
    print("\n[MINIO] Sauvegarde rapport comparatif...")
    report = build_comparison_report(results, best_name, tag)
    upload_json(
        client, BUCKET_REPORTS,
        f"models/comparison_{tag}.json",
        report,
    )

    # 7. Affichage récapitulatif
    print("\n" + "=" * 60)
    print("  RÉCAPITULATIF — MÉTRIQUES SUR JEU DE TEST")
    print("=" * 60)
    print(f"  {'Modèle':<20} {'R²':>8} {'RMSE (MW)':>12} {'MAPE (%)':>10} {'Temps (s)':>10}")
    print("  " + "-" * 62)
    for name, res in sorted(
        results.items(),
        key=lambda x: x[1]["test_metrics"]["r2"],
        reverse=True,
    ):
        m = res["test_metrics"]
        marker = " ← BEST" if name == best_name else ""
        print(
            f"  {name:<20} {m['r2']:>8.4f} {m['rmse_mw']:>12,.0f} "
            f"{m['mape_pct']:>10.2f} {m['train_time_s']:>10.3f}{marker}"
        )
    print("=" * 60)
    print(f"\n  Modèles sauvegardés → s3://{BUCKET_MODELS}/{tag}/")
    print(f"  Rapport             → s3://{BUCKET_REPORTS}/models/comparison_{tag}.json")

    return results, report


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entraîne les 4 modèles ML EDF")
    parser.add_argument(
        "--tag",
        default="2021-01-01_2023-12-31",
        help="Tag du dataset MinIO (ex: 2021-01-01_2023-12-31)",
    )
    args = parser.parse_args()
    run_training(tag=args.tag)