"""
api/main.py — API FastAPI de prédiction EDF
Projet EDF — Prédiction consommation électrique

Endpoints :
  GET  /health          → état de l'API et du modèle chargé
  GET  /model/info      → infos et métriques du modèle actif
  POST /predict         → prédiction pour un jour donné
  POST /predict/batch   → prédictions pour plusieurs jours
  GET  /metrics         → métriques de monitoring (nb appels, latences)
"""

import os
import sys
import time
import pickle
import numpy as np
from datetime import datetime, date
from typing import Optional
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    BUCKET_MODELS, BUCKET_REPORTS, FEATURE_COLS,
)
from minio_client import get_client, download_model, download_json

# ─────────────────────────────────────────────
# ÉTAT GLOBAL DE L'API
# ─────────────────────────────────────────────

class AppState:
    model        = None
    model_name   = "best_model"
    model_tag    = os.getenv("MODEL_TAG", "2021-01-01_2023-12-31")
    model_info   = {}
    call_count   = 0
    error_count  = 0
    latencies_ms = deque(maxlen=100)   # fenêtre glissante 100 derniers appels
    loaded_at    = None

state = AppState()


# ─────────────────────────────────────────────
# CHARGEMENT DU MODÈLE AU DÉMARRAGE
# ─────────────────────────────────────────────

def load_model_from_minio():
    """Charge le meilleur modèle depuis MinIO au démarrage de l'API."""
    print(f"[API] Chargement du modèle depuis MinIO (tag={state.model_tag})...")
    try:
        client = get_client()
        state.model = download_model(
            client, BUCKET_MODELS,
            f"{state.model_tag}/best_model.pkl"
        )
        # Chargement des métriques associées
        try:
            report = download_json(
                client, BUCKET_REPORTS,
                f"models/comparison_{state.model_tag}.json"
            )
            best_name = report["meilleur_modele"]["nom"]
            state.model_info = {
                "nom":       best_name,
                "tag":       state.model_tag,
                "r2_test":   report["meilleur_modele"]["r2_test"],
                "rmse_test": report["meilleur_modele"]["rmse_test"],
                "mape_test": report["meilleur_modele"]["mape_test"],
            }
        except Exception:
            state.model_info = {"nom": "best_model", "tag": state.model_tag}

        state.loaded_at = datetime.now().isoformat()
        print(f"[API] ✓ Modèle chargé : {state.model_info.get('nom', 'best_model')}")

    except Exception as e:
        print(f"[API] ✗ Impossible de charger le modèle depuis MinIO : {e}")
        print("[API] L'API démarre sans modèle — /predict retournera 503.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_from_minio()
    yield


# ─────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────

app = FastAPI(
    title="EDF — API de Prédiction de Consommation Électrique",
    description=(
        "API de prédiction de la consommation électrique journalière "
        "basée sur des modèles ML entraînés sur les données RTE éco2mix."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# SCHÉMAS PYDANTIC
# ─────────────────────────────────────────────

class PredictionInput(BaseModel):
    """Corps d'une requête de prédiction pour un jour donné."""

    date:             Optional[str]   = Field(None,  description="Date YYYY-MM-DD (optionnel, pour traçabilité)")
    temperature_moy:  float           = Field(...,   description="Température moyenne du jour (°C)", ge=-30, le=50)
    mois:             int             = Field(...,   description="Mois (1-12)", ge=1, le=12)
    jour_semaine:     int             = Field(...,   description="Jour de la semaine (0=lundi, 6=dimanche)", ge=0, le=6)
    is_weekend:       int             = Field(...,   description="1 si week-end, 0 sinon", ge=0, le=1)
    saison:           int             = Field(...,   description="Saison : 0=hiver, 1=printemps, 2=été, 3=automne", ge=0, le=3)
    trimestre:        int             = Field(...,   description="Trimestre (1-4)", ge=1, le=4)
    jour_annee:       int             = Field(...,   description="Jour de l'année (1-365)", ge=1, le=366)

    @field_validator("is_weekend")
    @classmethod
    def check_weekend_coherence(cls, v, info):
        """Vérifie la cohérence entre jour_semaine et is_weekend."""
        if "jour_semaine" in info.data:
            expected = 1 if info.data["jour_semaine"] >= 5 else 0
            if v != expected:
                raise ValueError(
                    f"is_weekend={v} incohérent avec jour_semaine={info.data['jour_semaine']}"
                )
        return v

    def to_feature_vector(self) -> np.ndarray:
        """Convertit l'input en vecteur de features dans l'ordre attendu par le modèle."""
        mois_sin = float(np.sin(2 * np.pi * self.mois / 12))
        mois_cos = float(np.cos(2 * np.pi * self.mois / 12))
        jour_sin = float(np.sin(2 * np.pi * self.jour_semaine / 7))
        jour_cos = float(np.cos(2 * np.pi * self.jour_semaine / 7))

        # Ordre identique à FEATURE_COLS dans config.py
        vector = [
            self.temperature_moy,
            mois_sin,
            mois_cos,
            jour_sin,
            jour_cos,
            self.is_weekend,
            self.saison,
            self.trimestre,
            self.jour_annee,
        ]
        return np.array(vector).reshape(1, -1)


class PredictionOutput(BaseModel):
    """Réponse d'une prédiction."""
    date:                Optional[str]
    consommation_predite_mw: float    = Field(..., description="Consommation prédite en MW")
    modele_utilise:      str
    latence_ms:          float
    timestamp:           str


class BatchInput(BaseModel):
    """Corps d'une requête de prédiction par lot."""
    predictions: list[PredictionInput] = Field(..., min_length=1, max_length=365)


class BatchOutput(BaseModel):
    predictions:  list[PredictionOutput]
    nb_predictions: int
    latence_totale_ms: float


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", tags=["Monitoring"])
def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    model_ok = state.model is not None
    return {
        "status":     "ok" if model_ok else "degraded",
        "model_ok":   model_ok,
        "model_tag":  state.model_tag,
        "loaded_at":  state.loaded_at,
        "timestamp":  datetime.now().isoformat(),
    }


@app.get("/model/info", tags=["Modèle"])
def model_info():
    """Retourne les informations et métriques du modèle actuellement chargé."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé.")
    return {
        "modele":      state.model_info,
        "features":    FEATURE_COLS,
        "loaded_at":   state.loaded_at,
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction"])
def predict(body: PredictionInput):
    """
    Prédit la consommation électrique journalière (en MW)
    à partir des features météo et temporelles du jour.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible. Vérifiez MinIO.")

    t0 = time.perf_counter()
    try:
        X = body.to_feature_vector()
        prediction = float(state.model.predict(X)[0])
        latence_ms = round((time.perf_counter() - t0) * 1000, 3)

        state.call_count += 1
        state.latencies_ms.append(latence_ms)

        return PredictionOutput(
            date=body.date,
            consommation_predite_mw=round(prediction, 1),
            modele_utilise=state.model_info.get("nom", "best_model"),
            latence_ms=latence_ms,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        state.error_count += 1
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")


@app.post("/predict/batch", response_model=BatchOutput, tags=["Prédiction"])
def predict_batch(body: BatchInput):
    """Prédit la consommation pour un lot de jours (max 365)."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    t0 = time.perf_counter()
    results = []
    try:
        X_all = np.vstack([item.to_feature_vector() for item in body.predictions])
        preds = state.model.predict(X_all)

        for item, pred in zip(body.predictions, preds):
            results.append(PredictionOutput(
                date=item.date,
                consommation_predite_mw=round(float(pred), 1),
                modele_utilise=state.model_info.get("nom", "best_model"),
                latence_ms=0,
                timestamp=datetime.now().isoformat(),
            ))

        latence_totale = round((time.perf_counter() - t0) * 1000, 3)
        state.call_count += len(body.predictions)
        state.latencies_ms.append(latence_totale)

        return BatchOutput(
            predictions=results,
            nb_predictions=len(results),
            latence_totale_ms=latence_totale,
        )

    except Exception as e:
        state.error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Métriques de monitoring en temps réel :
    nombre d'appels, taux d'erreur, latences (min/moy/max/p95).
    """
    latencies = list(state.latencies_ms)
    if latencies:
        lat_stats = {
            "min_ms":  round(min(latencies), 3),
            "moy_ms":  round(sum(latencies) / len(latencies), 3),
            "max_ms":  round(max(latencies), 3),
            "p95_ms":  round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        }
    else:
        lat_stats = {"min_ms": 0, "moy_ms": 0, "max_ms": 0, "p95_ms": 0}

    total = state.call_count + state.error_count
    return {
        "appels_total":    state.call_count,
        "erreurs_total":   state.error_count,
        "taux_erreur_pct": round(state.error_count / total * 100, 2) if total > 0 else 0,
        "latences":        lat_stats,
        "modele_tag":      state.model_tag,
        "timestamp":       datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
# LANCEMENT LOCAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )