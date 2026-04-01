"""
minio_client.py — Client MinIO centralisé
Projet EDF — Prédiction consommation électrique

Fournit :
- La connexion MinIO configurée
- La création automatique des buckets
- Les fonctions upload/download CSV, JSON, pickle
"""

import io
import json
import pickle
import pandas as pd
from minio import Minio
from minio.error import S3Error
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE,
    BUCKET_RAW, BUCKET_PROCESSED, BUCKET_MODELS, BUCKET_REPORTS,
)

ALL_BUCKETS = [BUCKET_RAW, BUCKET_PROCESSED, BUCKET_MODELS, BUCKET_REPORTS]


def get_client() -> Minio:
    """Retourne un client MinIO connecté et prêt à l'emploi."""
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    return client


def init_buckets(client: Minio):
    """Crée tous les buckets du projet s'ils n'existent pas encore."""
    for bucket in ALL_BUCKETS:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print(f"[MinIO] Bucket créé : {bucket}")
        else:
            print(f"[MinIO] Bucket existant : {bucket}")


# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

def upload_dataframe(client: Minio, bucket: str, object_name: str, df: pd.DataFrame):
    """
    Upload un DataFrame pandas en CSV vers MinIO.
    Le fichier est streamé en mémoire (pas de fichier temporaire).
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    buffer = io.BytesIO(csv_bytes)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=buffer,
        length=len(csv_bytes),
        content_type="text/csv",
    )
    print(f"[MinIO] ✓ Upload CSV → s3://{bucket}/{object_name}  ({len(csv_bytes)/1024:.1f} Ko)")


def upload_json(client: Minio, bucket: str, object_name: str, data: dict):
    """Upload un dictionnaire Python en JSON vers MinIO."""
    json_bytes = json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    buffer = io.BytesIO(json_bytes)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=buffer,
        length=len(json_bytes),
        content_type="application/json",
    )
    print(f"[MinIO] ✓ Upload JSON → s3://{bucket}/{object_name}  ({len(json_bytes)/1024:.1f} Ko)")


def upload_model(client: Minio, bucket: str, object_name: str, model_object):
    """Upload un modèle scikit-learn (ou tout objet Python) sérialisé en pickle."""
    model_bytes = pickle.dumps(model_object)
    buffer = io.BytesIO(model_bytes)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=buffer,
        length=len(model_bytes),
        content_type="application/octet-stream",
    )
    print(f"[MinIO] ✓ Upload modèle → s3://{bucket}/{object_name}  ({len(model_bytes)/1024:.1f} Ko)")


def upload_bytes(client: Minio, bucket: str, object_name: str, raw_bytes: bytes, content_type: str = "application/octet-stream"):
    """Upload des bytes bruts vers MinIO (générique)."""
    buffer = io.BytesIO(raw_bytes)
    client.put_object(
        bucket_name=bucket,
        object_name=object_name,
        data=buffer,
        length=len(raw_bytes),
        content_type=content_type,
    )
    print(f"[MinIO] ✓ Upload → s3://{bucket}/{object_name}  ({len(raw_bytes)/1024:.1f} Ko)")


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────

def download_dataframe(client: Minio, bucket: str, object_name: str) -> pd.DataFrame:
    """Télécharge un CSV depuis MinIO et retourne un DataFrame pandas."""
    response = client.get_object(bucket, object_name)
    df = pd.read_csv(io.BytesIO(response.read()))
    print(f"[MinIO] ✓ Download CSV ← s3://{bucket}/{object_name}  ({len(df)} lignes)")
    return df


def download_json(client: Minio, bucket: str, object_name: str) -> dict:
    """Télécharge un JSON depuis MinIO et retourne un dictionnaire Python."""
    response = client.get_object(bucket, object_name)
    data = json.loads(response.read().decode("utf-8"))
    print(f"[MinIO] ✓ Download JSON ← s3://{bucket}/{object_name}")
    return data


def download_model(client: Minio, bucket: str, object_name: str):
    """Télécharge et désérialise un modèle pickle depuis MinIO."""
    response = client.get_object(bucket, object_name)
    model = pickle.loads(response.read())
    print(f"[MinIO] ✓ Download modèle ← s3://{bucket}/{object_name}")
    return model


# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────

def list_objects(client: Minio, bucket: str, prefix: str = "") -> list:
    """Liste les objets d'un bucket (avec préfixe optionnel)."""
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    names = [obj.object_name for obj in objects]
    print(f"[MinIO] {len(names)} objets dans s3://{bucket}/{prefix}")
    return names


def object_exists(client: Minio, bucket: str, object_name: str) -> bool:
    """Vérifie si un objet existe dans MinIO (évite les re-traitements inutiles)."""
    try:
        client.stat_object(bucket, object_name)
        return True
    except S3Error:
        return False