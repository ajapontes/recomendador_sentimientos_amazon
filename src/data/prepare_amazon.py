"""
src/data/prepare_amazon.py — Fuente: parquet público (ClickHouse docs)
URL: https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/amazon_reviews_2015.snappy.parquet

Genera:
  - data/processed/electronics_full.parquet
  - data/processed/electronics_sample_100k.parquet
"""

from pathlib import Path
from typing import List
import random
import io
import requests
import pandas as pd
import numpy as np
import yaml

PARQUET_URL = "https://datasets-documentation.s3.eu-west-3.amazonaws.com/amazon_reviews/amazon_reviews_2015.snappy.parquet"

TARGET_COLUMNS = [
    "customer_id", "product_id", "star_rating",
    "review_body", "review_headline", "review_date", "product_title"
]
STANDARD_RENAME = {
    "customer_id": "user_id",
    "product_id": "product_id",
    "star_rating": "rating",
    "review_body": "review_text",
    "review_headline": "review_title",
    "review_date": "review_date",
    "product_title": "product_title",
}

def load_settings(path: str = "settings.yaml") -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def main() -> None:
    cfg = load_settings()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    raw_dir = Path(cfg["paths"]["data_raw"])
    proc_dir = Path(cfg["paths"]["data_processed"])
    ensure_dirs(raw_dir.as_posix(), proc_dir.as_posix())

    print("[INFO] Descargando parquet público (2015)...")
    r = requests.get(PARQUET_URL, timeout=120)
    r.raise_for_status()
    bio = io.BytesIO(r.content)

    print("[INFO] Leyendo parquet en memoria...")
    df_all = pd.read_parquet(bio)  # requiere pyarrow

    # Filtrar solo categoría Electronics
    if "product_category" not in df_all.columns:
        raise ValueError("El parquet no tiene 'product_category'; no podemos filtrar Electronics.")
    df = df_all[df_all["product_category"] == "Electronics"].copy()

    # Subseleccionar y renombrar columnas clave
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")
    df = df[TARGET_COLUMNS].rename(columns=STANDARD_RENAME)

    # Tipado y limpieza mínima
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Float64")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    for c in ("review_text", "review_title", "product_title"):
        df[c] = df[c].astype(str).str.strip()

    # Drops básicos
    df = df.dropna(subset=["user_id", "product_id", "rating", "review_text"])

    # Guardar full
    full_path = proc_dir / "electronics_full.parquet"
    print(f"[INFO] Guardando dataset completo en {full_path} ...")
    df.to_parquet(full_path, index=False)

    # Sample 100k reproducible
    sample_size = min(100_000, len(df))
    df_sample = df.sample(n=sample_size, random_state=seed)
    sample_path = proc_dir / "electronics_sample_100k.parquet"
    print(f"[INFO] Guardando muestra en {sample_path} ...")
    df_sample.to_parquet(sample_path, index=False)

    # Stats rápidas
    print("[INFO] Estadísticas:")
    print("  - Registros totales:", len(df))
    print("  - Registros muestra:", len(df_sample))
    print("  - Usuarios únicos:", df["user_id"].nunique())
    print("  - Productos únicos:", df["product_id"].nunique())
    print("  - Rating promedio:", round(df["rating"].astype(float).mean(), 3))
    if df["review_date"].notna().any():
        print("  - Fecha min:", df["review_date"].min())
        print("  - Fecha max:", df["review_date"].max())
    print("[OK] Preparación completada (Electronics).")

if __name__ == "__main__":
    main()
