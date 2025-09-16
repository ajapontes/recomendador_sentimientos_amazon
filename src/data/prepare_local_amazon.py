"""
src/data/prepare_local_amazon.py
--------------------------------
Procesa Parquet local (Amazon Reviews 2015) y genera datasets filtrados por
categorías 'electrónicas' (por defecto: ['Electronics']).

Entrada (por defecto):
  ./data/in/amazon_reviews_2015.snappy.parquet

Salida:
  - ./data/processed/electronics_full.parquet
  - ./data/processed/electronics_sample_100k.parquet

Requisitos: pandas, pyarrow, pyyaml
"""

from pathlib import Path
from typing import List
import random
import sys

import numpy as np
import pandas as pd
import yaml


# Columnas esperadas en el Parquet base 2015
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

DEFAULT_INPUT = Path("./data/in/amazon_reviews_2015.snappy.parquet")

# ✅ Lista de categorías a incluir (puedes ampliarla)
CATEGORIES_ELECTRONICS: List[str] = [
    "Electronics",
    # "Wireless", "PC", "Camera", "Mobile_Electronics", "Home Entertainment"
]


def load_settings(path: str = "settings.yaml") -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _decode_bytes_series_if_needed(s: pd.Series) -> pd.Series:
    """Si la serie contiene bytes (b'...'), los decodifica a str UTF-8."""
    if s.dtype == object and any(isinstance(x, (bytes, bytearray)) for x in s.head(100).tolist()):
        return s.apply(lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, bytearray)) else x)
    return s


def main() -> None:
    # Config
    cfg = load_settings()
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    raw_dir = Path(cfg["paths"]["data_raw"])
    proc_dir = Path(cfg["paths"]["data_processed"])
    ensure_dirs(raw_dir.as_posix(), proc_dir.as_posix())

    # Input path (permite argumento)
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    if not in_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de entrada: {in_path}\n"
            "Ejemplo:\n"
            "  python src/data/prepare_local_amazon.py C:/Proyectos/recomendador_sentimientos_amazon/data/in/amazon_reviews_2015.snappy.parquet"
        )

    print(f"[INFO] Leyendo parquet local: {in_path}")
    df_all = pd.read_parquet(in_path)  # requiere pyarrow

    # Validar y decodificar categoría
    if "product_category" not in df_all.columns:
        raise ValueError("El parquet base no tiene la columna 'product_category'.")

    # 👇 Decodificar si viene como bytes
    df_all["product_category"] = _decode_bytes_series_if_needed(df_all["product_category"])

    # Filtrar por categorías deseadas
    df = df_all[df_all["product_category"].isin(CATEGORIES_ELECTRONICS)].copy()
    if df.empty:
        cats_preview = df_all["product_category"].dropna().astype(str).unique()[:20]
        raise ValueError(
            "No se encontraron filas con categorías en "
            f"{CATEGORIES_ELECTRONICS}. Revisa nombres exactos. Ejemplos en el archivo: {list(cats_preview)}"
        )

    # Verificar columnas necesarias
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")

    # Seleccionar y renombrar
    df = df[TARGET_COLUMNS].rename(columns=STANDARD_RENAME)

    # Tipado y limpieza
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Float64")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    for c in ("review_text", "review_title", "product_title"):
        df[c] = df[c].astype(str).str.strip()

    # Eliminar registros incompletos mínimos
    before = len(df)
    df = df.dropna(subset=["user_id", "product_id", "rating", "review_text"])
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Registros eliminados por nulos mínimos: {dropped}")

    # Guardar full
    full_path = proc_dir / "electronics_full.parquet"
    print(f"[INFO] Guardando dataset completo en {full_path} ...")
    df.to_parquet(full_path, index=False)

    # Sample reproducible 100k (o menos)
    sample_size = min(100_000, len(df))
    df_sample = df.sample(n=sample_size, random_state=seed)
    sample_path = proc_dir / "electronics_sample_100k.parquet"
    print(f"[INFO] Guardando muestra en {sample_path} ...")
    df_sample.to_parquet(sample_path, index=False)

    # Stats
    print("[INFO] Estadísticas:")
    print("  - Registros totales:", len(df))
    print("  - Registros muestra:", len(df_sample))
    print("  - Usuarios únicos:", df["user_id"].nunique())
    print("  - Productos únicos:", df["product_id"].nunique())
    print("  - Rating promedio:", round(df["rating"].astype(float).mean(), 3))
    if df["review_date"].notna().any():
        print("  - Fecha min:", df["review_date"].min())
        print("  - Fecha max:", df["review_date"].max())
    print("[OK] Preparación local completada.")


if __name__ == "__main__":
    main()
