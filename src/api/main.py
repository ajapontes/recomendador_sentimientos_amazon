# src/api/main.py
# -*- coding: utf-8 -*-
"""
API del Recomendador con endpoints:
- /health
- /recommend/global
- /recommend/user/{user_id}
- /recommend/hybrid/global
- /recommend/hybrid/user/{user_id}

Notas:
- Carga datos (sample) y entrena el baseline de popularidad al iniciar.
- Carga agregación de sentimiento por producto y construye el ranking híbrido bajo demanda.
- Parámetros del híbrido ajustables vía query params (alpha, min_reviews_for_sent).
"""

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from src.utils.config import load_settings
from src.models.data_loader import load_ratings
from src.models.baseline import PopularityRecommender
from src.models.hybrid import (
    load_product_sentiment,
    combine_popularity_and_sentiment,
    HybridParams,
)

app = FastAPI(title="Recomendador Sentimientos Amazon", version="0.1.0")
settings = load_settings()

# Caches simples en memoria (para demo)
_df_cache = None               # DataFrame de reseñas (sample)
_model_pop_cache = None        # PopularityRecommender ya entrenado
_pop_scores_cache = None       # DataFrame con score bayesiano por product_id
_prod_sent_cache = None        # Agregación de sentimiento por product_id


@app.on_event("startup")
def startup_event() -> None:
    """
    Carga datos y modelos base al iniciar la API.
    - Usa el sample (100k) para agilidad.
    - Entrena popularidad (Bayes).
    - Carga agregación de sentimiento por producto (generada en Paso 4.2).
    """
    global _df_cache, _model_pop_cache, _pop_scores_cache, _prod_sent_cache

    # Datos
    _df_cache = load_ratings(sample=True)

    # Baseline: Popularidad (Bayes)
    _model_pop_cache = PopularityRecommender(m=None, topk=20).fit(_df_cache)
    _pop_scores_cache = _model_pop_cache.item_scores  # index=product_id

    # Sentimiento por producto
    try:
        _prod_sent_cache = load_product_sentiment()
    except Exception as e:
        # Permitimos levantar la API aunque falte el archivo de sentimiento
        # (los endpoints híbridos devolverán 500 si no está disponible)
        print(f"[WARN] No se pudo cargar sentimiento agregado: {e}")
        _prod_sent_cache = None


@app.get("/health")
def health() -> dict:
    """Endpoint de salud básica del servicio."""
    return {"status": "ok", "project": settings.project.name, "api_port": settings.api.port}


# =========================
# Endpoints baseline
# =========================

@app.get("/recommend/global")
def recommend_global(n: int = 10) -> list[dict]:
    """
    Top-N global por popularidad (Bayes).
    """
    if _model_pop_cache is None:
        raise HTTPException(500, "Modelo de popularidad no cargado")
    recs = _model_pop_cache.recommend_global(n)
    return recs.to_dict(orient="records")


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: str, n: int = 10) -> list[dict]:
    """
    Top-N para un usuario (excluye ítems ya vistos por el usuario) por popularidad.
    """
    if _model_pop_cache is None or _df_cache is None:
        raise HTTPException(500, "Modelo de popularidad o datos no cargados")
    recs = _model_pop_cache.recommend_for_user(user_id, _df_cache, n)
    return recs.to_dict(orient="records")


# =========================
# Endpoints híbridos
# =========================

def _ensure_hybrid_ready() -> None:
    """
    Valida que tengamos popularidad y sentimiento en memoria.
    Lanza HTTP 500 si no hay sentimiento (falta ejecutar Paso 4.2).
    """
    if _pop_scores_cache is None:
        raise HTTPException(500, "Scores de popularidad no disponibles.")
    if _prod_sent_cache is None:
        raise HTTPException(
            500,
            "Agregación de sentimiento no disponible. "
            "Ejecuta el Paso 4.2 (annotate_sentiment) antes de usar endpoints híbridos."
        )


@app.get("/recommend/hybrid/global")
def recommend_hybrid_global(
    n: int = 10,
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="Peso de popularidad (0..1)"),
    min_reviews_for_sent: int = Query(3, ge=0, description="Mín. reseñas con sentimiento para usar señal")
) -> list[dict]:
    """
    Top-N global híbrido: hybrid = alpha * pop_norm + (1 - alpha) * sent_norm_adjusted.
    """
    _ensure_hybrid_ready()
    params = HybridParams(alpha=alpha, min_reviews_for_sent=min_reviews_for_sent)
    hybrid = combine_popularity_and_sentiment(_pop_scores_cache, _prod_sent_cache, params)
    return hybrid.head(n).to_dict(orient="records")


@app.get("/recommend/hybrid/user/{user_id}")
def recommend_hybrid_user(
    user_id: str,
    n: int = 10,
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="Peso de popularidad (0..1)"),
    min_reviews_for_sent: int = Query(3, ge=0, description="Mín. reseñas con sentimiento para usar señal")
) -> list[dict]:
    """
    Top-N híbrido para un usuario:
    - Genera ranking híbrido global.
    - Excluye ítems ya vistos por el usuario.
    """
    _ensure_hybrid_ready()

    # 1) Ranking híbrido global
    params = HybridParams(alpha=alpha, min_reviews_for_sent=min_reviews_for_sent)
    hybrid = combine_popularity_and_sentiment(_pop_scores_cache, _prod_sent_cache, params)

    # 2) Excluir vistos por el usuario
    seen = set(_df_cache.loc[_df_cache["user_id"] == str(user_id), "product_id"].astype(str))
    hybrid_user = hybrid[~hybrid["product_id"].isin(seen)].head(n)
    return hybrid_user.to_dict(orient="records")
