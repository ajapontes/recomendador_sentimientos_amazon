# src/api/main.py
# -*- coding: utf-8 -*-
"""
API del Recomendador (Electrónica Amazon) con señales de popularidad y sentimiento.

Endpoints expuestos:
- GET /health
- GET /            -> ping simple
- GET /recommend/global
- GET /recommend/user/{user_id}
- GET /recommend/hybrid/global
- GET /recommend/hybrid/user/{user_id}
- GET /recommend/itemcf/user/{user_id}
- GET /recommend/hybrid_itemcf/user/{user_id}
- GET /metrics      -> métricas ligeras en memoria (6.5.B)

Notas:
- Modelos y datos viven en memoria (demo).
- Logging con timing + métricas por ruta/estado.
"""

import time
import logging
import threading
from collections import defaultdict
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request

from src.utils.logging_setup import setup_logging
from src.utils.config import load_settings
from src.models.data_loader import load_ratings
from src.models.baseline import PopularityRecommender
from src.models.hybrid import (
    load_product_sentiment,
    combine_popularity_and_sentiment,
    HybridParams,
)
from src.models.itemcf import ItemCFRecommender, ItemCFConfig
from src.models.hybrid_itemcf import ItemCFSentimentBooster, ItemCFHybridParams
from src.models.catalog import load_catalog, attach_titles  # catálogo

# ------------------------------------------------------------------------------
# App & Settings
# ------------------------------------------------------------------------------
app = FastAPI(title="Recomendador Sentimientos Amazon", version="0.1.0")
settings = load_settings()

# ------------------------------------------------------------------------------
# Caches simples (demo)
# ------------------------------------------------------------------------------
_df_cache = None               # DataFrame de reseñas (sample)
_model_pop_cache = None        # PopularityRecommender ya entrenado
_pop_scores_cache = None       # Series/DataFrame: score bayesiano por product_id (index)
_prod_sent_cache = None        # Agregación de sentimiento por product_id
_catalog_cache = None          # DataFrame: product_id -> product_title

_itemcf_model_cache = None     # ItemCFRecommender entrenado
_itemcf_cfg_cache = None       # Config usada para entrenar ItemCF
_itemcf_booster = None         # Reponderador por sentimiento para ItemCF

# ------------------------------------------------------------------------------
# Métricas en memoria (6.5.B)
# - Protegidas con un lock para threads de Uvicorn/ASGI.
# ------------------------------------------------------------------------------
_metrics_lock = threading.Lock()
_metrics_total: Dict[str, Any] = {
    "requests": 0,
    "ok": 0,
    "error": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
}
_metrics_by_status: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
    "count": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
    "avg_ms": 0.0,
})
_metrics_by_route: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
    "count": 0,
    "ok": 0,
    "error": 0,
    "total_ms": 0.0,
    "max_ms": 0.0,
    "avg_ms": 0.0,
    "last_status": None,
})

def _metrics_update(route_key: str, status: int, duration_ms: float) -> None:
    """Actualiza contadores globales, por status y por ruta."""
    with _metrics_lock:
        # Global
        _metrics_total["requests"] += 1
        _metrics_total["total_ms"] += duration_ms
        if duration_ms > _metrics_total["max_ms"]:
            _metrics_total["max_ms"] = duration_ms
        if 200 <= status < 400:
            _metrics_total["ok"] += 1
        else:
            _metrics_total["error"] += 1

        # Por status
        s = _metrics_by_status[status]
        s["count"] += 1
        s["total_ms"] += duration_ms
        if duration_ms > s["max_ms"]:
            s["max_ms"] = duration_ms
        s["avg_ms"] = s["total_ms"] / max(s["count"], 1)

        # Por ruta
        r = _metrics_by_route[route_key]
        r["count"] += 1
        r["total_ms"] += duration_ms
        if duration_ms > r["max_ms"]:
            r["max_ms"] = duration_ms
        if 200 <= status < 400:
            r["ok"] += 1
        else:
            r["error"] += 1
        r["avg_ms"] = r["total_ms"] / max(r["count"], 1)
        r["last_status"] = status

def _metrics_snapshot() -> Dict[str, Any]:
    """Copia segura de las métricas para exponer en /metrics."""
    with _metrics_lock:
        return {
            "total": {
                **_metrics_total,
                "avg_ms": (_metrics_total["total_ms"] / _metrics_total["requests"]) if _metrics_total["requests"] else 0.0
            },
            "by_status": {int(k): dict(v) for k, v in _metrics_by_status.items()},
            "by_route": {str(k): dict(v) for k, v in _metrics_by_route.items()},
        }

# ------------------------------------------------------------------------------
# Startup: carga datos y modelos en memoria
# ------------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    """
    Carga dataset y entrena/precarga los componentes del servicio.
    """
    global _df_cache, _model_pop_cache, _pop_scores_cache, _prod_sent_cache
    global _itemcf_model_cache, _itemcf_cfg_cache, _itemcf_booster, _catalog_cache

    # Logging
    try:
        setup_logging(level="INFO")   # o lee de settings si prefieres
    except Exception as e:
        print(f"[WARN] No se pudo inicializar logging: {e}")

    # 1) Datos (sample 100k para agilidad)
    _df_cache = load_ratings(sample=True)

    # 2) Catálogo product_id -> product_title
    try:
        _catalog_cache = load_catalog(sample=True)
    except Exception as e:
        print(f"[WARN] No se pudo construir catálogo de títulos: {e}")
        _catalog_cache = None

    # 3) Baseline: Popularidad (Bayes)
    _model_pop_cache = PopularityRecommender(m=None, topk=20).fit(_df_cache)
    _pop_scores_cache = _model_pop_cache.item_scores  # index=product_id

    # 4) Sentimiento por producto (puede fallar si no ejecutaste la anotación)
    try:
        _prod_sent_cache = load_product_sentiment()
    except Exception as e:
        print(f"[WARN] No se pudo cargar sentimiento agregado: {e}")
        _prod_sent_cache = None

    # 5) ItemCF (personalizado) + Booster de sentimiento
    try:
        icf_cfg = ItemCFConfig(
            min_rating_like=3.0,
            min_item_freq=5,
            min_user_interactions=2,
            n_neighbors=800,
            topk_recommendations=20,
        )
        _itemcf_cfg_cache = icf_cfg
        _itemcf_model_cache = ItemCFRecommender(icf_cfg).fit(_df_cache)
        _itemcf_booster = ItemCFSentimentBooster() if _prod_sent_cache is not None else None
        print("[INFO] ItemCF listo. Booster de sentimiento:", "OK" if _itemcf_booster else "NO DISPONIBLE")
    except Exception as e:
        print(f"[WARN] No se pudo entrenar ItemCF/booster: {e}")
        _itemcf_model_cache = None
        _itemcf_booster = None

# ------------------------------------------------------------------------------
# Utilidades internas
# ------------------------------------------------------------------------------
def _ensure_hybrid_ready() -> None:
    """
    Requiere popularidad y sentimiento en memoria para endpoints híbridos globales.
    """
    if _pop_scores_cache is None:
        raise HTTPException(500, "Scores de popularidad no disponibles.")
    if _prod_sent_cache is None:
        raise HTTPException(
            500,
            "Agregación de sentimiento no disponible. "
            "Ejecuta el Paso 4.2 (annotate_sentiment) antes de usar endpoints híbridos."
        )

def _enrich_response(
    df,
    include_titles: bool = True,
    include_sentiment: bool = False,
    keep_cols: list[str] | None = None,
):
    """
    Enriquecedor estándar de respuestas:
    - include_titles: adjunta product_title desde _catalog_cache
    - include_sentiment: adjunta n_scored y sentiment_mean (si _prod_sent_cache existe)
    - keep_cols: orden/selección final de columnas
    """
    if df is None:
        return []
    if hasattr(df, "empty") and df.empty:
        return []

    out = df.copy()
    if "product_id" in out.columns:
        out["product_id"] = out["product_id"].astype(str)

    if include_titles and _catalog_cache is not None and not _catalog_cache.empty:
        out = attach_titles(out, _catalog_cache)

    if include_sentiment and _prod_sent_cache is not None:
        tmp = _prod_sent_cache[["product_id", "n_scored", "sentiment_mean"]].copy()
        tmp["product_id"] = tmp["product_id"].astype(str)
        out = out.merge(tmp, on="product_id", how="left")

    if keep_cols:
        cols = [c for c in keep_cols if c in out.columns]
        out = out[cols]

    return out.to_dict(orient="records")

# ------------------------------------------------------------------------------
# Middleware de logging + métricas (timing)
# ------------------------------------------------------------------------------
logger = logging.getLogger("api")

@app.middleware("http")
async def access_logger(request: Request, call_next):
    """
    Logging de acceso por request + actualización de métricas.
    Registra: método, path, status, duración_ms, tamaño y user-agent.
    Métricas: globales, por status y por ruta (ruta “resuelta” si es posible).
    """
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        logger.exception("Unhandled exception: %s %s -> 500", request.method, request.url.path)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0

    # Intentar obtener la ruta "resuelta" (con parámetros), ej. /recommend/user/{user_id}
    route_path = request.scope.get("route").path if request.scope.get("route") else request.url.path

    # Logging humano
    resp_len = response.headers.get("content-length", "-")
    logger.info(
        "HTTP %s %s -> %s | %.2fms | bytes=%s | ua=%s",
        request.method,
        route_path,
        status,
        duration_ms,
        resp_len,
        request.headers.get("user-agent", "-"),
    )

    # Métricas
    _metrics_update(route_path, status, duration_ms)
    return response

# ------------------------------------------------------------------------------
# Endpoints básicos
# ------------------------------------------------------------------------------
@app.get("/")
def root() -> dict:
    """Ping simple con meta del servicio."""
    return {
        "status": "ok",
        "project": settings.project.name,
        "api_host": settings.api.host,
        "api_port": settings.api.port,
        "endpoints": [
            "/health",
            "/recommend/global",
            "/recommend/user/{user_id}",
            "/recommend/hybrid/global",
            "/recommend/hybrid/user/{user_id}",
            "/recommend/itemcf/user/{user_id}",
            "/recommend/hybrid_itemcf/user/{user_id}",
            "/metrics",
        ],
    }

@app.get("/health")
def health() -> dict:
    """Endpoint de salud básica del servicio."""
    return {"status": "ok", "project": settings.project.name, "api_port": settings.api.port}

# ------------------------------------------------------------------------------
# Endpoints: Popularidad (baseline)
# ------------------------------------------------------------------------------
@app.get("/recommend/global")
def recommend_global(n: int = 10) -> list[dict]:
    """Top-N global por popularidad (Bayes)."""
    if _model_pop_cache is None:
        raise HTTPException(500, "Modelo de popularidad no cargado.")
    recs = _model_pop_cache.recommend_global(n)
    return _enrich_response(
        recs, include_titles=True, include_sentiment=False,
        keep_cols=["product_id", "product_title", "score", "v", "R"],
    )

@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: str, n: int = 10) -> list[dict]:
    """Top-N por popularidad para un usuario, excluyendo ítems ya vistos."""
    if _model_pop_cache is None or _df_cache is None:
        raise HTTPException(500, "Modelo de popularidad o datos no cargados.")
    recs = _model_pop_cache.recommend_for_user(user_id, _df_cache, n)
    return _enrich_response(
        recs, include_titles=True, include_sentiment=False,
        keep_cols=["product_id", "product_title", "score", "v", "R"],
    )

# ------------------------------------------------------------------------------
# Endpoints: Popularidad ⨉ Sentimiento (híbrido global)
# ------------------------------------------------------------------------------
@app.get("/recommend/hybrid/global")
def recommend_hybrid_global(
    n: int = 10,
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="Peso de popularidad (0..1)"),
    min_reviews_for_sent: int = Query(3, ge=0, description="Mín. reseñas con sentimiento para usar señal"),
) -> list[dict]:
    """Top-N global híbrido: hybrid = alpha*pop_norm + (1-alpha)*sent_norm_adjusted."""
    _ensure_hybrid_ready()
    params = HybridParams(alpha=alpha, min_reviews_for_sent=min_reviews_for_sent)
    hybrid = combine_popularity_and_sentiment(_pop_scores_cache, _prod_sent_cache, params)
    return _enrich_response(
        hybrid.head(n), include_titles=True, include_sentiment=True,
        keep_cols=["product_id", "product_title", "hybrid_score", "score", "sentiment_mean", "n_scored"],
    )

@app.get("/recommend/hybrid/user/{user_id}")
def recommend_hybrid_user(
    user_id: str,
    n: int = 10,
    alpha: float = Query(0.7, ge=0.0, le=1.0, description="Peso de popularidad (0..1)"),
    min_reviews_for_sent: int = Query(3, ge=0, description="Mín. reseñas con sentimiento para usar señal"),
) -> list[dict]:
    """
    Top-N híbrido para un usuario:
    - Genera ranking híbrido global (popularidad ⨉ sentimiento).
    - Excluye ítems ya vistos por el usuario.
    """
    _ensure_hybrid_ready()
    params = HybridParams(alpha=alpha, min_reviews_for_sent=min_reviews_for_sent)
    hybrid = combine_popularity_and_sentiment(_pop_scores_cache, _prod_sent_cache, params)
    seen = set(_df_cache.loc[_df_cache["user_id"] == str(user_id), "product_id"].astype(str))
    hybrid_user = hybrid[~hybrid["product_id"].isin(seen)].head(n)
    return _enrich_response(
        hybrid_user, include_titles=True, include_sentiment=True,
        keep_cols=["product_id", "product_title", "hybrid_score", "score", "sentiment_mean", "n_scored"],
    )

# ------------------------------------------------------------------------------
# Endpoints: ItemCF (personalizado) y Hybrid-ItemCF
# ------------------------------------------------------------------------------
@app.get("/recommend/itemcf/user/{user_id}")
def recommend_itemcf_user(user_id: str, n: int = 10) -> list[dict]:
    """Top-N personalizado con ItemCF (sin sentimiento)."""
    if _itemcf_model_cache is None or _df_cache is None:
        raise HTTPException(500, "ItemCF no está disponible en el servidor.")
    recs = _itemcf_model_cache.recommend_for_user(user_id, _df_cache, n=n)
    return _enrich_response(
        recs.head(n), include_titles=True, include_sentiment=False,
        keep_cols=["product_id", "product_title", "score", "neighbors_hits"],
    )

@app.get("/recommend/hybrid_itemcf/user/{user_id}")
def recommend_hybrid_itemcf_user(
    user_id: str,
    n: int = 10,
    beta: float = Query(0.5, ge=0.0, le=1.0, description="Atenuación del boost de sentimiento (0=pleno, 1=sin efecto)"),
    min_reviews_for_sent: int = Query(3, ge=0, description="Mín. reseñas con sentimiento para usar señal"),
) -> list[dict]:
    """Top-N personalizado con ItemCF reponderado por sentimiento."""
    if _itemcf_model_cache is None or _df_cache is None:
        raise HTTPException(500, "ItemCF no está disponible en el servidor.")
    if _itemcf_booster is None:
        raise HTTPException(500, "Booster de sentimiento no disponible (falta agregación de sentimiento o error de carga).")

    base_recs = _itemcf_model_cache.recommend_for_user(user_id, _df_cache, n=max(n * 5, 50))
    params = ItemCFHybridParams(beta=beta, min_reviews_for_sent=min_reviews_for_sent)
    boosted = _itemcf_booster.boost(base_recs, params)
    return _enrich_response(
        boosted.head(n), include_titles=True, include_sentiment=True,
        keep_cols=["product_id", "product_title", "score_hybrid", "score", "sentiment_mean", "n_scored", "neighbors_hits"],
    )

# ------------------------------------------------------------------------------
# Endpoint de métricas (6.5.B)
# ------------------------------------------------------------------------------
@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """
    Métricas ligeras en memoria (JSON):
    - total: requests, ok, error, avg_ms, max_ms
    - by_status: {status: count, avg_ms, max_ms, ...}
    - by_route: {route: count, ok, error, avg_ms, max_ms, last_status}
    """
    return _metrics_snapshot()
