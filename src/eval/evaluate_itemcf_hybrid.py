# src/eval/evaluate_itemcf_hybrid.py
# -*- coding: utf-8 -*-
"""
Evaluación leave-last-1 comparando:
- ItemCF puro
- Híbrido ItemCF ⨉ Sentimiento (reponderado)

Métrica: HitRate@K
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.data_loader import load_ratings
from src.models.itemcf import ItemCFRecommender, ItemCFConfig
from src.models.hybrid_itemcf import ItemCFSentimentBooster, ItemCFHybridParams


@dataclass(frozen=True)
class EvalConfig:
    K: int = 10
    min_rating_like: float = 3.0
    min_item_freq: int = 5
    min_user_interactions: int = 2
    n_neighbors: int = 800
    beta: float = 0.5
    min_reviews_for_sent: int = 3


def _leave_last_split_user(df_u: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_u) < 2:
        return df_u.iloc[:0], df_u
    return df_u.iloc[:-1], df_u.iloc[-1:]


def hitrate_at_k(candidates: pd.Series, relevant: set, K: int) -> float:
    topk = set(candidates.head(K).tolist())
    return 1.0 if len(topk.intersection(relevant)) > 0 else 0.0


def main():
    cfg = EvalConfig()

    # 1) Cargar sample y ordenar estable por usuario (fecha si existe)
    df = load_ratings(sample=True).copy()
    if "review_date" in df.columns:
        df = df.sort_values(["user_id", "review_date"], kind="mergesort")
    else:
        df = df.sort_values(["user_id"], kind="mergesort")

    # 2) Pre-filtro por likes / usuarios
    likes = df[df["rating"].astype(float) >= cfg.min_rating_like]
    vc_u = likes["user_id"].value_counts()
    valid_users = set(vc_u[vc_u >= cfg.min_user_interactions].index.astype(str))
    df = df[df["user_id"].astype(str).isin(valid_users)].copy()
    if df.empty:
        print("No hay datos tras filtros; ajusta config.")
        return

    # 3) Leave-last-1 per user
    trains, tests = [], []
    for uid, df_u in df.groupby("user_id", sort=False):
        tr, te = _leave_last_split_user(df_u)
        if not tr.empty and not te.empty:
            trains.append(tr)
            tests.append(te)
    if not trains or not tests:
        print("No se pudo construir train/test; revisa filtros.")
        return

    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)

    # Filtrar catálogo por frecuencia en TRAIN
    icount = train["product_id"].value_counts()
    valid_items = set(icount[icount >= cfg.min_item_freq].index.astype(str))
    train = train[train["product_id"].astype(str).isin(valid_items)].copy()
    test = test[test["product_id"].astype(str).isin(valid_items)].copy()

    if train.empty or test.empty:
        print("TRAIN o TEST vacío tras filtros; ajusta min_item_freq.")
        return

    # 4) Entrenar ItemCF en TRAIN
    model = ItemCFRecommender(
        ItemCFConfig(
            min_rating_like=cfg.min_rating_like,
            min_item_freq=cfg.min_item_freq,
            min_user_interactions=cfg.min_user_interactions,
            n_neighbors=cfg.n_neighbors,
            topk_recommendations=cfg.K,
        )
    ).fit(train)

    booster = ItemCFSentimentBooster()

    # 5) Evaluación
    users = test["user_id"].astype(str).unique().tolist()
    hits_itemcf, hits_hybrid = [], []

    for u in tqdm(users, desc="Evaluando usuarios"):
        # Recs ItemCF excluyendo vistos en TRAIN
        recs_itemcf = model.recommend_for_user(u, train, n=cfg.K * 5)  # un poco más para no recortar tras boost
        top_itemcf = recs_itemcf["product_id"].astype(str).head(cfg.K)

        # Recs híbridas (reponderadas)
        recs_hybrid = booster.boost(recs_itemcf, ItemCFHybridParams(beta=cfg.beta, min_reviews_for_sent=cfg.min_reviews_for_sent))
        top_hybrid = recs_hybrid["product_id"].astype(str).head(cfg.K)

        # Relevante (último item del usuario)
        rel = set(test.loc[test["user_id"].astype(str) == u, "product_id"].astype(str).tolist())

        hits_itemcf.append(hitrate_at_k(top_itemcf, rel, cfg.K))
        hits_hybrid.append(hitrate_at_k(top_hybrid, rel, cfg.K))

    hr_itemcf = float(np.mean(hits_itemcf)) if hits_itemcf else 0.0
    hr_hybrid = float(np.mean(hits_hybrid)) if hits_hybrid else 0.0

    print("\n=== Leave-last-1 (HitRate@K) ===")
    print(f"Usuarios evaluados: {len(users)} | K={cfg.K}")
    print(f"ItemCF puro:        {hr_itemcf:.4f}")
    print(f"Híbrido ItemCF+S:   {hr_hybrid:.4f}   (beta={cfg.beta}, min_reviews_for_sent={cfg.min_reviews_for_sent})")


if __name__ == "__main__":
    main()
