# src/eval/evaluate_baseline_hybrid.py
"""
Evaluación temporal global (HitRate@K) para Baseline vs Híbrido.

Procedimiento:
- Split global por tiempo (cutoff p. ej. percentil 80 de review_date).
- Entrenar Baseline (popularidad Bayes) en TRAIN global.
- Construir Híbrido con sentimiento agregado (TRAIN global).
- Para cada usuario presente en TEST:
    * Excluir items vistos por ese usuario en TRAIN.
    * Medir HitRate@K: 1 si algún item de TEST del usuario aparece en Top-K, si no 0.
- Promediamos HitRate@K sobre usuarios.

Parámetros útiles:
- K (tamaño del ranking evaluado).
- min_user_interactions (para considerar usuarios con historial suficiente).
- min_item_freq (frecuencia mínima por item en TRAIN para estar en el catálogo).
- alpha y min_reviews_for_sent para el híbrido.

Requisitos: pandas, numpy, pyarrow, tqdm
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.baseline import PopularityRecommender
from src.models.data_loader import load_ratings
from src.models.hybrid import (
    HybridParams,
    combine_popularity_and_sentiment,
    load_product_sentiment,
)


@dataclass(frozen=True)
class EvalConfig:
    K: int = 10
    min_user_interactions: int = 3  # mínimo de interacciones por usuario en TRAIN
    min_item_freq: int = 5  # mínimo de interacciones por item en TRAIN para entrar al catálogo
    cutoff_quantile: float = 0.80  # percentil temporal para dividir TRAIN/TEST
    alpha: float = 0.7  # peso popularidad en híbrido
    min_reviews_for_sent: int = 3  # cobertura mínima de sentimiento por producto


def temporal_split_global(df: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal global: TRAIN = <= q-quantile(review_date), TEST = > q-quantile.

    Si no hay review_date válida, usa un split 80/20 por orden.
    """
    if "review_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["review_date"]):
        cutoff = df["review_date"].quantile(q)
        train = df[df["review_date"] <= cutoff].copy()
        test = df[df["review_date"] > cutoff].copy()
    else:
        # Fallback: 80/20 por índice
        n = len(df)
        idx = int(n * q)
        train = df.iloc[:idx].copy()
        test = df.iloc[idx:].copy()
    return train, test


def hitrate_at_k(recommended: pd.Series, relevant: set, K: int) -> float:
    """
    HitRate@K: 1 si recommended[:K] intersecta 'relevant', de lo contrario 0.
    """
    topk = set(recommended.head(K).tolist())
    return 1.0 if len(topk.intersection(relevant)) > 0 else 0.0


def main():
    cfg = EvalConfig()

    # 1) Cargar sample y tipar
    df = load_ratings(sample=True).copy()
    if "review_date" in df.columns:
        df = df.sort_values("review_date", kind="mergesort")

    # 2) Split global temporal
    train, test = temporal_split_global(df, cfg.cutoff_quantile)

    # 3) Filtrado de usuarios e items en TRAIN
    ucount = train["user_id"].value_counts()
    valid_users = set(ucount[ucount >= cfg.min_user_interactions].index.astype(str))

    icount = train["product_id"].value_counts()
    valid_items = set(icount[icount >= cfg.min_item_freq].index.astype(str))

    train = train[train["user_id"].isin(valid_users) & train["product_id"].isin(valid_items)].copy()
    test = test[test["user_id"].isin(valid_users)].copy()

    if train.empty or test.empty:
        print("Split temporal degenerado: ajusta cutoff_quantile o filtros.")
        return

    # 4) Entrenar popularidad en TRAIN global
    pop_model = PopularityRecommender(m=None, topk=cfg.K).fit(train)
    pop_scores = pop_model.item_scores  # index=product_id (v, R, score)
    # Limitar catálogo a valid_items (coherencia con TRAIN)
    pop_scores = pop_scores[pop_scores.index.isin(valid_items)]

    # 5) Sentimiento por producto (agregado, precomputado)
    prod_sent = load_product_sentiment()

    # 6) Híbrido
    params = HybridParams(alpha=cfg.alpha, min_reviews_for_sent=cfg.min_reviews_for_sent)
    hybrid_df = combine_popularity_and_sentiment(pop_scores, prod_sent, params)

    # 7) Evaluación usuario a usuario (HitRate@K)
    users = test["user_id"].astype(str).unique().tolist()
    hits_pop, hits_hyb, total_users = [], [], 0

    # Precomputar ranking global (product_id ordenado)
    pop_global_rank = (
        pop_scores.sort_values("score", ascending=False).reset_index()["product_id"].astype(str)
    )
    hyb_global_rank = hybrid_df.sort_values("hybrid_score", ascending=False)["product_id"].astype(
        str
    )

    for u in tqdm(users, desc="Evaluando usuarios"):
        seen_train = set(train.loc[train["user_id"] == u, "product_id"].astype(str))
        # Recomendaciones filtradas por usuario
        pop_rec = pop_global_rank[~pop_global_rank.isin(seen_train)]
        hyb_rec = hyb_global_rank[~hyb_global_rank.isin(seen_train)]

        # Relevantes del TEST (items que el usuario tocó en TEST y están en el catálogo)
        rel = set(test.loc[test["user_id"] == u, "product_id"].astype(str))
        rel = rel.intersection(set(valid_items))  # opcional: evalua solo items del catálogo

        if len(rel) == 0:
            continue

        h_pop = hitrate_at_k(pop_rec, rel, cfg.K)
        h_hyb = hitrate_at_k(hyb_rec, rel, cfg.K)

        hits_pop.append(h_pop)
        hits_hyb.append(h_hyb)
        total_users += 1

    pop_mean = float(np.mean(hits_pop)) if hits_pop else 0.0
    hyb_mean = float(np.mean(hits_hyb)) if hits_hyb else 0.0

    print("\n=== Resultados (HitRate@K) — Split temporal global ===")
    print(f"Usuarios evaluados: {total_users}")
    print(f"K = {cfg.K}")
    print(f"Baseline (popularidad): {pop_mean:.4f}")
    print(
        f"Híbrido (alpha={cfg.alpha}, min_reviews_for_sent={cfg.min_reviews_for_sent}): {hyb_mean:.4f}"
    )
    if total_users == 0:
        print(
            "Advertencia: 0 usuarios evaluados. Ajusta filtros (min_user_interactions/min_item_freq) o cutoff_quantile."
        )


if __name__ == "__main__":
    main()
