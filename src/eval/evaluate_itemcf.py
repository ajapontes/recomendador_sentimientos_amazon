# src/eval/evaluate_itemcf.py
"""
Evaluación leave-last-1 para Item-Item CF con HitRate@K.

Procedimiento:
1) Filtramos a usuarios con historial suficiente (>= min_user_interactions) y
   a ítems con frecuencia mínima (>= min_item_freq) para robustez.
2) Por usuario, ordenamos por fecha (si no es confiable, usamos orden estable)
   y dejamos la última interacción como TEST y el resto como TRAIN.
3) Entrenamos ItemCF con todo el TRAIN (global).
4) Para cada usuario, recomendamos Top-K excluyendo lo visto en TRAIN y medimos
   HitRate@K: 1 si su ítem de TEST aparece en el Top-K, de lo contrario 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.data_loader import load_ratings
from src.models.itemcf import ItemCFConfig, ItemCFRecommender


@dataclass(frozen=True)
class EvalConfig:
    K: int = 10
    min_rating_like: float = 3.0  # coherente con tu run_itemcf actual
    min_item_freq: int = 5
    min_user_interactions: int = 2
    n_neighbors: int = 800


def _leave_last_split_user(df_u: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, test) dejando la última fila como test."""
    if len(df_u) < 2:
        return df_u.iloc[:0], df_u
    return df_u.iloc[:-1], df_u.iloc[-1:]


def main():
    cfg = EvalConfig()

    # 1) Cargar sample y asegurar tipos
    df = load_ratings(sample=True).copy()
    # Orden temporal si está presente, si no orden estable por user_id
    if "review_date" in df.columns:
        df = df.sort_values(["user_id", "review_date"], kind="mergesort")
    else:
        df = df.sort_values(["user_id"], kind="mergesort")

    # 2) Filtrado de robustez previo a los splits
    likes = df[df["rating"].astype(float) >= cfg.min_rating_like]
    ucount = likes["user_id"].value_counts()
    valid_users = set(ucount[ucount >= cfg.min_user_interactions].index.astype(str))
    if not valid_users:
        print("No hay usuarios suficientes tras filtros; baja los umbrales.")
        return

    # Mantener solo usuarios válidos
    df = df[df["user_id"].astype(str).isin(valid_users)].copy()

    # 3) Split leave-last-1 por usuario → construir TRAIN global y TEST global
    trains, tests = [], []
    for _uid, df_u in df.groupby("user_id", sort=False):
        tr, te = _leave_last_split_user(df_u)
        if not tr.empty and not te.empty:
            trains.append(tr)
            tests.append(te)

    if not trains or not tests:
        print("No se pudo construir train/test; revisa filtros.")
        return

    train = pd.concat(trains, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)

    # Filtrar ítems con frecuencia mínima en TRAIN
    icount = train["product_id"].value_counts()
    valid_items = set(icount[icount >= cfg.min_item_freq].index.astype(str))
    train = train[train["product_id"].astype(str).isin(valid_items)].copy()
    test = test[test["product_id"].astype(str).isin(valid_items)].copy()

    if train.empty or test.empty:
        print("TRAIN o TEST vacío tras filtros; ajusta min_item_freq.")
        return

    # 4) Entrenar ItemCF en TRAIN global
    model = ItemCFRecommender(
        ItemCFConfig(
            min_rating_like=cfg.min_rating_like,
            min_item_freq=cfg.min_item_freq,
            min_user_interactions=cfg.min_user_interactions,
            n_neighbors=cfg.n_neighbors,
            topk_recommendations=cfg.K,
        )
    ).fit(train)

    # 5) Evaluación HitRate@K
    users = test["user_id"].astype(str).unique().tolist()
    hits = []
    for u in tqdm(users, desc="Evaluando usuarios"):
        # Recomendar excluyendo vistos en TRAIN
        recs = model.recommend_for_user(u, train, n=cfg.K)
        topk = set(recs["product_id"].astype(str).tolist())

        # Ítem de test del usuario (puede ser 1 después del filtrado)
        rel = set(test.loc[test["user_id"].astype(str) == u, "product_id"].astype(str).tolist())

        hit = 1.0 if len(topk.intersection(rel)) > 0 else 0.0
        hits.append(hit)

    hr = float(np.mean(hits)) if hits else 0.0
    print("\n=== ItemCF — Leave-last-1 (HitRate@K) ===")
    print(f"Usuarios evaluados: {len(hits)}")
    print(f"K = {cfg.K}")
    print(f"HitRate@{cfg.K}: {hr:.4f}")
    if not hits:
        print("No se evaluó ningún usuario; ajusta filtros y vuelve a ejecutar.")


if __name__ == "__main__":
    main()
