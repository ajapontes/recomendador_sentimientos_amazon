# tests/test_itemcf.py
# -*- coding: utf-8 -*-
import pandas as pd
from src.models.data_loader import load_ratings
from src.models.itemcf import ItemCFRecommender, ItemCFConfig


def test_itemcf_fit_and_recommend_shape():
    df = load_ratings(sample=True)
    cfg = ItemCFConfig(
        min_rating_like=3.0,
        min_item_freq=5,
        min_user_interactions=2,
        n_neighbors=150,
        topk_recommendations=10,
    )
    model = ItemCFRecommender(cfg).fit(df)

    # toma un usuario con interacciones
    u = str(df["user_id"].astype(str).iloc[0])
    recs = model.recommend_for_user(u, df, n=5)

    assert isinstance(recs, pd.DataFrame)
    assert len(recs) <= 5
    # columnas mínimas
    for col in ["product_id", "score", "neighbors_hits"]:
        assert col in recs.columns


def test_itemcf_fallback_on_cold_user():
    df = load_ratings(sample=True)
    cfg = ItemCFConfig(
        min_rating_like=3.0,
        min_item_freq=5,
        min_user_interactions=2,
        n_neighbors=150,
        topk_recommendations=10,
    )
    model = ItemCFRecommender(cfg).fit(df)

    # usuario inexistente → debe hacer fallback a popularidad interna
    recs = model.recommend_for_user("USER_NOT_IN_DATASET", df, n=5)
    assert len(recs) <= 5
    assert "product_id" in recs.columns
