# tests/test_hybrids.py
# -*- coding: utf-8 -*-
import pandas as pd

from src.models.data_loader import load_ratings
from src.models.hybrid import load_product_sentiment, combine_popularity_and_sentiment, HybridParams
from src.models.baseline import PopularityRecommender
from src.models.itemcf import ItemCFRecommender, ItemCFConfig
from src.models.hybrid_itemcf import ItemCFSentimentBooster, ItemCFHybridParams


def test_popularity_sentiment_hybrid_shapes():
    df = load_ratings(sample=True)
    pop = PopularityRecommender(m=None, topk=10).fit(df)
    pop_scores = pop.item_scores  # index: product_id, cols: score,v,R

    prod_sent = load_product_sentiment()  # generado en 4.2
    params = HybridParams(alpha=0.7, min_reviews_for_sent=3)

    hybrid = combine_popularity_and_sentiment(pop_scores, prod_sent, params)
    assert isinstance(hybrid, pd.DataFrame)
    # columnas mínimas esperadas
    for col in ["product_id", "hybrid_score", "score", "sentiment_mean", "n_scored"]:
        assert col in hybrid.columns
    assert len(hybrid) > 0


def test_itemcf_with_sentiment_boost():
    df = load_ratings(sample=True)
    icf = ItemCFRecommender(ItemCFConfig(
        min_rating_like=3.0, min_item_freq=5, min_user_interactions=2,
        n_neighbors=150, topk_recommendations=10
    )).fit(df)

    # base recs (para un usuario)
    u = str(df["user_id"].astype(str).iloc[0])
    base_recs = icf.recommend_for_user(u, df, n=20)

    # booster
    prod_sent = load_product_sentiment()
    booster = ItemCFSentimentBooster()
    boosted = booster.boost(base_recs, ItemCFHybridParams(beta=0.5, min_reviews_for_sent=3))

    assert isinstance(boosted, pd.DataFrame)
    for col in ["product_id", "score", "score_hybrid"]:
        assert col in boosted.columns
    assert len(boosted) > 0
