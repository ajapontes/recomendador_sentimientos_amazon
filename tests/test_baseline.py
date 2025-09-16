# tests/test_baseline.py
# -*- coding: utf-8 -*-
from src.models.data_loader import load_ratings
from src.models.baseline import PopularityRecommender

def test_popularity_recommender_topn():
    df = load_ratings(sample=True)
    model = PopularityRecommender(m=None, topk=10).fit(df)
    recs = model.recommend_global(n=5)
    assert list(recs.columns) == ["product_id", "score", "v", "R"]
    assert len(recs) <= 5
