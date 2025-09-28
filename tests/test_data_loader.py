# tests/test_data_loader.py
import pandas as pd

from src.models.data_loader import load_ratings


def test_load_ratings_sample_columns():
    df = load_ratings(sample=True)
    # columnas mínimas
    for col in ["user_id", "product_id", "rating"]:
        assert col in df.columns
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
