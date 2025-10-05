# tests/test_api_smoke.py
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"


def test_recommend_global_ok():
    r = client.get("/recommend/global?n=3")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        for k in ["product_id", "product_title", "score"]:
            assert k in data[0]


def test_recommend_itemcf_user_ok():
    # usuario dummy — la API debería responder 200 con una lista (puede estar vacía si no hay señal)
    r = client.get("/recommend/itemcf/user/TEST_USER?n=3")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
