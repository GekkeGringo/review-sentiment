import pytest
from fastapi.testclient import TestClient

from review_sentiment import api
from review_sentiment.db import Store


class FakePredictor:
    def predict(self, text):
        if "плохо" in text:
            return {"Bad": 0.8, "Neutral": 0.15, "Good": 0.05}
        return {"Bad": 0.1, "Neutral": 0.2, "Good": 0.7}


@pytest.fixture
def client(monkeypatch, tmp_path):
    db_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setattr(api, "create_predictor", lambda: FakePredictor())
    monkeypatch.setattr(api, "create_store", lambda: Store(db_url))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_long_text(client):
    r = client.post("/predict", json={"text": "слово " * 30})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "Good"
    assert set(body["probabilities"]) == {"Bad", "Neutral", "Good"}
    assert body["warning"] is None


def test_predict_short_text_has_warning(client):
    r = client.post("/predict", json={"text": "Скучно."})
    assert r.status_code == 200
    assert r.json()["warning"] is not None


@pytest.mark.parametrize("payload", [{"text": ""}, {}, {"text": "а" * 20001}])
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predictions_are_stored_and_aggregated(client):
    for _ in range(2):
        client.post("/predict", json={"text": "хорошо " * 30})
        client.post("/predict", json={"text": "плохо " * 30})

    daily = client.get("/stats/daily").json()
    assert len(daily) == 1
    assert daily[0]["total"] == 4
    assert daily[0]["bad_share"] == 0.5

    worst = client.get("/stats/negative", params={"limit": 2}).json()
    assert len(worst) == 2
    assert all(item["p_bad"] == 0.8 for item in worst)