import pytest
from fastapi.testclient import TestClient

from review_sentiment import api


class FakePredictor:
    def predict(self, text):
        return {"Bad": 0.1, "Neutral": 0.2, "Good": 0.7}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "create_predictor", lambda: FakePredictor())
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


@pytest.mark.parametrize(
    "payload", [{"text": ""}, {}, {"text": "а" * 20001}]
)
def test_invalid_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422