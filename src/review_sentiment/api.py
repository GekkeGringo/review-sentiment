from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from review_sentiment.db import Store

state = {}


def create_predictor():
    from review_sentiment.predict import Predictor

    return Predictor()


def create_store():
    return Store()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = create_predictor()
    state["store"] = create_store()
    yield
    state.clear()


app = FastAPI(title="Kinopoisk sentiment", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class PredictResponse(BaseModel):
    label: str
    probabilities: dict[str, float]
    warning: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    probs = state["predictor"].predict(req.text)
    label = max(probs, key=probs.get)
    state["store"].save(req.text, label, probs)
    warning = None
    if len(req.text.split()) < 20:
        warning = (
            "Короткий текст: модель обучена на развёрнутых рецензиях, "
            "надёжность ниже."
        )
    return PredictResponse(label=label, probabilities=probs, warning=warning)


@app.get("/stats/daily")
def stats_daily():
    return state["store"].daily_negative_share()


@app.get("/stats/negative")
def stats_negative(days: int = 7, limit: int = 10):
    return state["store"].most_negative(days=days, limit=limit)