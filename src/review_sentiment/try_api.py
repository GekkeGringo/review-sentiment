import httpx
import pandas as pd

val = pd.read_parquet("data/val.parquet")
sample = val.groupby("grade3").sample(5, random_state=1)  # по 5 случайных отзывов каждого класса

for _, row in sample.iterrows():
    r = httpx.post(
        "http://127.0.0.1:8000/predict", json={"text": row["content"]}, timeout=60
    ).json()
    print(f"истина={row['grade3']:8} модель={r['label']:8} {r['probabilities']}")
    