import time

import numpy as np
import pandas as pd

from review_sentiment.predict import Predictor

p = Predictor()
val = pd.read_parquet("data/val.parquet").sample(100, random_state=42)
texts = val["content"].tolist()
p.predict(texts[0])  # прогрев

times = []
for t in texts:
    start = time.perf_counter()
    p.predict(t)
    times.append(time.perf_counter() - start)
print(
    f"ruBERT-256, CPU, по одному отзыву: медиана {np.median(times):.2f} с, "
    f"95-й перцентиль {np.percentile(times, 95):.2f} с"
)