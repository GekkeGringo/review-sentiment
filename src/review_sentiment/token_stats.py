import pandas as pd
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
train = pd.read_parquet("data/train.parquet").sample(3000, random_state=42)
lengths = pd.Series(
    [len(x) for x in tok(train["content"].tolist(), add_special_tokens=True)["input_ids"]]
)
print("медиана:", int(lengths.median()))
print("95-й перцентиль:", int(lengths.quantile(0.95)))
print("доля длиннее 256:", round((lengths > 256).mean(), 3))
print("доля длиннее 512:", round((lengths > 512).mean(), 3))