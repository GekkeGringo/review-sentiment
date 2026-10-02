import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def load_raw() -> pd.DataFrame:
    ds = load_dataset("blinoff/kinopoisk", split="train")
    return ds.to_pandas()


def _group_split(df: pd.DataFrame, test_size: float, group_col: str):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    a_idx, b_idx = next(gss.split(df, groups=df[group_col]))
    return df.iloc[a_idx], df.iloc[b_idx]


def split_by_movie(df: pd.DataFrame, group_col: str = "movie_name"):
    train_val, test = _group_split(df, 0.2, group_col)
    train, val = _group_split(train_val, 0.125, group_col)
    return train, val, test


if __name__ == "__main__":
    df = load_raw()
    print("Колонки:", df.columns.tolist())
    print("Всего отзывов:", len(df))
    print(df["grade3"].value_counts())
    words = df["content"].str.split().str.len()
    print("Длина отзыва в словах: медиана", int(words.median()),
          ", 95-й перцентиль", int(words.quantile(0.95)))

    train, val, test = split_by_movie(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(f"data/{name}.parquet")
        shares = part["grade3"].value_counts(normalize=True).round(3).to_dict()
        print(name, len(part), shares)