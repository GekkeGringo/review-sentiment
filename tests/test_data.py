import pandas as pd

from review_sentiment.data import split_by_movie


def test_split_has_no_movie_leakage():
    rows = [
        {"movie_name": f"film{i}", "grade3": "Good", "content": f"отзыв {i}-{j}"}
        for i in range(20)
        for j in range(5)
    ]
    df = pd.DataFrame(rows)
    train, val, test = split_by_movie(df)

    assert len(train) + len(val) + len(test) == len(df)
    movies = [set(p["movie_name"]) for p in (train, val, test)]
    assert not (movies[0] & movies[1])
    assert not (movies[0] & movies[2])
    assert not (movies[1] & movies[2])