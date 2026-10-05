from review_sentiment.text import encode_text, truncate


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def test_short_text_is_unchanged():
    ids = list(range(10))
    assert truncate(ids, 256, "head") == ids
    assert truncate(ids, 256, "head_tail") == ids


def test_head_keeps_beginning():
    ids = list(range(1000))
    out = truncate(ids, 256, "head")
    assert out == ids[:254]


def test_head_tail_keeps_both_ends():
    ids = list(range(1000))
    out = truncate(ids, 256, "head_tail")
    assert len(out) == 254
    assert out[:63] == ids[:63]
    assert out[-1] == 999


def test_encode_adds_special_tokens_and_respects_limit():
    text = " ".join(["слово"] * 1000)
    ids = encode_text(text, FakeTokenizer(), 256, "head_tail")
    assert ids[0] == 101 and ids[-1] == 102
    assert len(ids) == 256