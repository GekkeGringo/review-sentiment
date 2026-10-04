def truncate(ids, max_len, strategy, head_frac=0.25):
    body = max_len - 2  # место под [CLS] и [SEP]
    if len(ids) <= body:
        return ids
    if strategy == "head":
        return ids[:body]
    head = int(body * head_frac)
    return ids[:head] + ids[-(body - head):]


def encode_text(text, tok, max_len, strategy):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return [tok.cls_token_id] + truncate(ids, max_len, strategy) + [tok.sep_token_id]