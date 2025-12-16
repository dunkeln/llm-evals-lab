# TODO:
# - [ ] embedding similarity
# - [ ] semantic similarity metrics
# - [ ] llm as judge
import re
from typing import Counter
from sentence_transformers import SentenceTransformer, util
from torch import norm

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    
    emb_pred = EMBED_MODEL.encode(pred, convert_to_tensor=True)
    emb_ref  = EMBED_MODEL.encode( ref, convert_to_tensor=True)
    similarity = util.cos_sim(emb_pred, emb_ref)
    return float((similarity + 1.0) / 2.0)


def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def squad(pred: str, ref: str) -> float:
    pred_tokens, ref_tokens = normalize_text(pred).split(), normalize_text(ref).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    print(pred_tokens, ref_tokens)
    pred_counts, ref_counts = Counter(pred_tokens), Counter(ref_tokens)
    print(pred_counts, ref_counts)
    common = pred_counts & ref_counts
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def get_metrics(x, y):
    return {
        'cos_sim': cosine_similarity(x, y),
    }


if __name__ == "__main__":
    # print(get_metrics("I am batman", "who is the batman?"))
    print(get_metrics("this is 11", "11 is the answer"))
