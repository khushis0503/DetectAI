"""
saliency_check.py - Are our word highlights faithful? (a "deletion test")

If an explanation is good, deleting the words it marks as most important should change the
prediction MORE than deleting the same number of random words.

For test texts the model classifies correctly we:
  1. compute word importance (Gradient x Input for BERT, coefficient x TF-IDF for the baseline)
  2. delete the 5 words that pushed hardest TOWARDS the predicted class -> measure the drop
  3. delete 5 random words instead                                     -> measure the drop
The drop is measured in log-odds, log(p / (1 - p)), because BERT's raw probabilities are
saturated near 0 or 1 where even big changes in evidence barely move the probability.
Output: results/saliency_faithfulness.json
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.detector import Detector  # noqa: E402

N_PER_CLASS = 75
K = 5
SEED = 42


def delete_spans(text, spans):
    """Remove the given (start, end) character spans from the text."""
    out, pos = [], 0
    for s, e in sorted(spans):
        out.append(text[pos:s])
        pos = e
    out.append(text[pos:])
    return " ".join("".join(out).split())


def log_odds(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def run(det, texts, labels, word_scores_fn):
    rng = random.Random(SEED)
    drops_top, drops_rand = [], []
    p_orig = det.main_proba(texts)
    edited_top, edited_rand, keep = [], [], []
    for i, (text, y) in enumerate(zip(texts, labels)):
        if (p_orig[i] >= 0.5) != bool(y):        # only correctly classified texts
            continue
        scores = [(s, e, v) for s, e, v in word_scores_fn(text) if text[s:e].strip(".,;:!?\"'()").isalpha()]
        if len(scores) < 3 * K:
            continue
        sign = 1 if y == 1 else -1               # "towards the predicted class"
        top = sorted(scores, key=lambda t: -sign * t[2])[:K]
        rand = rng.sample(scores, K)
        edited_top.append(delete_spans(text, [(s, e) for s, e, _ in top]))
        edited_rand.append(delete_spans(text, [(s, e) for s, e, _ in rand]))
        keep.append(i)
    p_top, p_rand = det.main_proba(edited_top), det.main_proba(edited_rand)
    for j, i in enumerate(keep):
        sign = 1 if labels[i] == 1 else -1       # drop in evidence for the TRUE (= predicted) class
        drops_top.append(sign * (log_odds(p_orig[i]) - log_odds(p_top[j])))
        drops_rand.append(sign * (log_odds(p_orig[i]) - log_odds(p_rand[j])))
    drops_top, drops_rand = np.array(drops_top), np.array(drops_rand)
    return {
        "n_texts": len(keep),
        "mean_logodds_drop_top5_words": round(float(drops_top.mean()), 4),
        "mean_logodds_drop_random5_words": round(float(drops_rand.mean()), 4),
        "share_where_top5_beats_random": round(float(np.mean(drops_top > drops_rand)), 4),
    }


def main():
    df = pd.read_csv(ROOT / "data" / "detectai_dataset.csv")
    test = df[(df["split"] == "test") & (df["n_words"] >= 60)]
    sample = pd.concat([test[test["label"] == c].sample(N_PER_CLASS, random_state=SEED) for c in [0, 1]])
    texts, labels = sample["text"].tolist(), sample["label"].values

    det = Detector()
    if det.kind == "bert":
        def scores(text):          # whole text fits in one BERT input for these lengths
            return det._bert_word_scores(text)
        method = "Gradient x Input (BERT)"
    else:
        scores = det._tfidf_word_scores
        method = "LR coefficient x TF-IDF"

    result = {"model": det.model_name, "method": method, "k_words_deleted": K, **run(det, texts, labels, scores)}
    print(json.dumps(result, indent=2))
    out = ROOT / "results" / "saliency_faithfulness.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing[det.kind] = result
    out.write_text(json.dumps(existing, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
