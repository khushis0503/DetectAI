"""
robustness_test.py - How well do the models survive light human editing of AI text?

We take the AI texts in the TEST split, apply the same light edits used for augmentation
(synonym replacement, sentence reordering, filler removal), and compare each saved
baseline model's accuracy on:
    original test set  vs  test set where the AI texts were edited
Human texts are left unchanged, so any drop comes from edited AI text slipping through.

Outputs: results/robustness.json, results/figures/robustness_accuracy_drop.png
"""

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.preprocessing import load_stylo_cache, save_stylo_cache  # noqa: E402
from augment import edit_text  # noqa: E402  (same edits as training augmentation)
from plot_style import AI_COLOR, HUMAN_COLOR, MUTED, plt  # noqa: E402
from train_baselines import SAVE_AS  # noqa: E402

MODELS = ROOT / "backend" / "models"
RESULTS = ROOT / "results"
STYLO_CACHE = ROOT / "data" / "stylometric_cache.joblib"
SEED = 7   # different seed from augmentation, so test edits are not identical to training edits


def main():
    load_stylo_cache(STYLO_CACHE)
    df = pd.read_csv(ROOT / "data" / "detectai_dataset.csv")
    test = df[df["split"] == "test"].reset_index(drop=True)
    y = test["label"].values

    # Build the edited version of the test set (edit AI texts only)
    rng = random.Random(SEED)
    edited = test["text"].copy()
    ai_rows = test["label"] == 1
    edited[ai_rows] = test.loc[ai_rows, "text"].apply(lambda t: edit_text(t, rng))

    results = {}
    for name, filename in SAVE_AS.items():
        model = joblib.load(MODELS / filename)
        pred_orig = model.predict(test["text"].tolist())
        pred_edit = model.predict(edited.tolist())
        acc_o, acc_e = accuracy_score(y, pred_orig), accuracy_score(y, pred_edit)
        rec_o, rec_e = recall_score(y, pred_orig), recall_score(y, pred_edit)
        results[name] = {
            "accuracy_original": round(acc_o, 4),
            "accuracy_edited": round(acc_e, 4),
            "accuracy_drop": round(acc_o - acc_e, 4),
            "ai_recall_original": round(rec_o, 4),   # share of AI texts caught
            "ai_recall_edited": round(rec_e, 4),
        }
        r = results[name]
        print(f"{name:34s} acc {r['accuracy_original']:.4f} -> {r['accuracy_edited']:.4f}  "
              f"(drop {r['accuracy_drop']:+.4f}) | AI recall {rec_o:.4f} -> {rec_e:.4f}")

    out = {"n_test": len(test), "n_ai_texts_edited": int(ai_rows.sum()),
           "edits": ["WordNet synonym replacement (~15% of content words)",
                     "sentence reordering", "filler-word removal"],
           "models": results}
    save_stylo_cache(STYLO_CACHE)
    (RESULTS / "robustness.json").write_text(json.dumps(out, indent=2))

    # Bar chart: original vs edited accuracy per model
    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - 0.2, [results[n]["accuracy_original"] for n in names], width=0.38,
                color=HUMAN_COLOR, label="Original test set")
    b2 = ax.bar(x + 0.2, [results[n]["accuracy_edited"] for n in names], width=0.38,
                color=AI_COLOR, label="AI texts lightly edited")
    ax.bar_label(b1, fmt="%.3f", fontsize=7, color=MUTED)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, color=MUTED)
    ax.set_xticks(x, [n.replace(" (", "\n(").replace(" + ", " +\n", 1) for n in names], fontsize=8)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness: accuracy before and after light editing of AI texts")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(RESULTS / "figures" / "robustness_accuracy_drop.png")
    plt.close(fig)
    print(f"\nSaved {RESULTS / 'robustness.json'} and figure")


if __name__ == "__main__":
    main()
