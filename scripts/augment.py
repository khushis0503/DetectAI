"""
augment.py - Makes lightly edited copies of AI texts (TRAIN split only).

Why: people often edit ChatGPT output a little before using it. If the model only ever
sees untouched AI text, small edits can fool it. Adding edited copies to training
teaches it to look at deeper patterns.

Edits (functions are reused by robustness_test.py):
  synonym_replace  - swap ~15% of content words for a WordNet synonym
  reorder_sentences- shuffle the order of sentences
  remove_fillers   - delete a few filler words ("very", "really", "additionally", ...)

Val and test splits are NEVER touched.
Output: data/train_augmented.csv (same columns as the dataset, split = "train_aug")
"""

import random
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "detectai_dataset.csv"
OUT = ROOT / "data" / "train_augmented.csv"

STOPWORDS = set(stopwords.words("english"))
FILLERS = {"very", "really", "just", "quite", "actually", "basically", "truly",
           "additionally", "furthermore", "moreover", "overall", "indeed", "simply",
           "certainly", "ultimately", "however,"}


def synonym_replace(text, rng, rate=0.15):
    """Replace about `rate` of the non-stopwords with a random WordNet synonym."""
    words = text.split()
    for i, word in enumerate(words):
        core = word.strip(".,;:!?\"'()")
        if not core.isalpha() or core.lower() in STOPWORDS or rng.random() > rate:
            continue
        synonyms = {l.name().replace("_", " ") for s in wordnet.synsets(core.lower())
                    for l in s.lemmas()} - {core.lower()}
        synonyms = [s for s in synonyms if s.isalpha()]       # single plain words only
        if synonyms:
            new = rng.choice(sorted(synonyms))
            if core[0].isupper():
                new = new.capitalize()
            words[i] = word.replace(core, new)
    return " ".join(words)


def reorder_sentences(text, rng):
    """Shuffle sentence order (keeps every sentence, changes the flow)."""
    sents = sent_tokenize(text)
    rng.shuffle(sents)
    return " ".join(sents)


def remove_fillers(text, rng, max_removed=4):
    """Delete up to `max_removed` filler words."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if w.lower().strip(".,;:!?") in FILLERS]
    drop = set(rng.sample(idx, min(len(idx), max_removed)))
    return " ".join(w for i, w in enumerate(words) if i not in drop)


def edit_text(text, rng):
    """Apply all three light edits (used for both augmentation and robustness testing)."""
    text = synonym_replace(text, rng)
    text = reorder_sentences(text, rng)
    text = remove_fillers(text, rng)
    return text


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(DATA)
    ai_train = df[(df["split"] == "train") & (df["label"] == 1)].copy()

    ai_train["text"] = ai_train["text"].apply(lambda t: edit_text(t, rng))
    ai_train["n_words"] = ai_train["text"].str.split().str.len()
    ai_train["split"] = "train_aug"
    ai_train["gen_style"] = ai_train["gen_style"] + "_edited"

    ai_train.to_csv(OUT, index=False)
    print(f"Created {len(ai_train)} edited AI copies from the TRAIN split -> {OUT}")
    print("\nExample (before -> after):")
    orig = df[(df["split"] == "train") & (df["label"] == 1)].iloc[0]["text"]
    print("BEFORE:", orig[:300])
    print("AFTER :", ai_train.iloc[0]["text"][:300])


if __name__ == "__main__":
    main()
