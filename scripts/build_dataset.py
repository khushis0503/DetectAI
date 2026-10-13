"""
build_dataset.py - Builds the DetectAI dataset from two public corpora.

Input : raw/            git clone of https://github.com/vivek3141/ghostbuster-data
                        (formal essays, creative stories "wp", Reuters news)
        raw/hc3/all.jsonl  HC3 (Hello-SimpleAI/HC3 on Hugging Face): questions answered by
                        humans and by ChatGPT - casual Q&A text, closer to live demo input
Output: data/detectai_dataset.csv  with columns
        text, label (0 = human, 1 = AI), source, domain, gen_style, n_words, split

Steps
1. For each domain (essay, wp, reuter) read ~1000 human texts and ~1000 AI texts.
   The AI texts are mixed evenly from 5 ChatGPT prompting styles.
2. Skip every file inside a "logprobs*" folder (those are token probabilities, not text)
   and read only the human folder and the five ChatGPT folders (all other folders are ignored).
3. Cut each text to a random length of 20-300 words, ending at a sentence boundary,
   so that text length cannot be used as a shortcut by the classifiers.
4. Remove texts shorter than 20 or longer than 300 words, and exact duplicates.
5. HC3: undo its tokenisation artefacts (e.g. "you 're", "word ." in human answers,
   literal backslash-n in ChatGPT answers) so formatting cannot become a shortcut, then take a
   balanced sample of human and ChatGPT answers from its 5 subsets.
6. Add a fixed, stratified 70/15/15 train/val/test split (seed 42) over everything.
"""

import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = ROOT / "data" / "detectai_dataset.csv"

DOMAINS = ["essay", "wp", "reuter"]
AI_FOLDERS = ["gpt", "gpt_prompt1", "gpt_prompt2", "gpt_writing", "gpt_semantic"]
PER_CLASS = 1000  # texts per class per domain

HC3_FILE = RAW / "hc3" / "all.jsonl"
HC3_SUBSETS = ["reddit_eli5", "open_qa", "wiki_csai", "finance", "medicine"]
HC3_PER_SUBSET = 300  # per class per subset -> ~1500 human + ~1500 ChatGPT

sys.path.insert(0, str(ROOT))
from backend.preprocessing import clean_text, strip_markdown  # noqa: E402

rng = random.Random(SEED)


def list_text_files(folder):
    """All .txt files under a folder, ignoring 'logprobs*' subfolders (e.g. logprobs_babbage)
    and notebook checkpoint folders."""
    files = [p for p in folder.rglob("*.txt")
             if not any(part.startswith(("logprobs", ".ipynb")) for part in p.parts)]
    return sorted(files)


TOKEN_LINE_RE = re.compile(r"^\S*\s+-?\d+(\.\d+)?(e-?\d+)?$")


def is_token_dump(text):
    """True if a file is a list of 'token  probability' lines (stray logprob file), not prose."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    token_lines = sum(1 for l in lines if TOKEN_LINE_RE.match(l.strip()))
    return token_lines / len(lines) > 0.5 or "Ġ" in text   # 'Ġ' = GPT tokenizer space marker


def normalise_quotes(text):
    """Curly quotes -> straight quotes. Human sources use curly quotes far more often,
    which is a formatting artefact, not writing style, so we remove that shortcut."""
    return (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"'))


def read_text(path):
    return normalise_quotes(path.read_text(encoding="utf-8", errors="ignore").strip())


def cut_to_random_length(text, min_words=20, max_words=300, target=None):
    """Keep whole sentences until we reach a random target length (20-300 words).
    We stop at the sentence boundary closest to the target (under or over)."""
    if target is None:
        target = rng.randint(min_words, max_words)
    kept, n_words = [], 0
    for sent in sent_tokenize(text):
        sent_words = len(sent.split())
        if n_words >= min_words:
            # stop if adding this sentence takes us further from the target than we are now,
            # or past the 300-word cap (we always keep adding while below the 20-word minimum)
            if abs(n_words + sent_words - target) > abs(n_words - target) or \
                    n_words + sent_words > max_words:
                break
        kept.append(sent)
        n_words += sent_words
    return " ".join(kept)


skipped_dumps = 0


def sample_texts(folder, n):
    """Read up to n random prose texts from a folder, skipping token-dump files."""
    global skipped_dumps
    files = list_text_files(folder)
    rng.shuffle(files)
    texts = []
    for f in files:
        if len(texts) == n:
            break
        text = read_text(f)
        if is_token_dump(text):
            skipped_dumps += 1
            continue
        texts.append(text)
    return texts


def collect_domain(domain):
    rows = []

    # Human texts (label 0)
    for text in sample_texts(RAW / domain / "human", PER_CLASS):
        rows.append({"raw": text, "label": 0, "source": "human",
                     "domain": domain, "gen_style": "human"})

    # AI texts (label 1): equal share from each ChatGPT prompting style
    per_style = PER_CLASS // len(AI_FOLDERS)
    for style in AI_FOLDERS:
        for text in sample_texts(RAW / domain / style, per_style):
            rows.append({"raw": text, "label": 1, "source": "ChatGPT",
                         "domain": domain, "gen_style": style})
    return rows


# ----------------------------------------------------------------------------
# HC3 (question-answer data)
# ----------------------------------------------------------------------------
def fix_hc3_spacing(text):
    """Undo HC3's word-tokenised spacing, e.g.  "you 're" -> "you're",  "end ." -> "end.",
    "( i.e. a PA )" -> "(i.e. a PA)",  "cable / dsl" -> "cable/dsl".
    Applied to BOTH classes so the two are formatted the same way."""
    text = text.replace("\\n", " ")                                   # literal backslash-n
    text = re.sub(r"\{\\displaystyle[^}]*\}+", " ", text)              # Wikipedia maths markup
    text = re.sub(r" (n't|'s|'re|'m|'ll|'ve|'d)\b", r"\1", text, flags=re.I)
    text = re.sub(r" ([.,!?;:%)\]])", r"\1", text)                     # no space before punctuation
    text = re.sub(r"([(\[$]) ", r"\1", text)                           # no space after ( [ $
    text = re.sub(r"(\w) ([-/]) (\w)", r"\1\2\3", text)                # oscar - winning -> oscar-winning
    text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)                  # " Best Seller " -> "Best Seller"
    text = re.sub(r"\.(?=[A-Z][a-z])", ". ", text)                     # "Hi.The" -> "Hi. The"
    text = re.sub(r",(?=[A-Za-z])", ", ", text)                        # "Hi,The" -> "Hi, The"
    return text


def collect_hc3():
    """Balanced sample: HC3_PER_SUBSET human + HC3_PER_SUBSET ChatGPT answers per subset."""
    pools = {}  # (subset, label) -> list of cleaned answers
    with open(HC3_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for label, key in [(0, "human_answers"), (1, "chatgpt_answers")]:
                for ans in row[key] or []:
                    text = clean_text(normalise_quotes(fix_hc3_spacing(ans)), remove_markdown=False)
                    if not is_token_dump(text) and len(text.split()) >= 20:
                        pools.setdefault((row["source"], label), []).append(text)

    rows = []
    for subset in HC3_SUBSETS:
        for label, source in [(0, "human"), (1, "ChatGPT")]:
            pool = sorted(set(pools.get((subset, label), [])))
            rng.shuffle(pool)
            for text in pool[:HC3_PER_SUBSET]:
                rows.append({"raw": text, "label": label, "source": source,
                             "domain": "Q&A (HC3)",
                             "gen_style": f"hc3_{subset}" + ("" if label == 0 else "_chatgpt")})
    return rows


def main():
    rows = []
    for domain in DOMAINS:
        rows.extend(collect_domain(domain))
    n_ghost = len(rows)
    rows.extend(collect_hc3())
    df = pd.DataFrame(rows)
    print(f"Raw texts read: {n_ghost} Ghostbuster + {len(df) - n_ghost} HC3 "
          f"(skipped {skipped_dumps} token-probability dump files)")

    # Same light cleaning for every text, then cut to a random length at a sentence boundary
    df["raw"] = df["raw"].apply(lambda t: clean_text(t, remove_markdown=False))
    df["text"] = df["raw"].apply(cut_to_random_length)
    df["n_words"] = df["text"].str.split().str.len()

    # HC3 human answers are naturally short (people answer briefly), ChatGPT ones long.
    # To stop length being a shortcut, cut each HC3 ChatGPT answer to a length drawn
    # from the HC3 human answer lengths, so both classes have the same distribution.
    hc3_human_lengths = df.loc[(df["domain"] == "Q&A (HC3)") & (df["label"] == 0), "n_words"].tolist()
    hc3_ai = (df["domain"] == "Q&A (HC3)") & (df["label"] == 1)
    df.loc[hc3_ai, "text"] = df.loc[hc3_ai, "raw"].apply(
        lambda t: cut_to_random_length(t, target=rng.choice(hc3_human_lengths)))
    df["n_words"] = df["text"].str.split().str.len()

    # Drop very short texts, over-long ones (a "sentence" with no full stop can overshoot
    # the 300-word cap), and duplicates
    before = len(df)
    df = df[(df["n_words"] >= 20) & (df["n_words"] <= 300)]
    df = df.drop_duplicates(subset="text")
    print(f"Removed {before - len(df)} too-short/too-long/duplicate texts")
    df = df.drop(columns="raw").reset_index(drop=True)

    # Fixed stratified split on label x domain: 70% train, 15% val, 15% test
    strata = df["label"].astype(str) + "_" + df["domain"]
    train_idx, temp_idx = train_test_split(df.index, test_size=0.30,
                                           stratify=strata, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50,
                                         stratify=strata[temp_idx], random_state=SEED)
    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Remove Reddit/markdown formatting (*emphasis*, (URL_0), "> quote"), which appears almost only
    # in human texts. Done AFTER filtering and splitting so every text keeps the same split as the
    # version our BERT model was trained on (the rows do not change, only their formatting).
    df["text"] = df["text"].apply(strip_markdown)
    df["n_words"] = df["text"].str.split().str.len()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label", "source", "domain", "gen_style", "n_words", "split"]].to_csv(OUT, index=False)

    print(f"\nSaved {len(df)} texts to {OUT}\n")
    print("Texts per domain and class:")
    print(pd.crosstab(df["domain"], df["source"], margins=True), "\n")
    print("Texts per split and class:")
    print(pd.crosstab(df["split"], df["source"], margins=True), "\n")
    print("Words per text (by domain and class):")
    print(df.groupby(["domain", "source"])["n_words"].describe().round(1))


if __name__ == "__main__":
    main()
