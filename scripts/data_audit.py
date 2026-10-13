"""
data_audit.py - Measures the data-quality problems we found and how the clean-up changed them.

Every number in the report's "data quality" table comes from this script:
  1. token-probability files hidden among the AI essays (GPT tokenizer symbol "Ġ")
  2. curly apostrophes per 1,000 words, human vs AI (Ghostbuster)
  3. HC3 "space before punctuation" artefact (Reddit ELI5 answers), before vs after fix_hc3_spacing
  4. literal backslash-n in HC3 answers, by class
  5. Wikipedia maths markup ({\\displaystyle ...}) in HC3 answers, by class
  6. HC3 answer length by class, before vs after length matching
  7. Reddit markdown (*emphasis*, URL_n placeholders, > quotes), by class, in the final dataset
Output: results/data_audit.json
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_dataset import AI_FOLDERS, DOMAINS, fix_hc3_spacing, list_text_files  # noqa: E402
from backend.preprocessing import strip_markdown  # noqa: E402

RAW = ROOT / "raw"
SPACE_PUNCT = re.compile(r" [.,!?;:]")


def per_1k_chars(texts, pattern):
    t = " ".join(texts)
    return round(1000 * len(pattern.findall(t)) / max(len(t), 1), 3)


def main():
    audit = {}

    # 1. token-probability files among the AI essay folders (outside the plain "logprobs" folders)
    dumps = [p for p in (RAW / "essay").rglob("*.txt")
             if any(part.startswith("logprobs_") for part in p.parts)]
    audit["token_probability_files"] = {
        "files_in_logprobs_star_subfolders_of_essay": len(dumps),
        "containing_GPT_space_symbol": sum("Ġ" in p.read_text(encoding="utf-8", errors="ignore") for p in dumps),
    }

    # 2. curly apostrophes, all Ghostbuster files we sample from (human vs AI folders)
    def apostrophes(folder_names):
        n_curly = n_words = 0
        for d in DOMAINS:
            for f in folder_names:
                for p in list_text_files(RAW / d / f):
                    t = p.read_text(encoding="utf-8", errors="ignore")
                    n_curly += t.count("’") + t.count("‘")
                    n_words += len(t.split())
        return round(1000 * n_curly / max(n_words, 1), 3)
    audit["curly_apostrophes_per_1000_words"] = {"human": apostrophes(["human"]), "ai": apostrophes(AI_FOLDERS)}

    # 3-6. HC3 artefacts
    rows = [json.loads(line) for line in open(RAW / "hc3" / "all.jsonl", encoding="utf-8")]
    eli5 = [r for r in rows if r["source"] == "reddit_eli5"]
    h_raw = [a for r in eli5 for a in r["human_answers"]]
    c_raw = [a for r in eli5 for a in r["chatgpt_answers"]]
    audit["hc3_eli5_space_before_punctuation_per_1000_chars"] = {
        "human_before": per_1k_chars(h_raw, SPACE_PUNCT), "chatgpt_before": per_1k_chars(c_raw, SPACE_PUNCT),
        "human_after": per_1k_chars([fix_hc3_spacing(a) for a in h_raw], SPACE_PUNCT),
        "chatgpt_after": per_1k_chars([fix_hc3_spacing(a) for a in c_raw], SPACE_PUNCT),
    }
    all_h = [a for r in rows for a in r["human_answers"]]
    all_c = [a for r in rows for a in r["chatgpt_answers"]]
    audit["hc3_answers_with_literal_backslash_n"] = {
        "human": sum("\\n" in a for a in all_h), "chatgpt": sum("\\n" in a for a in all_c),
        "n_human": len(all_h), "n_chatgpt": len(all_c)}
    audit["hc3_answers_with_displaystyle_markup"] = {
        "human": sum("displaystyle" in a for a in all_h), "chatgpt": sum("displaystyle" in a for a in all_c)}

    df = pd.read_csv(ROOT / "data" / "detectai_dataset.csv")
    hc3 = df[df["domain"] == "Q&A (HC3)"]
    audit["hc3_words_per_answer_in_final_dataset"] = {
        "human_median": float(hc3.loc[hc3.label == 0, "n_words"].median()),
        "chatgpt_median": float(hc3.loc[hc3.label == 1, "n_words"].median()),
        "human_mean": round(float(hc3.loc[hc3.label == 0, "n_words"].mean()), 1),
        "chatgpt_mean": round(float(hc3.loc[hc3.label == 1, "n_words"].mean()), 1)}
    audit["hc3_raw_answer_words_median"] = {
        "human": float(np.median([len(a.split()) for a in all_h])),
        "chatgpt": float(np.median([len(a.split()) for a in all_c]))}

    # 7. markdown: how many final texts would still change if markdown were not removed?
    #    (measured on the dataset version saved just before the markdown fix, if available)
    before = ROOT / "results" / "dataset_before_markdown_fix.csv"
    if before.exists():
        b = pd.read_csv(before)
        changed = b["text"] != b["text"].apply(strip_markdown)
        audit["texts_with_reddit_markdown_before_fix"] = {
            "human": int(changed[b.label == 0].sum()), "ai": int(changed[b.label == 1].sum())}
    audit["texts_with_markdown_in_final_dataset"] = int(df["text"].str.contains(r"\*|URL_\d", regex=True).sum())

    # final dataset composition
    audit["final_counts"] = {d: {"human": int(((df.domain == d) & (df.label == 0)).sum()),
                                 "chatgpt": int(((df.domain == d) & (df.label == 1)).sum())}
                             for d in sorted(df.domain.unique())}
    audit["split_counts"] = {s: {"human": int(((df.split == s) & (df.label == 0)).sum()),
                                 "chatgpt": int(((df.split == s) & (df.label == 1)).sum())}
                             for s in ["train", "val", "test"]}
    audit["median_words_by_domain_and_class"] = {
        d: {"human": float(df[(df.domain == d) & (df.label == 0)].n_words.median()),
            "chatgpt": float(df[(df.domain == d) & (df.label == 1)].n_words.median())}
        for d in sorted(df.domain.unique())}

    out = ROOT / "results" / "data_audit.json"
    out.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
