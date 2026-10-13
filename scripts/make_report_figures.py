"""
make_report_figures.py - Figures for the written report that combine BERT and baseline results.

The comparison charts drawn on Colab used the baseline numbers from before the markdown
clean-up. This script redraws them from the CURRENT results files so that every figure in the
report agrees with the tables:
    results/bert_metrics_v1.json + results/baseline_metrics.json + results/robustness.json
It also draws the two diagrams (processing pipeline, system architecture).

Outputs (results/figures/):
    report_model_comparison.png, report_per_domain.png, report_robustness.png,
    report_confusion_matrices.png, report_pipeline.png, report_architecture.png
"""

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from plot_style import AI_COLOR, HUMAN_COLOR, MUTED, SERIES, plt  # noqa: E402

RES = ROOT / "results"
FIGS = RES / "figures"
NAVY, TEAL, SLATE = "#1F2A44", "#2C7A7B", "#4A5568"

base = json.loads((RES / "baseline_metrics.json").read_text())["models"]
bert = json.loads((RES / "bert_metrics_v1.json").read_text())
robust = json.loads((RES / "robustness.json").read_text())["models"]

MODELS = {**base, "BERT (fine-tuned)": bert["test"]}
SHORT = {"BoW + Naive Bayes": "BoW + NB", "TF-IDF + LR (stopwords removed)": "TF-IDF + LR\n(no stopwords)",
         "TF-IDF + LR (stopwords kept)": "TF-IDF + LR\n(stopwords kept)", "Char n-gram TF-IDF + SVM": "Char n-gram\n+ SVM",
         "Stylometric + LR": "Stylometric\n+ LR", "BERT (fine-tuned)": "BERT\n(fine-tuned)"}


def model_comparison():
    keys, labels = ["accuracy", "precision", "recall", "f1", "roc_auc"], ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    names = list(MODELS)
    x, w = np.arange(len(keys)), 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (name, color) in enumerate(zip(names, SERIES)):
        bars = ax.bar(x + (i - (len(names) - 1) / 2) * w, [MODELS[name][k] for k in keys],
                      width=w * 0.92, color=color, label=name)
        if name.startswith("BERT"):
            ax.bar_label(bars, fmt="%.3f", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score on test set")
    ax.set_title("BERT and the five classical baselines (test set, 1,335 texts)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(FIGS / "report_model_comparison.png")
    plt.close(fig)


def per_domain():
    domains = ["Q&A (HC3)", "essay", "reuter", "wp"]
    show = ["BoW + Naive Bayes", "TF-IDF + LR (stopwords kept)", "Char n-gram TF-IDF + SVM", "BERT (fine-tuned)"]
    colors = [SERIES[0], SERIES[2], SERIES[3], SERIES[5]]
    x, w = np.arange(len(domains)), 0.8 / len(show)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, color) in enumerate(zip(show, colors)):
        bars = ax.bar(x + (i - (len(show) - 1) / 2) * w, [MODELS[name]["per_domain_accuracy"][d] for d in domains],
                      width=w * 0.92, color=color, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, color=MUTED)
    ax.set_xticks(x, ["Q&A (HC3)", "Essays", "Reuters news", "Stories (wp)"])
    ax.set_ylim(0.5, 1.04)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per domain (test set)")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "report_per_domain.png")
    plt.close(fig)


def robustness():
    r = dict(robust)
    r["BERT (fine-tuned)"] = bert["robustness"]
    names = ["BoW + Naive Bayes", "TF-IDF + LR (stopwords removed)", "TF-IDF + LR (stopwords kept)",
             "Char n-gram TF-IDF + SVM", "Stylometric + LR", "BERT (fine-tuned)"]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 4))
    b1 = ax.bar(x - 0.2, [r[n]["accuracy_original"] for n in names], 0.38, color=HUMAN_COLOR, label="Original test set")
    b2 = ax.bar(x + 0.2, [r[n]["accuracy_edited"] for n in names], 0.38, color=AI_COLOR, label="AI texts lightly edited")
    ax.bar_label(b1, fmt="%.3f", fontsize=7, color=MUTED)
    ax.bar_label(b2, fmt="%.3f", fontsize=7, color=MUTED)
    ax.set_xticks(x, [SHORT[n] for n in names], fontsize=8)
    ax.set_ylim(0.4, 1.04)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy before and after light editing of the AI test texts")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(FIGS / "report_robustness.png")
    plt.close(fig)


def confusion_side_by_side():
    """BERT vs the best baseline confusion matrices, side by side."""
    pairs = [("TF-IDF + LR (stopwords kept)", base["TF-IDF + LR (stopwords kept)"]["confusion_matrix"]),
             ("Char n-gram TF-IDF + SVM", base["Char n-gram TF-IDF + SVM"]["confusion_matrix"]),
             ("BERT (fine-tuned)", bert["test"]["confusion_matrix"])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (name, cm) in zip(axes, pairs):
        cm = np.array(cm)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > 400 else "#0b0b0b")
        ax.set_xticks([0, 1], ["human", "AI"])
        ax.set_yticks([0, 1], ["human", "AI"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIGS / "report_confusion_matrices.png")
    plt.close(fig)


# ---------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, title, sub="", fc="white", ec=NAVY, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2 + (0.022 if sub else 0), title, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=tc)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.035, sub, ha="center", va="center", fontsize=7.6, color=SLATE)


def arrow(ax, p0, p1, color=TEAL, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12, color=color, lw=1.4,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


def pipeline():
    fig = plt.figure(figsize=(10, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # row 1: data
    r1 = 0.74
    box(ax, 0.02, r1, 0.17, 0.16, "Raw corpora", "Ghostbuster + HC3")
    box(ax, 0.23, r1, 0.17, 0.16, "Cleaning & leak fixes", "7 data issues fixed")
    box(ax, 0.44, r1, 0.17, 0.16, "Length matching", "20-300 words")
    box(ax, 0.65, r1, 0.15, 0.16, "Stratified split", "70 / 15 / 15")
    box(ax, 0.84, r1, 0.14, 0.16, "Augmentation", "train AI only")
    for a, b in [(0.19, 0.23), (0.40, 0.44), (0.61, 0.65), (0.80, 0.84)]:
        arrow(ax, (a, r1 + 0.08), (b, r1 + 0.08))
    # row 2: two branches
    r2a, r2b = 0.44, 0.17
    ax.text(0.15, 0.61, "Classical branch", fontsize=8.5, color=SLATE, fontweight="bold")
    ax.text(0.02, 0.36, "Transformer branch", fontsize=8.5, color=SLATE, fontweight="bold")
    box(ax, 0.02, r2a, 0.2, 0.15, "Classical preprocessing", "tokenize, stopwords, lemmatize")
    box(ax, 0.26, r2a, 0.2, 0.15, "Features", "BoW, TF-IDF, char n-grams, POS")
    box(ax, 0.50, r2a, 0.18, 0.15, "NB / LR / SVM", "5 baselines")
    box(ax, 0.02, r2b, 0.2, 0.15, "Light cleaning", "keep stopwords & punctuation")
    box(ax, 0.26, r2b, 0.2, 0.15, "WordPiece tokens", "max 320, [CLS] ... [SEP]")
    box(ax, 0.50, r2b, 0.18, 0.15, "Fine-tuned BERT", "12-layer Transformer")
    for y in (r2a, r2b):
        arrow(ax, (0.22, y + 0.075), (0.26, y + 0.075))
        arrow(ax, (0.46, y + 0.075), (0.50, y + 0.075))
    arrow(ax, (0.91, r1), (0.91, 0.66), ls="--")
    ax.plot([0.12, 0.91], [0.66, 0.66], color=TEAL, lw=1.4, ls="--")
    arrow(ax, (0.12, 0.66), (0.12, r2a + 0.15), ls="--")
    ax.plot([0.005, 0.005], [r2b + 0.075, 0.66], color=TEAL, lw=1.4, ls="--")
    ax.plot([0.005, 0.12], [0.66, 0.66], color=TEAL, lw=1.4, ls="--")
    arrow(ax, (0.005, r2b + 0.075), (0.02, r2b + 0.075), ls="--")
    # right: calibration and outputs
    box(ax, 0.73, 0.30, 0.25, 0.2, "Calibration (validation set)",
        "temperature, ensemble, thresholds", fc="#EEF4F4", ec=TEAL)
    arrow(ax, (0.68, r2a + 0.075), (0.73, 0.43))
    arrow(ax, (0.68, r2b + 0.075), (0.73, 0.37))
    box(ax, 0.73, 0.03, 0.25, 0.17, "Verdict + explanation",
        "Likely AI / Uncertain / Likely Human", fc=NAVY, ec=NAVY, tc="white")
    ax.texts[-1].set_color("#C9D3E6")
    arrow(ax, (0.855, 0.30), (0.855, 0.20))
    fig.savefig(FIGS / "report_pipeline.png", dpi=170)
    plt.close(fig)


def architecture():
    fig = plt.figure(figsize=(10, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.02, 0.36, 0.165, 0.3, "Web browser", "index.html, style.css, app.js")
    box(ax, 0.29, 0.36, 0.19, 0.3, "FastAPI server", "app.py: validation, routes,\nstatic files")
    box(ax, 0.56, 0.36, 0.19, 0.3, "Detector", "detector.py: scoring,\nexplanation, verdict")
    box(ax, 0.82, 0.70, 0.16, 0.2, "Fine-tuned BERT", "detectai-bert/")
    box(ax, 0.82, 0.41, 0.16, 0.2, "Baselines", "TF-IDF LR, char SVM")
    box(ax, 0.82, 0.12, 0.16, 0.2, "Calibration files", "thresholds_*.json")
    arrow(ax, (0.19, 0.55), (0.29, 0.55))
    arrow(ax, (0.29, 0.47), (0.19, 0.47))
    ax.text(0.24, 0.585, "POST /api/predict", ha="center", fontsize=7.2, color=SLATE)
    ax.text(0.24, 0.415, "JSON result", ha="center", fontsize=7.2, color=SLATE)
    arrow(ax, (0.48, 0.51), (0.56, 0.51), style="<|-|>")
    for y in (0.80, 0.51, 0.22):
        arrow(ax, (0.75, 0.51), (0.82, y))
    box(ax, 0.29, 0.03, 0.46, 0.17, "preprocessing.py",
        "clean_text, classical_preprocess, stylometric_features (POS tags)", fc="#EEF4F4", ec=TEAL)
    arrow(ax, (0.655, 0.36), (0.655, 0.20), style="<|-|>")
    ax.text(0.02, 0.9, "Local machine (CPU only)", fontsize=9, color=SLATE, fontweight="bold")
    ax.text(0.56, 0.9, "BERT trained on Google Colab (T4 GPU)", fontsize=8, color=SLATE, style="italic")
    fig.savefig(FIGS / "report_architecture.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    model_comparison()
    per_domain()
    robustness()
    confusion_side_by_side()
    pipeline()
    architecture()
    print("Saved report figures to", FIGS)
