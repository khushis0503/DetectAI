"""
train_baselines.py - EDA figures + 5 classical baseline models.

Training data : train split + augmented (edited) AI copies
Evaluation    : the fixed test split (never seen during training)

Models (Unit 2 + Unit 4 of the syllabus):
  1. BoW + Naive Bayes                          (Bag of Words counts)
  2. TF-IDF + Logistic Regression, stopwords removed
  3. TF-IDF + Logistic Regression, stopwords kept   -> saved to backend/models/
  4. char n-gram TF-IDF + Linear SVM (calibrated)   -> saved to backend/models/
  5. Stylometric features + Logistic Regression     (POS tags, sentence lengths ...)

Outputs: results/baseline_metrics.json, results/figures/*.png, backend/models/*.joblib
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from backend.preprocessing import (STYLO_FEATURES, batch_classical_keep_stopwords,  # noqa: E402
                                   batch_classical_remove_stopwords, batch_clean,
                                   batch_stylometric, load_stylo_cache, save_stylo_cache)
from plot_style import AI_COLOR, HUMAN_COLOR, MUTED, SERIES, plt  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
MODELS = ROOT / "backend" / "models"
SEED = 42
STYLO_CACHE = DATA / "stylometric_cache.joblib"   # POS-tag features, computed once


# ----------------------------------------------------------------------------
# Model definitions - every model is a Pipeline that takes RAW text as input
# ----------------------------------------------------------------------------
def build_models():
    return {
        "BoW + Naive Bayes": Pipeline([
            ("preprocess", FunctionTransformer(batch_classical_remove_stopwords)),
            ("bow", CountVectorizer(min_df=2)),                       # word counts
            ("clf", MultinomialNB()),
        ]),
        "TF-IDF + LR (stopwords removed)": Pipeline([
            ("preprocess", FunctionTransformer(batch_classical_remove_stopwords)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=4.0)),
        ]),
        "TF-IDF + LR (stopwords kept)": Pipeline([
            ("preprocess", FunctionTransformer(batch_classical_keep_stopwords)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, C=4.0)),
        ]),
        "Char n-gram TF-IDF + SVM": Pipeline([
            ("preprocess", FunctionTransformer(batch_clean)),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                      sublinear_tf=True, max_features=300_000)),
            # LinearSVC gives no probabilities; calibration turns its scores into probabilities
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
        ]),
        "Stylometric + LR": Pipeline([
            ("features", FunctionTransformer(batch_stylometric)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]),
    }


SAVE_AS = {   # model name -> file in backend/models/
    "TF-IDF + LR (stopwords kept)": "tfidf_lr.joblib",
    "Char n-gram TF-IDF + SVM": "char_svm.joblib",
    # kept so robustness_test.py can reuse them without retraining
    "BoW + Naive Bayes": "other_baselines/bow_nb.joblib",
    "TF-IDF + LR (stopwords removed)": "other_baselines/tfidf_lr_nostop.joblib",
    "Stylometric + LR": "other_baselines/stylometric_lr.joblib",
}


def evaluate(y_true, y_prob, domains):
    """All the metrics we report (AI = positive class)."""
    y_pred = (y_prob >= 0.5).astype(int)
    per_domain = {d: round(accuracy_score(y_true[domains == d], y_pred[domains == d]), 4)
                  for d in sorted(set(domains))}
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "per_domain_accuracy": per_domain,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),   # [[TN, FP], [FN, TP]]
    }


# ----------------------------------------------------------------------------
# EDA figures
# ----------------------------------------------------------------------------
def eda_figures(df):
    # 1. Class balance per domain
    counts = pd.crosstab(df["domain"], df["source"])
    x = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, (src, color) in enumerate([("human", HUMAN_COLOR), ("ChatGPT", AI_COLOR)]):
        bars = ax.bar(x + (i - 0.5) * 0.38, counts[src], width=0.36, color=color, label=src)
        ax.bar_label(bars, fontsize=8, color=MUTED)
    ax.set_xticks(x, counts.index)
    ax.set_ylabel("Number of texts")
    ax.set_title("Class balance per domain")
    ax.legend(loc="lower right")
    fig.savefig(FIGS / "eda_class_balance.png")
    plt.close(fig)

    # 2. Length distribution (words per text) by class
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.arange(20, 320, 20)
    for src, color in [("human", HUMAN_COLOR), ("ChatGPT", AI_COLOR)]:
        ax.hist(df.loc[df["source"] == src, "n_words"], bins=bins, histtype="step",
                linewidth=2, color=color, label=src)
    ax.set_xlabel("Words per text")
    ax.set_ylabel("Number of texts")
    ax.set_title("Text length distribution (random 20-300 word cuts)")
    ax.legend()
    fig.savefig(FIGS / "eda_length_distribution.png")
    plt.close(fig)


def stylometric_boxplots(feats, labels):
    fig, axes = plt.subplots(3, 5, figsize=(14, 8))
    for ax, name in zip(axes.flat, STYLO_FEATURES):
        col = STYLO_FEATURES.index(name)
        data = [feats[labels == 0, col], feats[labels == 1, col]]
        bp = ax.boxplot(data, tick_labels=["human", "AI"], patch_artist=True, showfliers=False,
                        widths=0.55, medianprops={"color": "#0b0b0b"})
        for patch, color in zip(bp["boxes"], [HUMAN_COLOR, AI_COLOR]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(name, fontsize=10)
    for ax in list(axes.flat)[len(STYLO_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Stylometric features: human vs AI (train split, outliers hidden)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "eda_stylometric_boxplots.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Result figures
# ----------------------------------------------------------------------------
def result_figures(metrics, probs, y_test, test_domains):
    names = list(metrics)

    # Confusion matrices
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.3))
    for ax, name in zip(axes, names):
        cm = np.array(metrics[name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum(axis=1).max())
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "#0b0b0b")
        ax.set_xticks([0, 1], ["human", "AI"])
        ax.set_yticks([0, 1], ["human", "AI"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "baseline_confusion_matrices.png")
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, color in zip(names, SERIES):
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {metrics[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1)
    ax.set_xlabel("False positive rate (human flagged as AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title("ROC curves - test set")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(FIGS / "baseline_roc_curves.png")
    plt.close(fig)

    # Model comparison (accuracy / F1 / ROC-AUC)
    keys = ["accuracy", "f1", "roc_auc"]
    x = np.arange(len(keys))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, color) in enumerate(zip(names, SERIES)):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, [metrics[name][k] for k in keys],
               width=w * 0.92, color=color, label=name)
    ax.set_xticks(x, ["Accuracy", "F1", "ROC-AUC"])
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Baseline models on the test set")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(FIGS / "baseline_model_comparison.png")
    plt.close(fig)

    # Per-domain accuracy
    domains = sorted(set(test_domains))
    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, color) in enumerate(zip(names, SERIES)):
        ax.bar(x + (i - (len(names) - 1) / 2) * w,
               [metrics[name]["per_domain_accuracy"][d] for d in domains],
               width=w * 0.92, color=color, label=name)
    ax.set_xticks(x, domains)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-domain accuracy (test set)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.savefig(FIGS / "baseline_per_domain_accuracy.png")
    plt.close(fig)


def top_words(pipeline, k=15):
    """Words with the most negative (human) and most positive (AI) LR weights."""
    vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    human = [(vocab[i], round(float(coefs[i]), 3)) for i in order[:k]]
    ai = [(vocab[i], round(float(coefs[i]), 3)) for i in order[::-1][:k]]
    return {"human": human, "AI": ai}


def top_words_figure(words, name, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cls, color in [(axes[0], "human", HUMAN_COLOR), (axes[1], "AI", AI_COLOR)]:
        terms = [w for w, _ in words[cls]][::-1]
        weights = [abs(c) for _, c in words[cls]][::-1]
        ax.barh(terms, weights, color=color)
        ax.set_title(f"Top words -> {cls}")
        ax.set_xlabel("|Logistic Regression weight|")
    fig.suptitle(f"Most informative TF-IDF terms: {name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / filename)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    (MODELS / "other_baselines").mkdir(parents=True, exist_ok=True)

    print(f"Stylometric cache: {load_stylo_cache(STYLO_CACHE)} texts already computed")
    df = pd.read_csv(DATA / "detectai_dataset.csv")
    aug = pd.read_csv(DATA / "train_augmented.csv")
    train = pd.concat([df[df["split"] == "train"], aug], ignore_index=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Train: {len(train)} texts ({len(aug)} of them augmented AI copies)  |  Test: {len(test)}")

    X_train, y_train = train["text"].tolist(), train["label"].values
    X_test, y_test = test["text"].tolist(), test["label"].values
    test_domains = test["domain"].values

    print("Drawing EDA figures ...")
    eda_figures(df)

    metrics, probs, fitted = {}, {}, {}
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        probs[name] = model.predict_proba(X_test)[:, 1]
        metrics[name] = evaluate(y_test, probs[name], test_domains)
        metrics[name]["train_seconds"] = round(time.time() - t0, 1)
        joblib.dump(model, MODELS / SAVE_AS[name])
        fitted[name] = model
        m = metrics[name]
        print(f"{name:34s} acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  "
              f"({m['train_seconds']}s)")

        if name == "Stylometric + LR":   # reuse its feature step for the boxplots
            original_train = df[df["split"] == "train"]
            feats = model.named_steps["features"].transform(original_train["text"].tolist())
            stylometric_boxplots(feats, original_train["label"].values)
            coefs = model.named_steps["clf"].coef_[0]
            metrics[name]["feature_weights"] = {f: round(float(c), 3)
                                                for f, c in zip(STYLO_FEATURES, coefs)}

    top = {name: top_words(fitted[name])
           for name in ["TF-IDF + LR (stopwords removed)", "TF-IDF + LR (stopwords kept)"]}
    top_words_figure(top["TF-IDF + LR (stopwords removed)"], "stopwords removed",
                     "top_words_tfidf_nostop.png")
    top_words_figure(top["TF-IDF + LR (stopwords kept)"], "stopwords kept",
                     "top_words_tfidf_keepstop.png")

    result_figures(metrics, probs, y_test, test_domains)
    save_stylo_cache(STYLO_CACHE)

    out = {"n_train": len(train), "n_train_augmented": len(aug), "n_test": len(test),
           "models": metrics, "top_tfidf_words": top}
    (RESULTS / "baseline_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved metrics -> {RESULTS / 'baseline_metrics.json'}")
    print(f"Saved figures -> {FIGS}")
    print(f"Saved models  -> {MODELS}")

    print("\nTop TF-IDF words (stopwords removed):")
    for cls in ["human", "AI"]:
        print(f"  {cls:5s}:", ", ".join(w for w, _ in top["TF-IDF + LR (stopwords removed)"][cls]))


if __name__ == "__main__":
    main()
