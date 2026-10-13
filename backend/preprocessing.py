"""
preprocessing.py - Text preprocessing used across DetectAI.

Three functions:
  clean_text(text)            -> light cleaning for BERT (keeps punctuation and stopwords,
                                 because BERT uses them as context)
  classical_preprocess(text)  -> classic NLP pipeline for Bag-of-Words / TF-IDF models:
                                 lowercase -> tokenize -> remove stopwords -> lemmatize
  stylometric_features(text)  -> dictionary of hand-crafted writing-style features
                                 (sentence lengths, vocabulary richness, POS-tag ratios ...)
"""

import hashlib
import html
import os
import re
import string
import unicodedata

import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import PerceptronTagger
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()
TAGGER = PerceptronTagger()   # load the POS tagger once (nltk.pos_tag reloads it on every call)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
# Reddit/markdown formatting appears almost only in the human texts of our sources, so we remove
# it - otherwise a model could learn "asterisks = human" instead of real writing style.
URL_PLACEHOLDER_RE = re.compile(r"\(?\bURL_\d+\b\)?")
MD_UNDERSCORE_RE = re.compile(r"(?<!\w)_+|_+(?!\w)")
MD_QUOTE_RE = re.compile(r"(^|\s)>+\s(?=[A-Z\"'])")   # "> Quoted text" (but not "a > b")
QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


# ----------------------------------------------------------------------------
# 1. Light cleaning (for BERT)
# ----------------------------------------------------------------------------
def strip_markdown(text):
    """Remove Reddit/markdown formatting: link placeholders, *emphasis*, _emphasis_, > quotes."""
    text = URL_PLACEHOLDER_RE.sub(" ", text)           # "(URL_0)" link placeholders in Reddit data
    text = text.replace("*", "")                       # markdown emphasis: *word*, **bold**
    text = MD_UNDERSCORE_RE.sub("", text)              # markdown emphasis: _word_
    text = MD_QUOTE_RE.sub(r"\1", text)                # markdown quote marker: "> Quoted text"
    return SPACE_RE.sub(" ", text).strip()


def clean_text(text, remove_markdown=True):
    """Normalise unicode, remove URLs, HTML and markdown, collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text))   # e.g. curly quotes / odd spaces -> standard form
    text = html.unescape(text)                         # &amp; -> &
    text = text.translate(QUOTES)                      # curly quotes -> straight quotes
    text = HTML_TAG_RE.sub(" ", text)                  # <br> -> space
    text = URL_RE.sub(" ", text)                       # drop links
    if remove_markdown:
        text = strip_markdown(text)
    text = SPACE_RE.sub(" ", text).strip()             # many spaces/newlines -> one space
    return text


# ----------------------------------------------------------------------------
# 2. Classical preprocessing (for BoW / TF-IDF)
# ----------------------------------------------------------------------------
def classical_preprocess(text, remove_stopwords=True, use_stemming=False):
    """Lowercase, tokenize, drop punctuation (+ stopwords), then lemmatize (or stem)."""
    tokens = word_tokenize(clean_text(text).lower())
    tokens = [t for t in tokens if t.isalpha()]                 # keep words only
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if use_stemming:
        tokens = [STEMMER.stem(t) for t in tokens]              # running -> run (crude rule-based cut)
    else:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]      # studies -> study (dictionary form)
    return " ".join(tokens)


# ----------------------------------------------------------------------------
# 3. Stylometric features
# ----------------------------------------------------------------------------
STYLO_FEATURES = [
    "sent_len_mean", "sent_len_std", "type_token_ratio", "hapax_ratio",
    "avg_word_len", "stopword_ratio", "punct_density", "commas_per_sent",
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio",
]


def stylometric_features(text):
    """Return a dict of writing-style features for one text."""
    text = clean_text(text)
    sentences = sent_tokenize(text) or [text]
    tokens = word_tokenize(text)
    words = [t.lower() for t in tokens if t.isalpha()]
    n_words = max(len(words), 1)

    sent_lengths = [len([t for t in word_tokenize(s) if t.isalpha()]) for s in sentences]

    # Vocabulary richness
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    hapax = sum(1 for c in counts.values() if c == 1)          # words used exactly once

    # Part-of-speech tags (Penn Treebank tagset)
    tags = [tag for _, tag in TAGGER.tag(tokens)] if tokens else []
    n_tags = max(len(tags), 1)

    def tag_ratio(prefixes):
        return sum(1 for t in tags if t.startswith(prefixes)) / n_tags

    n_punct = sum(1 for ch in text if ch in string.punctuation)

    return {
        "sent_len_mean": float(np.mean(sent_lengths)),
        "sent_len_std": float(np.std(sent_lengths)),            # AI text is often more uniform
        "type_token_ratio": len(counts) / n_words,
        "hapax_ratio": hapax / n_words,
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "stopword_ratio": sum(1 for w in words if w in STOPWORDS) / n_words,
        "punct_density": n_punct / max(len(text), 1),
        "commas_per_sent": text.count(",") / len(sentences),
        "noun_ratio": tag_ratio(("NN",)),
        "verb_ratio": tag_ratio(("VB",)),
        "adj_ratio": tag_ratio(("JJ",)),
        "adv_ratio": tag_ratio(("RB",)),
        "pron_ratio": tag_ratio(("PRP", "WP")),
    }


# ----------------------------------------------------------------------------
# Batch helpers - used inside scikit-learn Pipelines (via FunctionTransformer)
# so a saved model can take raw text directly. They live here (not as lambdas)
# so that joblib can save and reload the pipelines.
# ----------------------------------------------------------------------------
def batch_clean(texts):
    return [clean_text(t) for t in texts]


def batch_classical_remove_stopwords(texts):
    return [classical_preprocess(t, remove_stopwords=True) for t in texts]


def batch_classical_keep_stopwords(texts):
    return [classical_preprocess(t, remove_stopwords=False) for t in texts]


# Stylometric features need POS tagging, which is slow, so results are cached by text.
# The cache is only an optional speed-up: without it the features are simply recomputed.
_STYLO_CACHE = {}


def _key(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_stylo_cache(path):
    if os.path.exists(path):
        _STYLO_CACHE.update(joblib.load(path))
    return len(_STYLO_CACHE)


def save_stylo_cache(path):
    joblib.dump(_STYLO_CACHE, path)


def batch_stylometric(texts):
    rows = []
    for t in texts:
        k = _key(t)
        if k not in _STYLO_CACHE:
            feats = stylometric_features(t)
            _STYLO_CACHE[k] = [feats[f] for f in STYLO_FEATURES]
        rows.append(_STYLO_CACHE[k])
    return np.array(rows)


if __name__ == "__main__":
    demo = "The studies were running quickly! <b>Visit</b> https://example.com for more, okay?"
    print("clean_text          :", clean_text(demo))
    print("classical_preprocess:", classical_preprocess(demo))
    print("with stemming       :", classical_preprocess(demo, use_stemming=True))
    print("stylometric_features:", {k: round(v, 3) for k, v in stylometric_features(demo).items()})
