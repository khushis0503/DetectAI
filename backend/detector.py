"""
detector.py - The DetectAI prediction engine used by the website.

Which model?
  * If a fine-tuned BERT folder exists (backend/models/detectai-bert/), BERT is used.
  * Otherwise we fall back to the TF-IDF + Logistic Regression baseline (tfidf_lr.joblib).
  * Optionally the char n-gram SVM is averaged in (ensemble) - decided by scripts/calibrate.py
    and stored in backend/models/thresholds.json.

predict(text) returns:
  verdict ("Likely AI" / "Uncertain" / "Likely Human"), AI probability, confidence, model used,
  word-level importance (Gradient x Input for BERT, coefficient x TF-IDF for the baseline),
  per-sentence AI scores, a per-paragraph (or per-chunk) breakdown, and stylometric features.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from nltk.tokenize import PunktTokenizer

from backend.preprocessing import LEMMATIZER, clean_text, stylometric_features

BACKEND = Path(__file__).resolve().parent
MODELS = BACKEND / "models"
RESULTS = BACKEND.parent / "results"

# BERT folder: the name we agreed on, or the name inside the Colab zip
BERT_DIRS = [MODELS / "detectai-bert", MODELS / "detectai_bert_model"]
if os.environ.get("DETECTAI_BERT_DIR"):               # optional override (used for testing)
    BERT_DIRS.insert(0, Path(os.environ["DETECTAI_BERT_DIR"]))

MAX_LEN = 320           # BERT tokens per chunk (same as training)
SHORT_TEXT_WORDS = 50   # below this the result is flagged as less reliable
DEFAULT_THRESHOLDS = {"likely_ai": 0.75, "likely_human": 0.25}   # only if not calibrated yet

SENTENCES = PunktTokenizer("english")
WORD_RE = re.compile(r"[A-Za-z]+")


class Detector:
    def __init__(self, use_calibration=True):
        self.bert_dir = next((d for d in BERT_DIRS if (d / "config.json").exists()), None)
        if os.environ.get("DETECTAI_FORCE_BASELINE") == "1":   # e.g. demo laptop without enough RAM
            self.bert_dir = None

        if self.bert_dir:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.kind = "bert"
            self.model_name = "Fine-tuned BERT (bert-base-uncased)"
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_dir)
            self.bert = AutoModelForSequenceClassification.from_pretrained(self.bert_dir).eval()
        else:
            self.kind = "tfidf_lr"
            self.model_name = "TF-IDF + Logistic Regression (baseline)"
            self.pipe = joblib.load(MODELS / "tfidf_lr.joblib")
            self.vocab = self.pipe.named_steps["tfidf"].get_feature_names_out()
            self.coef = self.pipe.named_steps["clf"].coef_[0]

        self.svm = joblib.load(MODELS / "char_svm.joblib")

        # Thresholds, temperature + ensemble settings (made by scripts/calibrate.py for THIS model)
        self.config, self.calibrated = {}, False
        if use_calibration:
            for cfg_path in [MODELS / f"thresholds_{self.kind}.json", MODELS / "thresholds.json"]:
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text())
                    if cfg.get("main_model") == self.kind:
                        self.config, self.calibrated = cfg, True
                        break
        self.thresholds = self.config.get("thresholds", DEFAULT_THRESHOLDS)
        self.temperature = self.config.get("temperature", 1.0)   # >1 softens over-confident scores
        ens = self.config.get("ensemble", {})
        self.use_ensemble = bool(ens.get("used", False))
        self.w_main = ens.get("weight_main", 1.0) if self.use_ensemble else 1.0

        ref_path = MODELS / "style_reference.json"
        self.style_reference = json.loads(ref_path.read_text()) if ref_path.exists() else {}

        # How trustworthy are the word highlights? (from scripts/saliency_check.py, if it was run)
        self.highlight_note = ""
        faith_path = RESULTS / "saliency_faithfulness.json"
        if faith_path.exists():
            f = json.loads(faith_path.read_text()).get(self.kind)
            if f:
                self.highlight_note = (
                    f"Deletion test on {f['n_texts']} test texts: removing the 5 most-highlighted words "
                    f"beat removing 5 random words in {f['share_where_top5_beats_random']:.0%} of texts"
                    + (" - an approximate explanation, best read as a rough guide."
                       if self.kind == "bert" else " - exact for this linear model."))

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------
    def _bert_chunks(self, text):
        """Split a text into pieces of at most MAX_LEN tokens, cutting only between sentences.
        Returns (start, end) character spans."""
        limit = MAX_LEN - 2                     # room for [CLS] and [SEP]
        chunks, cur_start, cur_end, cur_len = [], None, None, 0
        for s, e in SENTENCES.span_tokenize(text):
            n = len(self.tokenizer.tokenize(text[s:e]))
            if cur_start is not None and cur_len + n > limit:
                chunks.append((cur_start, cur_end))
                cur_start, cur_len = None, 0
            if cur_start is None:
                cur_start = s
            cur_end, cur_len = e, cur_len + n
        if cur_start is not None:
            chunks.append((cur_start, cur_end))
        return chunks or [(0, len(text))]

    def _bert_proba_raw(self, pieces, batch_size=16):
        """AI probability for short pieces of text (each fits in one BERT input)."""
        torch = self.torch
        probs = np.zeros(len(pieces))
        order = np.argsort([len(p) for p in pieces])            # similar lengths together = faster
        for i in range(0, len(pieces), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tokenizer([pieces[j] for j in idx], truncation=True, max_length=MAX_LEN,
                                 padding=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.bert(**enc).logits
            probs[idx] = torch.softmax(logits, -1)[:, 1].numpy()
        return probs

    def main_proba(self, texts):
        """AI probability from the main model (BERT, or TF-IDF + LR)."""
        if self.kind != "bert":
            return self.pipe.predict_proba(list(texts))[:, 1]
        # BERT: long texts are split into chunks; text score = token-weighted mean of chunk scores
        pieces, owner, weight = [], [], []
        for t_i, text in enumerate(texts):
            for s, e in self._bert_chunks(text):
                pieces.append(text[s:e])
                owner.append(t_i)
                weight.append(len(self.tokenizer.tokenize(text[s:e])) or 1)
        p = self._bert_proba_raw(pieces)
        owner, weight = np.array(owner), np.array(weight, dtype=float)
        return np.array([np.average(p[owner == i], weights=weight[owner == i]) for i in range(len(texts))])

    def svm_proba(self, texts):
        return self.svm.predict_proba(list(texts))[:, 1]

    @staticmethod
    def apply_temperature(p, temperature):
        """Temperature scaling: divide the model's log-odds by T. T > 1 pulls over-confident
        probabilities (0.999...) towards 0.5 without changing which text scores higher."""
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return 1 / (1 + np.exp(-np.log(p / (1 - p)) / temperature))

    def score(self, texts):
        """Final AI probability (temperature-scaled, and ensemble if calibration chose it)."""
        p = self.apply_temperature(self.main_proba(texts), self.temperature)
        if self.use_ensemble:
            p = self.w_main * p + (1 - self.w_main) * self.svm_proba(texts)
        return p

    # ------------------------------------------------------------------
    # Word-level importance
    # ------------------------------------------------------------------
    def _bert_word_scores(self, text, offset=0):
        """Gradient x Input, exactly as in the Colab notebook: for every word, the effect of its
        token embeddings on (AI logit - human logit). Returns [(start, end, score)]."""
        torch = self.torch
        enc = self.tokenizer(text, truncation=True, max_length=MAX_LEN, return_tensors="pt",
                             return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids()
        embeds = self.bert.get_input_embeddings()(enc["input_ids"]).detach().requires_grad_(True)
        logits = self.bert(inputs_embeds=embeds, attention_mask=enc["attention_mask"]).logits[0]
        (logits[1] - logits[0]).backward()
        token_sal = (embeds.grad[0] * embeds[0]).sum(-1).detach().numpy()

        words = {}                                   # merge WordPieces back into whole words
        for tok_i, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            s, e = offsets[tok_i]
            if w_id not in words:
                words[w_id] = [s, e, 0.0]
            words[w_id][1] = e
            words[w_id][2] += float(token_sal[tok_i])
        return [(s + offset, e + offset, v) for s, e, v in words.values()]

    def _tfidf_word_scores(self, text):
        """Baseline: contribution of each word = LR coefficient x its TF-IDF value.
        Bigram contributions are split between their two words; a word that appears several
        times shares its contribution between its occurrences."""
        processed = self.pipe.named_steps["preprocess"].transform([text])
        X = self.pipe.named_steps["tfidf"].transform(processed)
        lemma_score = defaultdict(float)
        for f, value in zip(X.indices, X.data):
            parts = self.vocab[f].split()
            for part in parts:
                lemma_score[part] += value * self.coef[f] / len(parts)

        tokens = [(m.start(), m.end(), LEMMATIZER.lemmatize(m.group().lower()))
                  for m in WORD_RE.finditer(text)]
        counts = defaultdict(int)
        for _, _, lemma in tokens:
            counts[lemma] += 1
        return [(s, e, lemma_score.get(lemma, 0.0) / counts[lemma]) for s, e, lemma in tokens]

    @staticmethod
    def _segments(text, word_scores):
        """Turn (start, end, score) into display segments covering the whole text."""
        peak = max((abs(v) for _, _, v in word_scores), default=0) or 1.0
        segs, pos = [], 0
        for s, e, v in sorted(word_scores):
            if s < pos:
                continue
            if s > pos:
                segs.append({"t": text[pos:s], "s": None})
            segs.append({"t": text[s:e], "s": round(v / peak, 3)})
            pos = e
        if pos < len(text):
            segs.append({"t": text[pos:], "s": None})
        return segs

    # ------------------------------------------------------------------
    def verdict(self, p):
        if p >= self.thresholds["likely_ai"]:
            return "Likely AI"
        if p <= self.thresholds["likely_human"]:
            return "Likely Human"
        return "Uncertain"

    def predict(self, raw_text):
        # Keep paragraph structure for display; clean each paragraph the same way as training data
        paragraphs = [clean_text(p) for p in re.split(r"\n\s*\n|\n", raw_text) if clean_text(p)]
        text = "\n\n".join(paragraphs)
        n_words = len(text.split())

        prob = float(self.score([text])[0])
        verdict = self.verdict(prob)

        # Confidence = distance from 50/50, reduced for short texts
        confidence = abs(prob - 0.5) * 2
        short = n_words < SHORT_TEXT_WORDS
        if short:
            confidence *= max(0.3, n_words / SHORT_TEXT_WORDS)

        # Character spans of paragraphs inside `text`
        para_spans, pos = [], 0
        for p in paragraphs:
            para_spans.append((pos, pos + len(p)))
            pos += len(p) + 2

        # Word importance (main model only)
        if self.kind == "bert":
            word_scores, chunk_spans = [], []
            for ps, pe in para_spans:
                for cs, ce in self._bert_chunks(text[ps:pe]):
                    chunk_spans.append((ps + cs, ps + ce))
                    word_scores += self._bert_word_scores(text[ps + cs:ps + ce], offset=ps + cs)
        else:
            word_scores = self._tfidf_word_scores(text)
            chunk_spans = para_spans

        # Per-sentence scores
        sent_spans = [(ps + s, ps + e) for ps, pe in para_spans
                      for s, e in SENTENCES.span_tokenize(text[ps:pe])]
        sent_texts = [text[s:e] for s, e in sent_spans]
        sent_probs = self.score(sent_texts) if sent_texts else []
        sentences = [{"text": t, "ai_probability": round(float(p), 4), "n_words": len(t.split())}
                     for t, p in zip(sent_texts, sent_probs)]

        # Per-paragraph breakdown (or per-chunk if one long paragraph was split for BERT)
        parts = para_spans if len(para_spans) > 1 else (chunk_spans if len(chunk_spans) > 1 else [])
        part_label = "Paragraph" if len(para_spans) > 1 else "Part"
        part_probs = self.score([text[s:e] for s, e in parts]) if parts else []
        breakdown = [{"label": f"{part_label} {i + 1}", "preview": text[s:e][:140],
                      "n_words": len(text[s:e].split()), "ai_probability": round(float(p), 4),
                      "verdict": self.verdict(float(p))}
                     for i, ((s, e), p) in enumerate(zip(parts, part_probs))]

        return {
            "verdict": verdict,
            "ai_probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "n_words": n_words,
            "short_text_warning": short,
            "model_used": self.model_name + (" + char n-gram SVM ensemble" if self.use_ensemble else ""),
            "thresholds": self.thresholds,
            "highlighted": self._segments(text, word_scores),
            "highlight_method": "Gradient x Input (BERT)" if self.kind == "bert"
                                else "Logistic Regression coefficient x TF-IDF",
            "highlight_note": self.highlight_note,
            "sentences": sentences,
            "breakdown": breakdown,
            "style": {k: round(v, 4) for k, v in stylometric_features(text).items()},
            "style_reference": self.style_reference,
        }

    def info(self):
        """Model description + test-set metrics for /api/model-info."""
        if self.kind == "bert":
            metrics_file = next((p for p in [RESULTS / "bert_metrics.json", self.bert_dir / "bert_metrics.json",
                                             self.bert_dir.parent / "bert_metrics.json"] if p.exists()), None)
            test = json.loads(metrics_file.read_text()).get("test") if metrics_file else None
        else:
            test = json.loads((RESULTS / "baseline_metrics.json").read_text())["models"][
                "TF-IDF + LR (stopwords kept)"]
        return {
            "model": self.kind,
            "model_name": self.model_name,
            "ensemble_used": self.use_ensemble,
            "ensemble": self.config.get("ensemble"),
            "thresholds": self.thresholds,
            "temperature": self.temperature,
            "thresholds_calibrated": self.calibrated,
            "threshold_check": self.config.get("verdict_rates"),
            "main_model_test_metrics": test,
        }
