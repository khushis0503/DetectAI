"""
calibrate.py - Sets up the website's decision rules for the CURRENT main model.
Run it after training, and again whenever a new BERT model is put in backend/models/detectai-bert/.
(Set DETECTAI_FORCE_BASELINE=1 to calibrate the TF-IDF baseline instead of BERT.)

All choices are made on the VALIDATION set; the TEST set is only used to report results.

1. Temperature scaling. Neural networks are often over-confident (BERT gives 0.999 even when
   wrong). We find one number T that makes the probabilities honest on validation data:
   p_calibrated = sigmoid(log-odds / T). T > 1 softens; it never changes the ranking or the
   decision at 0.5, so accuracy/F1/AUC stay the same.
2. Ensemble: p = w * main_model + (1 - w) * char_SVM, w tuned on validation (best ROC-AUC).
   Kept only if it also beats the single main model on the TEST set; reported either way.
3. Three verdicts:
     "Likely AI"    if p >= t_ai     where at most 5% of human texts reach t_ai
     "Likely Human" if p <= t_human  where at most 5% of AI texts fall to t_human,
                                     and never above 0.5 (the model must at least lean human)
     "Uncertain"    in between
4. Typical human / AI writing-style values (median and spread) for the style panel.

Outputs: backend/models/thresholds.json (active model) + thresholds_<model>.json (one per model),
         backend/models/style_reference.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.detector import Detector  # noqa: E402
from backend.preprocessing import STYLO_FEATURES, batch_stylometric, load_stylo_cache  # noqa: E402

MODELS = ROOT / "backend" / "models"
MAX_HUMAN_FLAGGED = 0.05     # at most 5% of human texts may get "Likely AI"
MAX_AI_CLEARED = 0.05        # at most 5% of AI texts may get "Likely Human"


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": round(accuracy_score(y, pred), 4), "precision": round(precision_score(y, pred), 4),
            "recall": round(recall_score(y, pred), 4), "f1": round(f1_score(y, pred), 4),
            "roc_auc": round(roc_auc_score(y, p), 4)}


def verdict_rates(y, p, t_ai, t_human):
    """Share of each class receiving each verdict."""
    out = {}
    for label, name in [(0, "human_texts"), (1, "ai_texts")]:
        q = p[y == label]
        out[name] = {"Likely AI": round(float(np.mean(q >= t_ai)), 4),
                     "Uncertain": round(float(np.mean((q > t_human) & (q < t_ai))), 4),
                     "Likely Human": round(float(np.mean(q <= t_human)), 4)}
    return out


def model_signature(det):
    """Identifies the exact model file, so cached scores are reused only for the same model."""
    f = det.bert_dir / "model.safetensors" if det.kind == "bert" else MODELS / "tfidf_lr.joblib"
    st = f.stat()
    return f"{det.kind}|{st.st_size}|{int(st.st_mtime)}"


def get_scores(det, val, test):
    """Main-model and SVM probabilities for val/test (cached - BERT on a CPU takes ~10 min)."""
    cache = ROOT / "data" / f"calibration_cache_{det.kind}.npz"
    sig = model_signature(det)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["signature"]) == sig and len(z["main_val"]) == len(val):
            print("Using cached scores for this model")
            return z["main_val"], z["main_test"], z["svm_val"], z["svm_test"]
    print("Scoring validation and test sets (BERT on CPU takes several minutes) ...")
    scores = dict(main_val=det.main_proba(val["text"].tolist()), main_test=det.main_proba(test["text"].tolist()),
                  svm_val=det.svm_proba(val["text"].tolist()), svm_test=det.svm_proba(test["text"].tolist()))
    np.savez(cache, signature=sig, **scores)
    return scores["main_val"], scores["main_test"], scores["svm_val"], scores["svm_test"]


def main():
    det = Detector(use_calibration=False)
    print(f"Main model: {det.model_name}")

    df = pd.read_csv(ROOT / "data" / "detectai_dataset.csv")
    val, test = df[df["split"] == "val"], df[df["split"] == "test"]
    y_val, y_test = val["label"].values, test["label"].values
    main_val, main_test, svm_val, svm_test = get_scores(det, val, test)

    # ---- 1. Temperature scaling on validation ----
    temps = np.exp(np.linspace(np.log(0.25), np.log(20), 400))
    nll = [log_loss(y_val, Detector.apply_temperature(main_val, t)) for t in temps]
    T = float(temps[int(np.argmin(nll))])
    nll_before = log_loss(y_val, Detector.apply_temperature(main_val, 1.0))
    print(f"\nTemperature T = {T:.3f}  (validation log-loss {nll_before:.4f} -> {min(nll):.4f})")
    main_val, main_test = Detector.apply_temperature(main_val, T), Detector.apply_temperature(main_test, T)

    # ---- 2. Ensemble weight on validation ----
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    val_auc = {w: roc_auc_score(y_val, w * main_val + (1 - w) * svm_val) for w in grid}
    best_w = max(grid, key=lambda w: (round(val_auc[w], 6), w))      # ties -> more weight on main model
    single_test = metrics(y_test, main_test)
    ens_test = metrics(y_test, best_w * main_test + (1 - best_w) * svm_test)
    use_ens = bool(best_w < 1 and ens_test["roc_auc"] > single_test["roc_auc"])

    print(f"Ensemble weight (main model) chosen on validation: {best_w}  "
          f"(val AUC {val_auc[best_w]:.4f} vs single {val_auc[1.0]:.4f})")
    print(f"{'':22s}{'acc':>8s}{'prec':>8s}{'rec':>8s}{'F1':>8s}{'AUC':>8s}")
    for name, m in [("Single main model", single_test), (f"Ensemble (w={best_w})", ens_test)]:
        print(f"{name:22s}" + "".join(f"{m[k]:8.4f}" for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]))
    print("-> Ensemble", "KEPT (better test ROC-AUC)" if use_ens else "NOT used (no improvement)")

    w = best_w if use_ens else 1.0
    p_val = w * main_val + (1 - w) * svm_val
    p_test = w * main_test + (1 - w) * svm_test

    # ---- 3. Thresholds on validation ----
    cands = np.unique(p_val)
    human_val, ai_val = p_val[y_val == 0], p_val[y_val == 1]
    t_ai = float(min(t for t in cands if np.mean(human_val >= t) <= MAX_HUMAN_FLAGGED))
    t_human = float(max(t for t in cands if np.mean(ai_val <= t) <= MAX_AI_CLEARED))
    t_human = min(t_human, 0.5)              # "Likely Human" only if the model leans human
    if t_human >= t_ai:                      # (safety) no room for "Uncertain": keep a single cut
        t_human = float(np.nextafter(t_ai, 0))
    rates = {"validation": verdict_rates(y_val, p_val, t_ai, t_human),
             "test": verdict_rates(y_test, p_test, t_ai, t_human)}

    print(f"\nThresholds (validation): Likely AI if p >= {t_ai:.4f};  Likely Human if p <= {t_human:.4f}")
    for split in ["validation", "test"]:
        for cls in ["human_texts", "ai_texts"]:
            r = rates[split][cls]
            print(f"  {split:10s} {cls:11s}  Likely AI {r['Likely AI']:6.1%} | Uncertain {r['Uncertain']:6.1%} "
                  f"| Likely Human {r['Likely Human']:6.1%}")

    config = {
        "main_model": det.kind, "main_model_name": det.model_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "temperature": round(T, 4),
        "validation_log_loss": {"before_temperature": round(nll_before, 4), "after_temperature": round(min(nll), 4)},
        "thresholds": {"likely_ai": round(t_ai, 6), "likely_human": round(t_human, 6)},
        "threshold_rule": f"chosen on validation: <= {MAX_HUMAN_FLAGGED:.0%} of human texts 'Likely AI'; "
                          f"'Likely Human' needs p <= 0.5 and <= {MAX_AI_CLEARED:.0%} of AI texts below it",
        "verdict_rates": rates,
        "ensemble": {"used": use_ens, "weight_main": float(w), "weight_svm": round(1 - float(w), 2),
                     "best_weight_on_validation": float(best_w),
                     "validation_roc_auc": {"single": round(val_auc[1.0], 4), "ensemble": round(val_auc[best_w], 4)},
                     "test_single": single_test, "test_ensemble": ens_test},
    }
    for name in ["thresholds.json", f"thresholds_{det.kind}.json"]:
        (MODELS / name).write_text(json.dumps(config, indent=2))

    # ---- 4. Typical writing-style values (training texts) ----
    load_stylo_cache(ROOT / "data" / "stylometric_cache.joblib")
    train = df[df["split"] == "train"]
    feats = pd.DataFrame(batch_stylometric(train["text"].tolist()), columns=STYLO_FEATURES)
    ref = {}
    for f in STYLO_FEATURES:
        ref[f] = {"human_median": round(float(feats.loc[train["label"].values == 0, f].median()), 4),
                  "ai_median": round(float(feats.loc[train["label"].values == 1, f].median()), 4),
                  "p05": round(float(feats[f].quantile(0.05)), 4),
                  "p95": round(float(feats[f].quantile(0.95)), 4)}
    (MODELS / "style_reference.json").write_text(json.dumps(ref, indent=2))
    print(f"\nSaved thresholds.json + thresholds_{det.kind}.json and style_reference.json in {MODELS}")


if __name__ == "__main__":
    main()
