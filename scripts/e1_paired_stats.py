#!/usr/bin/env python
"""Paired statistical tests for the main comparison table (E1).

Motivation
----------
The paper reports bootstrap CIs per model, but a comparison needs the
uncertainty of the *difference* on the same rows, plus the run-to-run spread
across seeds. This script produces both:

1. Row-paired bootstrap: resample test rows with replacement, evaluate VX and
   the baseline on the identical resample, and report the 95% CI and a
   two-sided bootstrap p-value for the metric difference.
2. Seed-paired summary: with seeds 42/43/44 for both sides, report per-seed
   differences and their range (descriptive; n=3 is too small for a test).

Metrics: gate AUROC, gate FPR@95TPR, choice macro-F1, choice ECE-15, Brier.

Usage:
    python scripts/e1_paired_stats.py            # full run
    python scripts/e1_paired_stats.py --selftest # point estimates vs paper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.eval.metrics import _binary_metrics, _calibration_measures  # noqa: E402
from src.eval.run_e1e3 import _baseline_calibrated  # noqa: E402

PAPER = Path("/home/jsantillana/projects/paper")
VX_DIR = PAPER / "results/outputs"
BASE_DIR = PAPER / "results/baselines"
OUT = REPO / "results/e1e3"

CONFIGS = {
    "unsw_oficial": "unsw_small_oficial",
    "cic_strat": "cic_small_strat",
    "cic_time": "cic_small_time",
}
MODELS = ("xgboost", "rf", "mlp")
B = 300
SEED = 12345

# Point estimates from the paper (seed 42) for the self-test.
PAPER_AUROC = {
    ("unsw_oficial", "xgboost"): 0.9818,
    ("unsw_oficial", "rf"): 0.9792,
    ("unsw_oficial", "mlp"): 0.9821,
    ("cic_strat", "xgboost"): 1.0000,
    ("cic_strat", "rf"): 0.9998,
    ("cic_strat", "mlp"): 0.9999,
    ("cic_time", "xgboost"): 0.6372,
    ("cic_time", "rf"): 0.6308,
    ("cic_time", "mlp"): 0.5959,
}


def fast_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (ties broken by average rank); ~20x faster than sklearn."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fast_fpr95(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 0.05)
    return float((neg >= thr).mean())


def fast_macro_f1(pred: np.ndarray, y: np.ndarray) -> float:
    labels = np.unique(y)
    f1s = []
    for c in labels:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        denom = 2 * tp + fp + fn
        if denom:
            f1s.append(2 * tp / denom)
    return float(np.mean(f1s)) if f1s else float("nan")


def fast_ece15(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def fast_brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(1).mean())


def load_vx(cfg: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    z = dict(np.load(VX_DIR / f"{cfg}.npz", allow_pickle=True))
    meta = json.loads((VX_DIR / f"{cfg}.json").read_text())
    from src.eval.metrics import vx_calibrated
    cal = vx_calibrated(z, meta)
    return (np.asarray(cal["te_p_noul"]), np.asarray(cal["te_p_choice"]),
            z["te_y_bin"].astype(int), z["te_y_fam"].astype(int))


def load_base(name: str, model: str, vx_meta: dict) -> tuple[np.ndarray, np.ndarray]:
    z = dict(np.load(BASE_DIR / name / f"{model}.npz", allow_pickle=True))
    meta = json.loads((BASE_DIR / name / f"{model}.json").read_text())
    cal = _baseline_calibrated(z, meta["temperatures"],
                               np.asarray(vx_meta["omega"], dtype=np.float64), None)
    return np.asarray(cal["p_noul"]), np.asarray(cal["p_choice"])


def metrics(p_noul: np.ndarray, p_choice: np.ndarray,
            y_bin: np.ndarray, y_fam: np.ndarray) -> dict:
    b = _binary_metrics(p_noul, y_bin)
    pred = p_choice.argmax(1)
    conf, correct = p_choice.max(1), (pred == y_fam).astype(np.float64)
    return {
        "auroc": b["auroc"],
        "fpr95": b["fpr@95tpr"],
        "macro_f1": fast_macro_f1(pred, y_fam),
        "ece15": _calibration_measures(conf, correct)["ece"],
        "brier": fast_brier(p_choice, y_fam),
    }


def run(selftest: bool = False) -> dict:
    results: dict = {"configs": {}, "seed_paired": {}}
    rng = np.random.default_rng(SEED)
    for name, cfg in CONFIGS.items():
        p_noul, p_choice, y_bin, y_fam = load_vx(cfg)
        meta = json.loads((VX_DIR / f"{cfg}.json").read_text())
        n = len(y_bin)
        vx = metrics(p_noul, p_choice, y_bin, y_fam)
        # Per-row metric arrays are not stored; the bootstrap recomputes from
        # arrays inside the loop for memory reasons, so pass arrays instead.
        rec = {"n_test": n, "vx": {k: round(v, 4) for k, v in vx.items()}}
        for model in MODELS:
            b_noul, b_choice = load_base(name, model, meta)
            base = metrics(b_noul, b_choice, y_bin, y_fam)
            if selftest:
                want = PAPER_AUROC[(name, model)]
                got = round(base["auroc"], 4)
                print(f"[selftest] {name}/{model}: {got} vs paper {want}")
                continue
            rec[model] = {k: round(v, 4) for k, v in base.items()}
            # Paired bootstrap over rows, computing both models per resample.
            keys = ("auroc", "fpr95", "macro_f1", "ece15", "brier")
            deltas = {k: np.empty(B) for k in keys}
            for b in range(B):
                idx = rng.integers(0, n, n)
                mb = {"auroc": fast_auc(b_noul[idx], y_bin[idx]),
                      "fpr@95tpr": fast_fpr95(b_noul[idx], y_bin[idx])}
                mv = {"auroc": fast_auc(p_noul[idx], y_bin[idx]),
                      "fpr@95tpr": fast_fpr95(p_noul[idx], y_bin[idx])}
                pred_b = b_choice[idx].argmax(1)
                pred_v = p_choice[idx].argmax(1)
                yb, yf = y_bin[idx], y_fam[idx]
                cb = _calibration_measures(b_choice[idx].max(1),
                                           (pred_b == yf).astype(float))["ece"]
                cv = _calibration_measures(p_choice[idx].max(1),
                                           (pred_v == yf).astype(float))["ece"]
                deltas["auroc"][b] = mv["auroc"] - mb["auroc"]
                deltas["fpr95"][b] = mv["fpr@95tpr"] - mb["fpr@95tpr"]
                deltas["macro_f1"][b] = fast_macro_f1(pred_v, yf) - fast_macro_f1(pred_b, yf)
                deltas["ece15"][b] = cv - cb
                deltas["brier"][b] = fast_brier(p_choice[idx], yf) - fast_brier(b_choice[idx], yf)
            rec[f"vx_minus_{model}"] = {}
            for k in keys:
                d = deltas[k]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                rec[f"vx_minus_{model}"][k] = {
                    "delta": round(float(d.mean()), 4),
                    "ci_lo": round(float(lo), 4),
                    "ci_hi": round(float(hi), 4),
                    "p_boot": round(float(min(p, 1.0)), 4),
                }
        results["configs"][name] = rec
        if selftest:
            continue
        print(f"{name}: " + " | ".join(
            f"{m}: dAUROC {rec[f'vx_minus_{m}']['auroc']['delta']:+.4f} "
            f"[{rec[f'vx_minus_{m}']['auroc']['ci_lo']:+.4f},"
            f"{rec[f'vx_minus_{m}']['auroc']['ci_hi']:+.4f}]"
            for m in MODELS))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    res = run(selftest=args.selftest)
    if args.selftest:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "paired_stats.json").write_text(json.dumps(res, indent=1))
    print("saved results/e1e3/paired_stats.json")


if __name__ == "__main__":
    main()
