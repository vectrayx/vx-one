#!/usr/bin/env python
"""E8-F2 — Non-parametric recalibration frontier for the CIC by-day split.

Motivation (external review): the paper's E8 "calibration ceiling is
discrimination" claim is supported by a PARAMETRIC (Platt/temperature)
day-self oracle, which cannot close the gap even though a saturated
score could be re-binned into few levels and calibrate perfectly.
A non-parametric oracle (isotonic / quantile binning) can drive ECE15
towards zero *by collapsing the ranking*, at the cost of AUROC. The
operationally honest object is therefore the (AUROC, ECE15) frontier
achieved by re-binning the day scores at increasing granularity.

Protocol (reuses e8_window_recal's frozen conventions):
  * Gate head = noul.wav2vec-style sigmoid score per event (te_noul_logits),
    binary label te_y_bin, capture-order rows of cic_small_time.npz
    (test = Friday-DDos n=138,145 -> Friday-PortScan n=286,467).
  * For each test day window d and each variant, recalibrate the gate:
      - temp_val   : temperature fitted on val logits (=static in paper)
      - iso_val    : isotonic fit on val logits (deployable, feasible in
                     production without day labels)
      - iso_day1lb : isotonic fit on test-day-1 labels (delayed-label
                     streaming recalibration, only available for day 2)
      - iso_day    : day-self isotonic (oracle ceiling, diagnostic only)
      - binK       : day-self quantile binning into K=1,2,4,8,16,32,64
                     confidence levels -> the (AUROC, ECE15) frontier.
  * Metrics: ECE15 (15-bin top-1), Brier, AUROC after recalibration
    (invariance is NOT assumed for the binned variants -- score ties do
    affect ranking), FPR@TPR95 with the day's own 95th-percentile of the
    POSITIVE scores within each variant (same criterion as the repo).

Output: results/exp_f2/e8_nonparam_frontier.json (+ .md table).
Runner: needs numpy/scipy/sklearn; no GPU, no parquet, no training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
NPZ = REPO / "results/outputs_seed/cic_small_time.npz"
OUT = REPO / "results/exp_f2"

DAYS = [("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX", 138145),
        ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX", 286467)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ece_bin(score: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = np.maximum(score, 1.0 - score)
    pred = (score >= 0.5).astype(np.int8)
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += abs(pred[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def brier(score: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((score - y.astype(np.float64)) ** 2))


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))

def fpr_at_tpr95(score: np.ndarray, y: np.ndarray) -> float:
    """FPR at 95% TPR using the day's own positive-score quantile."""
    pos = score[y == 1]
    if pos.size == 0:
        return float("nan")
    thr = np.quantile(pos, 0.05)
    neg = score[y == 0]
    return float((neg > thr).mean()) if neg.size else float("nan")


def temp_fit(logits: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    ln = logits.astype(np.float64)
    nll = lambda logt: float(np.mean(-(
        y * np.log(np.clip(sigmoid(ln / np.exp(logt)), 1e-7, 1 - 1e-7)) +
        (1 - y) * np.log(np.clip(1 - sigmoid(ln / np.exp(logt)), 1e-7, 1 - 1e-7)))))
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(200.)),
                                        method="bounded").x))


def iso_fit(logits: np.ndarray, y: np.ndarray) -> object:
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(sigmoid(logits.astype(np.float64)), y.astype(np.float64))
    return ir


def quantile_bin_recal(p: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Free-form (non-monotone) K-bin recalibration oracle: quantile bins of
    the raw score, predictions = within-bin empirical positive rate. The
    ECE15 ceiling object when reordering is allowed; report auroc_recalc
    to show what reordering costs."""
    if int(k) == 1:  # constant prior: kills all discrimination
        return np.full_like(p, p.mean())
    k = max(2, int(k))
    q = np.quantile(p, np.linspace(0, 1, k + 1)[1:-1])
    lvl = np.searchsorted(q, p, side="right")
    out = np.empty_like(p)
    for v in np.unique(lvl):
        m = lvl == v
        out[m] = y[m].mean()
    return out


def quantile_level_scores(p: np.ndarray, k: int) -> np.ndarray:
    """Collapse the DIRECTED attack probability into exactly k levels by
    quantile bins, keeping the mean directed probability per level."""
    k = max(2, int(k))
    if k == 1:  # constant prior predictor (kills all discrimination)
        out = np.full_like(p, p.mean())
        return out
    q = np.quantile(p, np.linspace(0, 1, k + 1)[1:-1])
    lvl = np.searchsorted(q, p, side="right")
    out = np.empty_like(p)
    for v in np.unique(lvl):
        m = lvl == v
        out[m] = p[m].mean()
    return out


def row(name: str, score: np.ndarray, y: np.ndarray, auroc_raw: float) -> dict:
    conf = np.maximum(score, 1.0 - score)
    # tie-affected AUROC of the recalibrated score (diagnostic); the ranking
    # metric of the paper is the raw-score AUROC, invariant to recalibration
    try:
        auc_r = round(auroc(score, y), 4)
    except ValueError:
        auc_r = None
    return {"variant": name,
            "auroc_raw": auroc_raw,
            "auroc_recalc": auc_r,
            "ece15": round(ece_bin(score, y), 4),
            "brier": round(brier(score, y), 4),
            "fpr@95": round(fpr_at_tpr95(score, y), 4),
            "n_unique_conf": int(np.unique(np.round(conf, 6)).size)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ, allow_pickle=True)
    n1, _ = DAYS[0]
    counts = [DAYS[0][1], DAYS[1][1]]
    assert sum(counts) == z["te_y_bin"].size, (sum(counts), z["te_y_bin"].size)
    days = []
    lo = 0
    val_logits = z["va_noul_logits"].astype(np.float64)
    val_y = z["va_y_bin"].astype(np.int64)
    for d_i, (name, n) in enumerate(DAYS):
        hi = lo + n
        logits = z["te_noul_logits"][lo:hi].astype(np.float64)
        y = z["te_y_bin"][lo:hi].astype(np.int64)
        lo = hi
        raw = sigmoid(logits)
        auroc_raw = round(auroc(raw, y), 4)
        rows: list[dict] = []
        # static (paper's current serving variant)
        T = temp_fit(val_logits, val_y)
        rows.append(row("temp_val", sigmoid(logits / T), y, auroc_raw))
        # deployable non-parametric variants
        ir_val = iso_fit(val_logits, val_y)
        rows.append(row("iso_val", ir_val.predict(raw), y, auroc_raw))
        if d_i == 1:
            day1_logits = z["te_noul_logits"][:counts[0]].astype(np.float64)
            day1_y = z["te_y_bin"][:counts[0]].astype(np.int64)
            ir_d1 = iso_fit(day1_logits, day1_y)
            rows.append(row("iso_day1lb(delayed)", ir_d1.predict(raw), y, auroc_raw))
        # oracle ceiling: day-self isotonic
        ir_day = iso_fit(logits, y)
        rows.append(row("iso_day(oracle)", ir_day.predict(raw), y, auroc_raw))
        # free-form bin oracle: K levels fit to the day's own labels
        for k in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            rows.append(row(f"binfit{k}",
                            quantile_bin_recal(raw, y.astype(np.float64), k), y, auroc_raw))
        days.append({"day": name, "n": int(n), "prevalence": float(y.mean()),
                     "auroc_raw": auroc_raw, "rows": rows})
        print(f"[e8NonParam] day={name} n={n} prev={y.mean():.3f} auroc_raw={auroc_raw}")
        for r in rows:
            print(f"  {r['variant']:>20s}  auroc_recalc={r['auroc_recalc']}  ece15={r['ece15']:.4f}"
                  f"  brier={r['brier']:.4f}  fpr@95={r['fpr@95']:.4f}"
                  f"  levels={r['n_unique_conf']}")

    out = {"protocol": {"npz": str(NPZ), "windows": [d[0] for d in DAYS],
                        "head": "gate (noul)", "ece15-bins": 15,
                        "binning": "day-self quantile bins; iso=sklearn isotonic"},
           "days": days}
    (OUT / "e8_nonparam_frontier.json").write_text(json.dumps(out, indent=1))
    # markdown table
    md = ["| day | variant | AUROC(raw) | AUROC(recalc) | ECE15 | Brier | FPR@95 | levels |",
          "|---|---|---|---|---|---|---|---|"]
    for d in days:
        for r in d["rows"]:
            md.append(f"| {d['day'][:24]} | {r['variant']} | {d['auroc_raw']} | {r['auroc_recalc']} "
                      f"| {r['ece15']} | {r['brier']} | {r['fpr@95']} | {r['n_unique_conf']} |")
    (OUT / "e8_nonparam_frontier.md").write_text("\n".join(md) + "\n")
    print(f"[e8NonParam] -> {OUT/'e8_nonparam_frontier.json'}")


if __name__ == "__main__":
    sys.exit(main())
