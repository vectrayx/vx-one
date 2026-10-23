#!/usr/bin/env python
"""E9-F2 — Non-parametric (isotonic / binning) oracle bound on RETRAINED scores.

Companion to e8_nonparam.py. The paper's E9 paragraph quotes the oracle
bound collapse 0.445 -> 0.117 (one pass) / 0.100 (two passes) computed with
the PARAMETRIC Platt family. A reviewer correctly objects that a
non-parametric day-self recalibrator is the true ceiling object. This
script computes, for each retrained artifact (E9 one-pass ep / two-pass
ep2 / four-pass ep4 on the CIC PortScan day), the day-self isotonic and
quantile-binning oracle bounds, plus the constant-prior floor.

Usage: paper-novel/.venv/bin/python scripts/e9_nonparam_oracle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e8_nonparam import (  # noqa: E402
    DAYS, OUT, NPZ, ece_bin, brier, auroc, iso_fit, quantile_bin_recal,
)

REPO = Path(__file__).resolve().parent.parent
E9 = REPO / "results/e9_retrain"
DAY2 = DAYS[1][0]


def row(name: str, score: np.ndarray, y: np.ndarray, auroc_raw: float) -> dict:
    conf = np.maximum(score, 1.0 - score)
    return {"variant": name,
            "auroc_raw": auroc_raw,
            "ece15": round(ece_bin(score, y), 4),
            "brier": round(brier(score, y), 4),
            "n_unique_conf": int(np.unique(np.round(conf, 6)).size)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lo = DAYS[0][1]
    hi = lo + DAYS[1][1]
    results = []
    for tag, npz_name, key in (("static (temp-val serving)", None, None),
                               ("e9 1-pass", "cic_small_time_friday.npz", "te2_noul"),
                               ("e9 2-pass", "cic_small_time_fridayep2.npz", "te2_noul"),
                               ("e9 4-pass", "cic_small_time_fridayep4.npz", "te2_noul")):
        if npz_name is None:
            z = np.load(NPZ, allow_pickle=True)
            logits_all = z["te_noul_logits"]
            # static serving applies the val-fitted temperature (E8 protocol)
            from e8_nonparam import temp_fit
            val_logits = z["va_noul_logits"].astype(np.float64)
            val_y = z["va_y_bin"].astype(np.int64)
            T = temp_fit(val_logits, val_y)
            logits = logits_all[lo:hi].astype(np.float64) / T
        else:
            z = np.load(E9 / npz_name, allow_pickle=True)
            logits = z[key].astype(np.float64)  # te2_* is ALREADY the day-2 window
        y_all = z["te_y_bin"] if npz_name is None else z["te2_y_bin"]
        y = (y_all[lo:hi].astype(np.int64) if npz_name is None
             else y_all.astype(np.int64))
        raw = 1.0 / (1.0 + np.exp(-logits))
        auroc_raw = round(auroc(raw, y), 4)
        rows = [row("raw", raw, y, auroc_raw),
                row("constant prior (1 level)", np.full_like(raw, raw.mean()), y, auroc_raw),
                row("iso_day(oracle)", iso_fit(logits, y).predict(raw), y, auroc_raw)]
        for k in (4, 16, 64, 256):
            rows.append(row(f"binfit{k}", quantile_bin_recal(raw, y.astype(np.float64), k), y, auroc_raw))
        results.append({"tag": tag, "npz": str(E9 / npz_name) if npz_name else str(NPZ), "auroc_raw": auroc_raw,
                        "rows": rows})
        print(f"\n[{tag}] day={DAY2} auroc_raw={auroc_raw}")
        for r in rows:
            print(f"  {r['variant']:>26s} ece15={r['ece15']:.4f} brier={r['brier']:.4f}"
                  f" levels={r['n_unique_conf']}")
    (OUT / "e9_nonparam_oracle.json").write_text(json.dumps(results, indent=1))
    print(f"\n[done] -> {OUT/'e9_nonparam_oracle.json'}")


if __name__ == "__main__":
    sys.exit(main())
