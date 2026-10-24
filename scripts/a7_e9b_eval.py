#!/usr/bin/env python
"""E9b under outlier exposure (OE): derived gate (a7) vs dedicated gate.

E9b asks whether outlier exposure teaches abstention on the excluded family.
The typed model answers with a dedicated binary gate plus a choice head; the
a7 ablation removes the gate and derives it as ``1 - p(normal)`` from the
same family softmax. Both runs see the same OE pool (train rows of the
excluded family with a uniform target on the choice head), so the comparison
isolates the gate's typing.

Metrics are the E9b ones, evaluated at tau=0.5 with the ``vx_calibrated``
contract: abstention, false attribution, benign absorption, gate AUROC,
known-class macro-F1, and the median max choice probability on excluded rows.
Deltas are a7 - typed; the direction of "better" depends on the metric and is
declared in ``DIRECTION``. The fold-level exact Wilcoxon is the verdict.

Usage
-----
    python scripts/a7_e9b_eval.py --selftest
    python scripts/a7_e9b_eval.py \
        --a7 results/e6_a7/oe \
        --typed results/outputs_oe \
        --typed-meta results/e9b_oe/meta/oe_json \
        --out results/e6_a7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.e9b_oe_eval import UNSW_FOLDS, _load_pair, fold_metrics  # noqa: E402

TAU = "0.5"

# metric -> "higher" or "lower" is better
DIRECTION = {
    "abst": "higher",
    "false_att": "lower",
    "normal_absorb": "lower",
    "gate_auroc": "higher",
    "known_macroF1": "higher",
    "median_maxprob_excl": "lower",
}


def _metrics(npz: dict, meta: dict) -> dict:
    m = fold_metrics(npz, meta)
    s = m["sweep"][TAU]
    return {
        "abst": s["abst"],
        "false_att": s["false_att"],
        "normal_absorb": s["normal_absorb"],
        "gate_auroc": m["gate_auroc"],
        "known_macroF1": m["known_macroF1"],
        "median_maxprob_excl": m["median_maxprob_excl"],
    }


def _better(metric: str, delta: float) -> bool:
    return delta > 0 if DIRECTION[metric] == "higher" else delta < 0


def _summary(rows: list[dict]) -> dict:
    from scipy.stats import wilcoxon

    out = {}
    for metric in DIRECTION:
        a = np.array([r["a7"][metric] for r in rows])
        t = np.array([r["typed"][metric] for r in rows])
        d = a - t
        rec = {
            "a7_mean": round(float(a.mean()), 4),
            "typed_mean": round(float(t.mean()), 4),
            "delta_mean": round(float(d.mean()), 4),
            "delta_sd": round(float(d.std(ddof=1)) if len(d) > 1 else 0.0, 4),
            "a7_better": int(sum(_better(metric, x) for x in d)),
            "n": int(len(d)),
        }
        try:
            rec["wilcoxon_p"] = round(float(wilcoxon(d).pvalue), 4)
        except ValueError:
            rec["wilcoxon_p"] = None
        out[metric] = rec
    return out


def evaluate(a7_dir: Path, typed_dir: Path, typed_meta: Path) -> dict:
    rows, folds = [], {}
    for fam in UNSW_FOLDS:
        stem = f"unsw_small_oficial_loao_{fam}"
        a7 = _load_pair(a7_dir / f"{stem}_abl_a7_oe0.5.npz",
                        a7_dir / f"{stem}_abl_a7_oe0.5.json")
        ref = _load_pair(typed_dir / f"{stem}_oe0.5.npz",
                         typed_meta / f"{stem}_oe0.5.json")
        if a7 is None or ref is None:
            continue
        ma, mr = _metrics(*a7), _metrics(*ref)
        delta = {k: round(ma[k] - mr[k], 4) for k in DIRECTION}
        rec = {"a7": ma, "typed": mr, "delta": delta}
        folds[fam] = rec
        rows.append(rec)
    return {"folds": folds, "summary": _summary(rows)}


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# E9b + a7 - derived gate vs dedicated gate under outlier exposure",
        "",
        "Both models see the excluded family as an OE pool with a uniform",
        "target on the choice head. Delta = a7 - typed at tau=0.5; the",
        "direction of improvement is metric-specific (see the summary).",
        "",
        "## Abstention (higher is better)",
        "",
        "| fold | abst a7 | abst typed | delta | false-att a7 | false-att typed | delta | absorb a7 | absorb typed | delta |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam, r in payload["folds"].items():
        a, t, d = r["a7"], r["typed"], r["delta"]
        lines.append(
            f"| {fam} | {a['abst']:.3f} | {t['abst']:.3f} | {d['abst']:+.3f} "
            f"| {a['false_att']:.3f} | {t['false_att']:.3f} | {d['false_att']:+.3f} "
            f"| {a['normal_absorb']:.3f} | {t['normal_absorb']:.3f} | {d['normal_absorb']:+.3f} |")
    lines += [
        "",
        "## Detection and control (higher is better)",
        "",
        "| fold | gate a7 | gate typed | delta | F1 a7 | F1 typed | delta | median maxp a7 | typed | delta |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fam, r in payload["folds"].items():
        a, t, d = r["a7"], r["typed"], r["delta"]
        lines.append(
            f"| {fam} | {a['gate_auroc']:.3f} | {t['gate_auroc']:.3f} | {d['gate_auroc']:+.3f} "
            f"| {a['known_macroF1']:.3f} | {t['known_macroF1']:.3f} | {d['known_macroF1']:+.3f} "
            f"| {a['median_maxprob_excl']:.3f} | {t['median_maxprob_excl']:.3f} "
            f"| {d['median_maxprob_excl']:+.3f} |")
    lines += [
        "",
        "## Summary",
        "",
        "| metric | a7 | typed | delta (mean +- sd) | a7 better | Wilcoxon p |",
        "|---|---|---|---|---|---|",
    ]
    for metric, s in payload["summary"].items():
        lines.append(
            f"| {metric} | {s['a7_mean']:.3f} | {s['typed_mean']:.3f} "
            f"| {s['delta_mean']:+.3f} +- {s['delta_sd']:.3f} "
            f"| {s['a7_better']}/{s['n']} | {s['wilcoxon_p']} |")
    path.write_text("\n".join(lines) + "\n")


def selftest() -> int:
    """Pairing logic must yield zero deltas when both sides are the same run."""
    stem = "unsw_small_oficial_loao_dos"
    pair = _load_pair(REPO / "results/outputs_oe" / f"{stem}_oe0.5.npz",
                      REPO / "results/e9b_oe/meta/oe_json" / f"{stem}_oe0.5.json")
    if pair is None:
        print("[selftest] missing typed dos OE pair")
        return 1
    m = _metrics(*pair)
    rec = {"a7": m, "typed": m}
    d = {k: round(rec["a7"][k] - rec["typed"][k], 6) for k in DIRECTION}
    ok = all(abs(v) < 1e-9 for v in d.values())
    print(f"[selftest] a7 vs itself: max |delta|={max(abs(v) for v in d.values()):.1e} "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7", default="results/e6_a7/oe")
    ap.add_argument("--typed", default="results/outputs_oe")
    ap.add_argument("--typed-meta", default="results/e9b_oe/meta/oe_json")
    ap.add_argument("--out", default="results/e6_a7")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())

    payload = evaluate(Path(args.a7), Path(args.typed), Path(args.typed_meta))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "oe_a7.json").write_text(json.dumps(payload, indent=1))
    write_markdown(payload, out / "oe_a7.md")
    for metric, s in payload["summary"].items():
        print(f"[oe-a7] {metric}: a7={s['a7_mean']:+.4f} typed={s['typed_mean']:+.4f} "
              f"delta={s['delta_mean']:+.4f} better={s['a7_better']}/{s['n']} "
              f"p={s['wilcoxon_p']}")


if __name__ == "__main__":
    main()
