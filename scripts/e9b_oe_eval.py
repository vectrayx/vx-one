#!/usr/bin/env python
"""E9b evaluator under the paper's calibration contract (vx_calibrated).

Why this script exists
----------------------
The original E9b sweep applied temperature scaling only, while E2 (and every
table in the paper) applies the engine contract: temperature scaling plus the
omega prior de-tilt on the choice head. Mixing the two produces different
abstention numbers for the *same* checkpoint (e.g. UNSW/dos: 0.028 under the
contract vs 0.710 temperature-only). This evaluator computes every E9b metric
with `vx_calibrated`, so reference and outlier-exposure (OE) rows share one
estimator.

Metrics (per LOAO fold, rows of the excluded family)
----------------------------------------------------
- gate_auroc / gate_fpr95: noul gate on excluded-vs-normal rows.
- abstention at tau in {0.3, 0.5, 0.7, 0.9}: max calibrated choice prob < tau.
- false_att: attributed (>= tau) to a known attack family.
- normal_absorb: attributed (>= tau) to the normal class.
- known_macro_f1: macro-F1 on the non-excluded rows (control).
- median_maxprob: median calibrated max choice prob on excluded rows.

Usage
-----
    python scripts/e9b_oe_eval.py            # full sweep, writes artifacts
    python scripts/e9b_oe_eval.py --selftest # consistency check against E2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.eval.metrics import vx_calibrated  # noqa: E402
from src.eval.run_e2 import eval_loao_fold  # noqa: E402

REF_DIR = Path("/home/jsantillana/projects/paper/results/outputs")
OE_DIR = REPO / "results/outputs_oe"
SWEEP_DIR = REPO / "results/outputs_oe_sweep"
SWEEP_JSON_DIR = REPO / "results/e9b_oe/meta/sweep_json"
OE_JSON_DIR = REPO / "results/e9b_oe/meta/oe_json"
OUT = REPO / "results/e9b_oe"

UNSW_FOLDS = ["analysis", "backdoor", "dos", "exploitation", "fuzzing",
              "generic", "recon", "shellcode", "worms"]
CIC_FOLDS = ["botnet", "bruteforce", "dos_ddos", "exploitation",
             "infiltration", "recon", "webattack"]
TAUS = (0.3, 0.5, 0.7, 0.9)

# variant tag -> filename suffix
UNSW_VARIANTS = {
    "oe0.2": "_oe0.2",
    "oe0.5": "_oe0.5",
    "oe1.0": "_oe1.0",
    "noise0.5": "_oe0.5noise",
}
CIC_VARIANTS = {"oe0.5": "_oe0.5"}


def _load_pair(npz_path: Path, json_path: Path) -> tuple[dict, dict] | None:
    if not npz_path.exists() or not json_path.exists():
        return None
    return dict(np.load(npz_path, allow_pickle=True)), json.loads(json_path.read_text())


def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def _fpr_at_tpr95(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score)
    ys = y[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    idx = int(np.searchsorted(tpr, 0.95, side="left"))
    return float(fpr[min(idx, len(fpr) - 1)])


def _macro_f1(y_true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def fold_metrics(npz: dict, meta: dict) -> dict:
    """All E9b metrics for one fold under the vx_calibrated contract."""
    cal = vx_calibrated(npz, meta)
    p_noul = np.asarray(cal["te_p_noul"], dtype=np.float64)
    p_choice = np.asarray(cal["te_p_choice"], dtype=np.float64)
    y_fam = npz["te_y_fam"].astype(np.int64)
    excl = y_fam == -1
    # Class indices come from the declared candidate list, never from the
    # per-row family-name array (whose positions are rows, not classes).
    candidates = [str(c) for c in meta["candidates"]]
    normal_idx = candidates.index("normal")
    is_normal = y_fam == normal_idx
    known = ~excl

    mask = excl | is_normal
    gate_auroc = _auroc(p_noul[mask], excl[mask].astype(int))
    gate_fpr95 = _fpr_at_tpr95(p_noul[mask], excl[mask].astype(int))

    max_prob = p_choice.max(axis=1)
    arg = p_choice.argmax(axis=1)
    attack_idx = [i for i, n in enumerate(candidates) if n != "normal"]

    sweep = {}
    for tau in TAUS:
        over = max_prob[excl] >= tau
        sweep[str(tau)] = {
            "abst": round(float((~over).mean()), 4),
            "false_att": round(float((over & np.isin(arg[excl], attack_idx)).mean()), 4),
            "normal_absorb": round(float((over & (arg[excl] == normal_idx)).mean()), 4),
        }

    return {
        "gate_auroc": round(gate_auroc, 4),
        "gate_fpr95": round(gate_fpr95, 4),
        "sweep": sweep,
        "known_macroF1": round(_macro_f1(y_fam[known], arg[known]), 4),
        "median_maxprob_excl": round(float(np.median(max_prob[excl])), 4),
        "n_excluded": int(excl.sum()),
    }


def _materialize(npz_path: Path, json_path: Path) -> Path:
    """vx_calibrated needs <stem>.json next to <stem>.npz."""
    if json_path.parent != npz_path.parent or json_path.stem != npz_path.stem:
        tmp = Path(tempfile.mkdtemp()) / npz_path.stem
        shutil.copy(npz_path, tmp.with_suffix(".npz"))
        shutil.copy(json_path, tmp.with_suffix(".json"))
        return tmp.with_suffix(".npz")
    return npz_path


def evaluate_dataset(prefix: str, folds: list[str], variants: dict[str, str],
                     variant_npz_dir: Path, variant_json_dir: Path) -> dict:
    out: dict = {"folds": {}}
    for fam in folds:
        stem = f"{prefix}_loao_{fam}"
        ref = _load_pair(REF_DIR / f"{stem}.npz", REF_DIR / f"{stem}.json")
        if ref is None:
            continue
        rec = {"ref": fold_metrics(*ref)}
        for tag, suffix in variants.items():
            npz_p = variant_npz_dir / f"{stem}{suffix}.npz"
            json_p = variant_json_dir / f"{stem}{suffix}.json"
            pair = _load_pair(npz_p, json_p)
            rec[tag] = fold_metrics(*pair) if pair else None
        out["folds"][fam] = rec
    return out


def summarize(block: dict, tags: list[str]) -> dict:
    import statistics as st

    def mean_metric(tag: str, path: list[str]):
        vals = []
        for rec in block["folds"].values():
            node = rec.get(tag)
            if node is None:
                continue
            for key in path:
                node = node[key]
            vals.append(node)
        return round(st.mean(vals), 4) if vals else None

    summary = {}
    for tag in ["ref"] + tags:
        summary[tag] = {
            "abst@0.5": mean_metric(tag, ["sweep", "0.5", "abst"]),
            "false_att@0.5": mean_metric(tag, ["sweep", "0.5", "false_att"]),
            "normal_absorb@0.5": mean_metric(tag, ["sweep", "0.5", "normal_absorb"]),
            "known_macroF1": mean_metric(tag, ["known_macroF1"]),
            "gate_auroc": mean_metric(tag, ["gate_auroc"]),
        }
    return summary


def _selftest() -> int:
    """Reference folds must reproduce E2's abstention within rounding."""
    expected = {"dos": 0.0276, "exploitation": 0.0819, "analysis": 0.9232}
    failures = 0
    for fam, want in expected.items():
        pair = _load_pair(REF_DIR / f"unsw_small_oficial_loao_{fam}.npz",
                          REF_DIR / f"unsw_small_oficial_loao_{fam}.json")
        got = fold_metrics(*pair)["sweep"]["0.5"]["abst"]
        ok = abs(got - want) < 0.001
        failures += not ok
        print(f"[selftest] {fam}: contract={got:.4f} E2={want:.4f} {'OK' if ok else 'MISMATCH'}")
    return failures


def write_markdown(payload: dict, path: Path) -> None:
    """Human-readable companion of the JSON artifact."""
    lines = ["# E9b - outlier exposure under the paper's calibration contract", "",
             "Estimator: `vx_calibrated` (temperature + omega de-tilt), the same",
             "contract as E2. Reference and OE rows therefore share one estimator.",
             "Supersedes the earlier temperature-only readout.", ""]
    for dataset, tag in (("unsw", "oe0.5"), ("cic", "oe0.5")):
        folds = payload[dataset]["folds"]
        if not folds:
            continue
        lines += [f"## {dataset.upper()}", "",
                  "| fold | abst ref | abst oe | false-att ref | false-att oe | absorb ref | absorb oe | F1 ref | F1 oe | gate ref | gate oe |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for fam, rec in folds.items():
            r, o = rec["ref"], rec[tag]
            lines.append(
                f"| {fam} | {r['sweep']['0.5']['abst']:.3f} | {o['sweep']['0.5']['abst']:.3f} "
                f"| {r['sweep']['0.5']['false_att']:.3f} | {o['sweep']['0.5']['false_att']:.3f} "
                f"| {r['sweep']['0.5']['normal_absorb']:.3f} | {o['sweep']['0.5']['normal_absorb']:.3f} "
                f"| {r['known_macroF1']:.3f} | {o['known_macroF1']:.3f} "
                f"| {r['gate_auroc']:.3f} | {o['gate_auroc']:.3f} |")
        summary = payload["summary"][dataset]
        lines += ["", "Means: " + "; ".join(
            f"{k} abst {v['abst@0.5']:.3f}, F1 {v['known_macroF1']:.3f}, "
            f"gate {v['gate_auroc']:.3f}" for k, v in summary.items()), ""]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())

    OUT.mkdir(parents=True, exist_ok=True)
    unsw = evaluate_dataset("unsw_small_oficial", UNSW_FOLDS, UNSW_VARIANTS,
                            SWEEP_DIR, SWEEP_JSON_DIR)
    cic = evaluate_dataset("cic_small_strat", CIC_FOLDS, CIC_VARIANTS,
                           OE_DIR, OE_JSON_DIR)
    # UNSW oe0.5 lives in outputs_oe (the original E9b run)
    for fam, rec in unsw["folds"].items():
        pair = _load_pair(OE_DIR / f"unsw_small_oficial_loao_{fam}_oe0.5.npz",
                          OE_JSON_DIR / f"unsw_small_oficial_loao_{fam}_oe0.5.json")
        rec["oe0.5"] = fold_metrics(*pair) if pair else None

    payload = {
        "estimator": "vx_calibrated (temperature + omega de-tilt), E2 contract",
        "unsw": unsw,
        "cic": cic,
        "summary": {
            "unsw": summarize(unsw, ["oe0.2", "oe0.5", "oe1.0", "noise0.5"]),
            "cic": summarize(cic, ["oe0.5"]),
        },
    }
    (OUT / "e9b_oe_contract.json").write_text(json.dumps(payload, indent=1))
    write_markdown(payload, OUT / "e9b_oe.md")
    print(json.dumps(payload["summary"], indent=1))


if __name__ == "__main__":
    main()
