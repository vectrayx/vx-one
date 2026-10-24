"""A7 ablation: dedicated gate vs gate derived from a single family softmax.

The typed reference exposes three heads (gate, choice, score). A7 keeps the
trunk, the family softmax (same weighted CE and analytic de-tilt), and the
CORN score head, but removes the dedicated gate: the gate is derived as
``1 - p(normal)`` from the choice distribution. This isolates the *typing*
of the gate primitive from multi-task sharing.

For each seed the script pairs the A7 run with the typed reference run of the
same seed, reports E1/E3 metrics, and estimates row-paired bootstrap
intervals (B=300) for the deltas (A7 - typed). Verdicts use the seed-level
criterion (mean over seeds), not the row bootstrap.

Usage:
    python scripts/a7_derived_gate_eval.py --a7 DIR --typed DIR --out DIR
    python scripts/a7_derived_gate_eval.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.metrics import eval_vx_run, vx_calibrated  # noqa: E402
SEEDS = (42, 43, 44)
B = 300
RNG_SEED = 20261023


def fast_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC; ties get average ranks (vectorized, O(n log n))."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    n = len(s)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], n]
    avg_rank = (starts + 1 + ends) / 2.0  # 1-based average rank per tie group
    ranks_sorted = np.repeat(avg_rank, ends - starts)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    n_pos = float(y.sum())
    n_neg = float(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_fpr95(score: np.ndarray, y: np.ndarray) -> float:
    """FPR at the threshold that reaches >=95% TPR (lowest such FPR)."""
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((1 - y).sum()), 1)
    ok = np.where(tp >= 0.95 * n_pos)[0]
    if len(ok) == 0:
        return 1.0
    k = ok[0]
    return float(fp[k] / n_neg)


def fast_macro_f1(pred: np.ndarray, y: np.ndarray) -> float:
    """Macro-F1 over the classes present in either vector."""
    labels = np.union1d(np.unique(y), np.unique(pred))
    f1s = []
    for c in labels:
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def fast_ece15(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def load_run(npz_path: Path) -> dict:
    npz = dict(np.load(npz_path, allow_pickle=True))
    meta = json.loads(Path(str(npz_path).replace(".npz", ".json")).read_text())
    cal = vx_calibrated(npz, meta)
    y_bin = npz["te_y_bin"].astype(np.int64)
    y_fam = npz["te_y_fam"].astype(np.int64)
    fam_mask = y_fam >= 0
    return {
        "path": npz_path,
        "p_noul": cal["te_p_noul"].astype(np.float64),
        "p_choice": cal["te_p_choice"].astype(np.float64),
        "y_bin": y_bin,
        "y_fam": y_fam,
        "fam_mask": fam_mask,
        "temps": meta.get("temperatures", {}),
        "seed": meta.get("seed"),
        "n": int(len(y_bin)),
    }


def point_metrics(run: dict) -> dict:
    """Use the canonical E1/E3 evaluator so numbers match Table 6 exactly."""
    te = eval_vx_run(run["path"])["te"]
    return {
        "auroc": te["noul"]["e1"]["auroc"],
        "fpr95": te["noul"]["e1"]["fpr@95tpr"],
        "macro_f1": te["choice"]["e1"]["macro_f1"],
        "ece15": te["noul"]["e3"]["cal"]["ece"],
        "ece_choice": te["choice"]["e3"]["cal"]["ece"],
    }


def bootstrap_delta(a7: dict, typed: dict, rng: np.random.Generator) -> dict:
    """Row-paired percentile CI for (A7 - typed) on the same resampled rows."""
    n = a7["n"]
    assert typed["n"] == n, "A7 and typed runs must share the test rows"
    assert np.array_equal(a7["y_bin"], typed["y_bin"])
    deltas = {"auroc": [], "fpr95": [], "macro_f1": [], "ece15": []}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = a7["y_bin"][idx]
        if yb.min() == yb.max():
            continue
        pa = a7["p_noul"][idx]
        pt = typed["p_noul"][idx]
        deltas["auroc"].append(fast_auc(pa, yb) - fast_auc(pt, yb))
        deltas["fpr95"].append(fast_fpr95(pa, yb) - fast_fpr95(pt, yb))
        deltas["ece15"].append(fast_ece15(pa, yb) - fast_ece15(pt, yb))
        fm = a7["fam_mask"][idx]
        if fm.any():
            ya = a7["y_fam"][idx][fm]
            deltas["macro_f1"].append(
                fast_macro_f1(a7["p_choice"][idx][fm].argmax(1), ya)
                - fast_macro_f1(typed["p_choice"][idx][fm].argmax(1), ya))
    out = {}
    for k, v in deltas.items():
        if not v:
            out[k] = {"delta": None, "lo": None, "hi": None}
            continue
        arr = np.asarray(v)
        out[k] = {"delta": float(arr.mean()),
                  "lo": float(np.percentile(arr, 2.5)),
                  "hi": float(np.percentile(arr, 97.5))}
    return out


def _seed_of(path: Path) -> int:
    m = re.search(r"_s(\d+)$", path.stem)
    return int(m.group(1)) if m else 42


def evaluate(a7_dir: Path, typed_dir: Path) -> dict:
    runs = {}
    for npz in sorted(a7_dir.glob("*_abl_a7*.npz")):
        seed = _seed_of(npz)
        base_stem = npz.stem.replace("_abl_a7", "")
        base = typed_dir / f"{base_stem}.npz"
        if not base.exists():
            runs[npz.stem] = {"error": f"missing typed baseline {base.name}"}
            continue
        a7 = load_run(npz)
        typed = load_run(base)
        rng = np.random.default_rng(RNG_SEED + seed)
        runs[npz.stem] = {
            "seed": seed,
            "a7": point_metrics(a7),
            "typed": point_metrics(typed),
            "bootstrap": bootstrap_delta(a7, typed, rng),
        }

    ok = [v for v in runs.values() if "error" not in v]
    by_source: dict[str, list[dict]] = {}
    for stem, v in runs.items():
        if "error" in v:
            continue
        src = stem.split("_", 1)[0]
        by_source.setdefault(src, []).append(v)
    summary = {}
    for src, vals in by_source.items():
        summary[src] = {}
        for k in ("auroc", "fpr95", "macro_f1", "ece15", "ece_choice"):
            a = np.array([v["a7"][k] for v in vals])
            t = np.array([v["typed"][k] for v in vals])
            d = a - t
            summary[src][k] = {
                "a7_mean": float(a.mean()),
                "a7_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
                "typed_mean": float(t.mean()),
                "typed_sd": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
                "delta_mean": float(d.mean()),
                "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                "wins": int((d < 0).sum()) if k in ("fpr95", "ece15", "ece_choice")
                        else int((d > 0).sum()),
                "n_seeds": int(len(d)),
            }
    return {"runs": runs, "summary": summary}


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def write_report(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a7.json").write_text(json.dumps(res, indent=1))
    lines = [
        "# A7 - dedicated gate vs gate derived from the family softmax",
        "",
        "Paired by seed and by test rows. Delta = A7 - typed; negative is better",
        "for FPR@95 and ECE. Bootstrap intervals are descriptive (B=300).",
        "",
        "| run | seed | AUROC a7 | AUROC typed | dAUROC [95%] | m-F1 a7 | m-F1 typed | dF1 [95%] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for stem, v in res["runs"].items():
        if "error" in v:
            lines.append(f"| {stem} | — | — | — | — | — | — | — |")
            continue
        b = v["bootstrap"]
        lines.append(
            f"| {stem} | {v['seed']} | {_fmt(v['a7']['auroc'])} | {_fmt(v['typed']['auroc'])} | "
            f"{_fmt(b['auroc']['delta'])} [{_fmt(b['auroc']['lo'])},{_fmt(b['auroc']['hi'])}] | "
            f"{_fmt(v['a7']['macro_f1'])} | {_fmt(v['typed']['macro_f1'])} | "
            f"{_fmt(b['macro_f1']['delta'])} [{_fmt(b['macro_f1']['lo'])},{_fmt(b['macro_f1']['hi'])}] |")
    for src, summ in res["summary"].items():
        lines += ["", f"## Seed summary - {src}", "",
                  "| metric | A7 mean±sd | typed mean±sd | delta mean±sd | wins (better) |",
                  "|---|---|---|---|---|"]
        for k, v in summ.items():
            lines.append(f"| {k} | {v['a7_mean']:.4f}±{v['a7_sd']:.4f} | "
                         f"{v['typed_mean']:.4f}±{v['typed_sd']:.4f} | "
                         f"{v['delta_mean']:+.4f}±{v['delta_sd']:.4f} | "
                         f"{v['wins']}/{v['n_seeds']} |")
    (out_dir / "a7.md").write_text("\n".join(lines) + "\n")


def selftest() -> None:
    """Synthetic checks: fast metrics match sklearn and self-delta is zero."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.3).astype(np.int64)
    score = rng.normal(size=n) + 1.5 * y
    assert abs(fast_auc(score, y) - roc_auc_score(y, score)) < 1e-12
    assert fast_fpr95(score, y) <= 1.0
    fam = rng.integers(0, 4, size=n)
    p_choice = np.eye(4)[fam] * 0.8 + 0.05
    run = {"p_noul": score, "p_choice": p_choice, "y_bin": y,
           "y_fam": fam, "fam_mask": np.ones(n, dtype=bool), "n": n}
    b = bootstrap_delta(run, run, np.random.default_rng(RNG_SEED))
    for k, v in b.items():
        if v["delta"] is not None:
            assert abs(v["delta"]) < 1e-12, (k, v)
    print("[selftest] synthetic AUROC matches sklearn; A7 vs itself "
          "gives zero deltas - OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7", default="results/e6_a7")
    ap.add_argument("--typed", default="results/outputs_seed")
    ap.add_argument("--out", default="results/e6_a7")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    res = evaluate(Path(args.a7), Path(args.typed))
    write_report(res, Path(args.out))
    for src, summ in res["summary"].items():
        for k, v in summ.items():
            print(f"[a7:{src}] {k}: A7 {v['a7_mean']:.4f} vs typed "
                  f"{v['typed_mean']:.4f} (delta {v['delta_mean']:+.4f}, "
                  f"wins {v['wins']}/{v['n_seeds']})")


if __name__ == "__main__":
    main()
