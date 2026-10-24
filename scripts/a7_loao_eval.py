"""A7 under LOAO: does a derived gate transfer to unseen families?

Each LOAO fold trains without one family and tests the gate one-vs-normal on
the excluded rows. The typed reference and A7 share folds and seed, so every
fold is a paired comparison: dedicated gate vs gate derived from the family
softmax (1 - p(normal)).

Usage:
    python scripts/a7_loao_eval.py --a7 DIR --typed DIR --out DIR
    python scripts/a7_loao_eval.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.run_e2 import eval_loao_fold  # noqa: E402
from scripts.a7_derived_gate_eval import fast_auc  # noqa: E402

B = 300
RNG_SEED = 20261023


def load_fold(npz_path: Path) -> dict:
    npz = dict(np.load(npz_path, allow_pickle=True))
    meta = json.loads(Path(str(npz_path).replace(".npz", ".json")).read_text())
    names = np.asarray(npz["te_fam_names"], dtype=object)
    excl = meta["exclude"]
    mask = (names == excl) | (names == "normal")
    p_n = 1.0 / (1.0 + np.exp(-npz["te_noul_logits"].astype(np.float64)
                              / meta["temperatures"]["noul"]))
    return {"excl": excl, "p_n": p_n, "mask": mask,
            "y": (names == excl).astype(np.int64), "n": int(mask.sum())}


def bootstrap_delta(a7: dict, typed: dict, rng: np.random.Generator) -> dict:
    assert a7["excl"] == typed["excl"]
    assert np.array_equal(a7["mask"], typed["mask"])
    idx_all = np.flatnonzero(a7["mask"])
    deltas = []
    for _ in range(B):
        idx = rng.choice(idx_all, len(idx_all), replace=True)
        y = a7["y"][idx]
        if y.min() == y.max():
            continue
        deltas.append(fast_auc(a7["p_n"][idx], y) - fast_auc(typed["p_n"][idx], y))
    arr = np.asarray(deltas)
    return {"delta": float(arr.mean()),
            "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)),
            "n_boot": int(len(arr))}


def evaluate(a7_dir: Path, typed_dir: Path) -> dict:
    runs = {}
    for npz in sorted(a7_dir.glob("*_loao_*_abl_a7.npz")):
        fold = npz.stem.split("_loao_")[1].replace("_abl_a7", "")
        source = npz.stem.split("_", 1)[0]
        key = f"{source}/{fold}"
        typed = typed_dir / npz.name.replace("_abl_a7", "")
        if not typed.exists():
            runs[key] = {"source": source, "fold": fold,
                         "error": f"missing typed fold {typed.name}"}
            continue
        a7 = load_fold(npz)
        ref = load_fold(typed)
        rng = np.random.default_rng(RNG_SEED + zlib.crc32(key.encode()) % 10_000)
        runs[key] = {
            "source": source, "fold": fold,
            "a7": eval_loao_fold(npz)["noul_on_excluded"],
            "typed": eval_loao_fold(typed)["noul_on_excluded"],
            "bootstrap": bootstrap_delta(a7, ref, rng),
        }

    def _summary(vals: list[dict]) -> dict:
        d = np.array([v["a7"]["auroc"] - v["typed"]["auroc"] for v in vals])
        out = {
            "n_folds": int(len(d)),
            "delta_median": float(np.median(d)) if len(d) else None,
            "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()),
            "delta_mean": float(d.mean()) if len(d) else None,
            "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        }
        if len(d) >= 5:
            from scipy.stats import wilcoxon
            try:
                out["wilcoxon_p"] = float(wilcoxon(d).pvalue)
            except ValueError:
                out["wilcoxon_p"] = None
        return out

    ok = [v for v in runs.values() if "error" not in v]
    sources = sorted({v["source"] for v in ok})
    summary = {"all": _summary(ok)}
    for src in sources:
        summary[src] = _summary([v for v in ok if v["source"] == src])
    return {"runs": runs, "summary": summary}


def write_report(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loao.json").write_text(json.dumps(res, indent=1))
    lines = [
        "# A7 under LOAO - derived gate vs dedicated gate",
        "",
        "One-vs-normal detection on the excluded family (same folds, seed 42).",
        "Delta = A7 - typed; positive is better. Bootstrap intervals are",
        "descriptive (B=300); the fold-level sign test is the verdict.",
        "",
        "| fold | AUROC a7 | AUROC typed | delta | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for key, v in res["runs"].items():
        if "error" in v:
            lines.append(f"| {key} | — | — | — | — |")
            continue
        b = v["bootstrap"]
        lines.append(f"| {key} | {v['a7']['auroc']:.4f} | {v['typed']['auroc']:.4f} | "
                     f"{b['delta']:+.4f} | [{b['lo']:+.4f},{b['hi']:+.4f}] |")
    lines += ["", "## Summary", ""]
    for name, s in res["summary"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- folds: {s['n_folds']}")
        lines.append(f"- median delta: {s['delta_median']:+.4f}")
        lines.append(f"- mean delta: {s['delta_mean']:+.4f} ± {s['delta_sd']:.4f}")
        lines.append(f"- A7 better on {s['wins']}/{s['n_folds']} folds")
        if s.get("wilcoxon_p") is not None:
            lines.append(f"- exact Wilcoxon signed-rank p = {s['wilcoxon_p']:.4f}")
        lines.append("")
    (out_dir / "loao.md").write_text("\n".join(lines) + "\n")


def seed_check(seed_dir: Path, out_dir: Path, base_dir: Path | None = None) -> dict:
    """Per-fold AUROC of the a7 gate across retraining seeds (open-set check)."""
    rows: dict[str, dict[str, float]] = {}

    def _add(npz: Path, seed: int) -> None:
        fold = npz.stem.split("_loao_")[1].split("_abl_a7")[0]
        source = npz.stem.split("_", 1)[0]
        a7 = load_fold(npz)
        m = a7["mask"]
        rows.setdefault(f"{source}/{fold}", {})[str(seed)] = fast_auc(
            a7["p_n"][m], a7["y"][m])

    for npz in sorted(seed_dir.glob("*_abl_a7_s*.npz")):
        _add(npz, int(npz.stem.rsplit("_s", 1)[1]))
    if base_dir is not None:
        for npz in sorted(base_dir.glob("*_loao_*_abl_a7.npz")):
            _add(npz, 42)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "loao_seeds.json").write_text(json.dumps(rows, indent=1))
    lines = ["# A7 LOAO - seed check", "",
             "One-vs-normal AUROC of the derived gate per retraining seed.", "",
             "| fold | seed 42 | seed 43 | seed 44 | range |",
             "|---|---|---|---|---|"]
    for fold, vals in sorted(rows.items()):
        v = [vals.get(k) for k in ("42", "43", "44")]
        got = [x for x in v if x is not None]
        lines.append("| {} | {} | {} | {} | {} |".format(
            fold, *["—" if x is None else f"{x:.4f}" for x in v],
            f"{min(got):.4f}–{max(got):.4f}"))
    (out_dir / "loao_seeds.md").write_text("\n".join(lines) + "\n")
    return rows


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.5).astype(np.int64)
    score = rng.normal(size=n) + 1.0 * y
    run = {"excl": "x", "p_n": score, "mask": np.ones(n, dtype=bool),
           "y": y, "n": n}
    b = bootstrap_delta(run, run, np.random.default_rng(RNG_SEED))
    assert abs(b["delta"]) < 1e-12, b
    print("[selftest] A7 vs itself under LOAO: delta is zero - OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7", default="results/e6_a7/loao")
    ap.add_argument("--typed", default="results/outputs")
    ap.add_argument("--out", default="results/e6_a7")
    ap.add_argument("--seeds-dir", default="results/e6_a7/loao_seeds")
    ap.add_argument("--seed-check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.seed_check:
        rows = seed_check(Path(args.seeds_dir), Path(args.out),
                          Path(args.a7))
        for fold, vals in rows.items():
            print(f"[a7-loao-seed] {fold}: {vals}")
        return
    res = evaluate(Path(args.a7), Path(args.typed))
    write_report(res, Path(args.out))
    for name, s in res["summary"].items():
        print(f"[a7-loao:{name}] folds={s['n_folds']} "
              f"median={s['delta_median']:+.4f} wins={s['wins']}/{s['n_folds']} "
              f"p={s.get('wilcoxon_p')}")


if __name__ == "__main__":
    main()
