"""E6 — ablations (spec F2 §7, a Small; A3/A4/A5 in the 3 datasets).

The ablations re-TRAIN engine variants with a single change each:
A1 without textual branch (embeddings to zero; only UNSW, the only one with text)
A2 without multi-task (single-head, total budget equal) [pending]
A3 without calibration (T=1; threshold-free metrics do not change — they are
verifies — and E3 shows absolute semantics degradation)
A4 uniform weights ω≡1 (without de-tilt: the posterior inherits the tilt of the train)
A5 unknown off (EMBER: 10-way without unknown; UNSW/CIC: τ_abs→1, no abstention)
A6 dropout 0 (regularization)

Each ablation produces the SAME artifact/npz as a normal run, so
The standard E1-E3 evaluator is used to compare against the entire run.

Use:
modal run modal/app.py::ablation --kind a3 --source unsw (in Modal)
python -m src.eval.run_e6 --vx results/outputs-abl (local eval)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import eval_vx_run

ABLATIONS = {
    "a1": "sin rama textual (UNSW)",
    "a3": "sin calibración (T=1)",
    "a4": "pesos uniformes ω≡1",
    "a5": "unknown off",
    "a6": "dropout 0",
}


def compare(run_dir: Path, baseline_dir: Path = Path("results/outputs")) -> dict:
    """Table ablation vs complete run (Small) per dataset."""
    out: dict = {}
    for npz in sorted(run_dir.glob("*.npz")):
        stem = npz.stem
        if stem.startswith("smoke"):
            continue
        r = eval_vx_run(npz)
        te = r.get("te") or {}
        base_stem = stem.replace("_abl", "")
        base_npz = baseline_dir / f"{base_stem}.npz"
        row: dict = {"ablation": r}
        if base_npz.exists():
            row["baseline"] = eval_vx_run(base_npz)
        out[stem] = row
    return out


def _row(te: dict, no_cal: bool = False) -> tuple:
    n, c = te.get("noul"), te.get("choice")
    if not n:
        return None
    if no_cal:  # A3: probs sin T = raw
        return (n["e1"]["auroc"], n["e1"]["fpr@95tpr"],
                c["e1"]["macro_f1"] if c else None,
                n["e3"]["raw"]["ece"], c["e3"]["raw"]["ece"] if c else None)
    return (n["e1"]["auroc"], n["e1"]["fpr@95tpr"],
            c["e1"]["macro_f1"] if c else None,
            n["e3"]["cal"]["ece"], c["e3"]["cal"]["ece"] if c else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abl", default="results/outputs-abl")
    ap.add_argument("--vx", default="results/outputs")
    ap.add_argument("--out", default="results/e6")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = compare(Path(args.abl), Path(args.vx))
    (out / "e6.json").write_text(json.dumps(results, indent=1, default=str))
    lines = ["| corrida | AUROC | FPR@95TPR | macro-F1 | ECE noul | ECE choice |",
             "|---|---|---|---|---|---|"]
    for stem, row in results.items():
        for tag in ("baseline", "ablation"):
            if tag not in row:
                continue
            te = row[tag].get("te") or {}
            r = _row(te, no_cal="_abl_a3" in stem)
            if r:
                label = f"{stem} ({tag})" if tag == "baseline" else stem
                lines.append("| {} | {} | {} | {} | {} | {} |".format(
                    label, *[f"{v:.4f}" if isinstance(v, float) else "—" for v in r]))
    (out / "e6.md").write_text("\n".join(lines) + "\n")
    print(f"[e6] {len(results)} ablations → {out/'e6.json'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
