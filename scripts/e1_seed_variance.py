#!/usr/bin/env python3
"""Agrega la varianza 3-seed de las corridas principales (evaluadores E1
locales, sin re-entrenar). Resultados en results/e1_seed/.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import eval_vx_run

RUNS = [
    "unsw_small_oficial", "unsw_small_oficial_s43", "unsw_small_oficial_s44",
    "cic_small_strat", "cic_small_strat_s43", "cic_small_strat_s44",
    "cic_small_time", "cic_small_time_s43", "cic_small_time_s44",
    "ember_small_temporal", "ember_small_temporal_s43",
    "ember_small_temporal_s44",
]
BASE = Path("results/outputs_seed")
OUT = Path("results/e1_seed")
OUT.mkdir(parents=True, exist_ok=True)


def pick(res: dict) -> dict:
    te = res["te"]
    out = {
        "auroc": te["noul"]["e1"]["auroc"],
        "fpr@95tpr": te["noul"]["e1"]["fpr@95tpr"],
        "pr_auc": te["noul"]["e1"]["pr_auc"],
    }
    ch = te.get("choice")
    if ch:
        out["macro_f1"] = ch["e1"]["macro_f1"]
        out["ece_choice"] = ch["e3"]["cal"]["ece"]
    out["ece_noul"] = te["noul"]["e3"]["cal"]["ece"]
    return out


def main() -> None:
    allres = {}
    for run in RUNS:
        res = eval_vx_run(BASE / f"{run}.npz")
        allres[run] = {"flat": pick(res), "full": res}

    groups = ["unsw_small_oficial", "cic_small_strat", "cic_small_time",
              "ember_small_temporal"]
    agg = {}
    for g in groups:
        seeds = [allres[f"{g}"], allres[f"{g}_s43"], allres[f"{g}_s44"]]
        agg[g] = {}
        for k in ("auroc", "fpr@95tpr", "macro_f1", "ece_choice", "ece_noul"):
            vals = []
            for s in seeds:
                v = s["flat"].get(k)
                vals.append(None if v is None else float(v))
            if all(v is not None for v in vals):
                a = np.asarray(vals, dtype=np.float64)
                agg[g][k] = {"mean": float(a.mean()), "std": float(a.std(ddof=0)),
                             "vals": vals}
            else:
                agg[g][k] = {"vals": vals}

    summary = {"seeds": [42, 43, 44], "eval": "src.eval.metrics.eval_vx_run "
               "(de-tilt + T; mismo contrato del engine)", "groups": agg,
               "per_run": {r: allres[r]["flat"] for r in RUNS}}
    json.dump(summary, open(OUT / "seed_variance.json", "w"), indent=2)

    with open(OUT / "seed_variance.md", "w") as f:
        f.write("# Varianza 3-seed corridas principales (test, mismo protocolo E1)\n\n")
        f.write("| corrida | métrica | s42 | s43 | s44 | mean±std |\n|---|---|---|---|---|---|\n")
        for g in groups:
            for k in ("auroc", "fpr@95tpr", "macro_f1", "ece_choice", "ece_noul"):
                a = agg[g].get(k, {})
                vals = a.get("vals", [])
                if vals and all(v is not None for v in vals):
                    f.write(f"| {g} | {k} | " + " | ".join(f"{v:.4f}" for v in vals)
                            + f" | {a['mean']:.4f}±{a['std']:.4f} |\n")
                else:
                    f.write(f"| {g} | {k} | " + " | ".join(
                        "—" if v is None else f"{v:.4f}" for v in vals) + " | n/d |\n")
            f.write("|\n")
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
