#!/usr/bin/env python
"""E9b-eval — LOAO con outlier exposure vs LOAO referencia (por fold).

Compara, sobre las MISMAS filas test (excluida vs normal):
  * gate AUROC / FPR@95   (noul)
  * abstención @τ∈{0.3,0.5,0.7} sobre filas de la familia excluida
    (max_k p_choice < τ, candidatos conocidos)
  * atribución falsa a clase de ataque conocida @0.5
  * macro-F1 de control sobre filas conocidas
Referencia: results/outputs/unsw_small_oficial_loao_{fam}.npz
OE (λ=0.5): results/outputs_oe/unsw_small_oficial_loao_{fam}_oe0.5.npz

Salida: results/e9b_oe/e9b_oe.{json,md}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REF = Path("/home/jsantillana/projects/paper/results/outputs")
OE = ROOT / "results/outputs_oe"
OUT = ROOT / "results/e9b_oe"
FAMS = ["analysis", "backdoor", "dos", "exploitation", "fuzzing",
        "generic", "recon", "shellcode", "worms"]
TAUS = [0.3, 0.5, 0.7, 0.9]


def auroc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def fpr_at_tpr95(score, y):
    order = np.argsort(-score)
    ys = y[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    i = int(np.searchsorted(tpr, 0.95, side="left"))
    return float(fpr[min(i, len(fpr) - 1)]) if i < len(fpr) else 1.0


def macro_f1_known(y_true, pred):
    f1s = []
    for c in np.unique(np.concatenate([y_true, pred])):
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        if (2 * tp + fp + fn):
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def fold_metrics(npz_path: Path) -> dict | None:
    if not npz_path.exists():
        return None
    z = np.load(npz_path, allow_pickle=True)
    yf = z["te_y_fam"].astype(np.int64)
    names = [str(s) for s in z["te_fam_names"]]
    excl = yf == -1
    ni = names.index("normal")
    is_norm = yf == ni
    rec = {}
    g = 1 / (1 + np.exp(-np.clip(z["te_noul_logits"], -30, 30)))
    m = excl | is_norm
    rec["gate_auroc"] = round(auroc(g[m], excl[m].astype(int)), 4)
    rec["gate_fpr95"] = round(fpr_at_tpr95(g[m], excl[m].astype(int)), 4)
    T = 1.0
    meta_p = npz_path.with_suffix(".json")
    if meta_p.exists():
        T = json.loads(meta_p.read_text())["temperatures"]["choice"]
    zl = z["te_choice_logits"].astype(np.float64) / T
    p = np.exp(zl - zl.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    mp, am = p.max(1), p.argmax(1)
    att_known = [i for i, n in enumerate(names) if n != "normal"]
    rec["sweep"] = {str(t): {
        "abst": round(float((mp[excl] < t).mean()), 4),
        "false_att": round(float(((mp[excl] >= t) & np.isin(am[excl], att_known)).mean()), 4),
        "normal_absorb": round(float(((mp[excl] >= t) & (am[excl] == ni)).mean()), 4)}
        for t in TAUS}
    known = ~excl
    rec["known_macroF1"] = round(macro_f1_known(yf[known], am[known]), 4)
    rec["median_maxprob_excl"] = round(float(np.median(mp[excl])), 4)
    return rec


def main() -> None:
    out = {"folds": {}}
    for fam in FAMS:
        ref = fold_metrics(REF / f"unsw_small_oficial_loao_{fam}.npz")
        oe = fold_metrics(OE / f"unsw_small_oficial_loao_{fam}_oe0.5.npz")
        out["folds"][fam] = {"ref": ref, "oe": oe}
        if ref and oe:
            print(f"{fam:13s} gate {ref['gate_auroc']:.4f}→{oe['gate_auroc']:.4f} | "
                  f"abst@.5 {ref['sweep']['0.5']['abst']:.3f}→{oe['sweep']['0.5']['abst']:.3f} | "
                  f"falseAtt@.5 {ref['sweep']['0.5']['false_att']:.3f}→{oe['sweep']['0.5']['false_att']:.3f} | "
                  f"knownF1 {ref['known_macroF1']:.3f}→{oe['known_macroF1']:.3f}")
        else:
            print(f"{fam:13s} pendiente (ref={ref is not None}, oe={oe is not None})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e9b_oe.json").write_text(json.dumps(out, indent=1))
    md = ["# E9b — LOAO + outlier exposure (λ=0.5, familia excluida como pool OE)", "",
          "| fold | gate AUROC ref→oe | abst@.5 ref→oe | falseAtt@.5 ref→oe | knownF1 ref→oe | median maxp excl ref→oe |",
          "|---|---|---|---|---|---|"]
    for fam, v in out["folds"].items():
        r, o = v["ref"], v["oe"]
        if not (r and o):
            md.append(f"| {fam} | pendiente | | | | |")
            continue
        md.append(f"| {fam} | {r['gate_auroc']}→{o['gate_auroc']} | "
                  f"{r['sweep']['0.5']['abst']}→{o['sweep']['0.5']['abst']} | "
                  f"{r['sweep']['0.5']['false_att']}→{o['sweep']['0.5']['false_att']} | "
                  f"{r['known_macroF1']}→{o['known_macroF1']} | "
                  f"{r['median_maxprob_excl']}→{o['median_maxprob_excl']} |")
    (OUT / "e9b_oe.md").write_text("\n".join(md) + "\n")
    print(f"[e9b] → {OUT}/e9b_oe.md")


if __name__ == "__main__":
    main()
