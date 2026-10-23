#!/usr/bin/env python
"""E2c — evaluador LOAO de baselines (mismo protocolo que run_e2 para VX).

Consume results/baselines_loao/<run_id>/{xgboost,rf,mlp} (npz espejo de
run_tabular: probs crudas + json de temperaturas, convención y_fam=-1 en las
filas de la familia excluida) y results/baselines_loao/<run_id>ood-dir/
{logreg,mahalanobis} (solo gate). Calibra p_choice/p_noul con el MISMO
esquema de E1 (_baseline_calibrated: logits=log p, temperaturas val del
propio modelo, ω del artefacto VX del fold). Produce:

  * por fold y modelo: gate AUROC + FPR@95 (excluida vs normal); para los
    modelos con choice: barrido τ_abs ∈{0.05..0.99} → abstención y
    atribución a una clase de ataque conocida; @0.5 y macro-F1 de control
    sobre filas conocidas.

Salida: results/e2_baselines/e2_loao_baselines_{source}.{json,md}

Uso: PYTHONPATH=. python scripts/e2_loao_baselines_eval.py --source unsw
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOAO = ROOT / "results/baselines_loao"
OUT = ROOT / "results/e2_baselines"
MODELS = ["xgboost", "rf", "mlp", "logreg", "mahalanobis"]
TAUS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

FAMS = {
    "unsw": ["analysis", "backdoor", "dos", "exploitation", "fuzzing",
             "generic", "recon", "shellcode", "worms"],
    "cic": ["botnet", "bruteforce", "dos_ddos", "exploitation",
            "infiltration", "recon", "webattack"],
}
VX_JSON_BASE = ROOT / "results/outputs"   # metadatos ω de los folds VX


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def fpr_at_tpr95(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score)
    ys = y[order]
    tpr = np.cumsum(ys) / max(ys.sum(), 1)
    fpr = np.cumsum(1 - ys) / max((1 - ys).sum(), 1)
    i = np.searchsorted(tpr, 0.95, side="left")
    return float(fpr[min(i, len(fpr) - 1)]) if i < len(fpr) else 1.0


def macro_f1_known(y_true: np.ndarray, pred: np.ndarray) -> float:
    f1s = []
    for c in np.unique(np.concatenate([y_true, pred])):
        tp = int(((pred == c) & (y_true == c)).sum())
        fp = int(((pred == c) & (y_true != c)).sum())
        fn = int(((pred != c) & (y_true == c)).sum())
        if (2 * tp + fp + fn):
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def eval_model(dir_: Path, model: str, vx_json: Path | None) -> dict | None:
    from src.eval.run_e1e3 import _baseline_calibrated
    npz = dir_ / f"{model}.npz"
    if not npz.exists():
        return None
    z = np.load(npz, allow_pickle=True)
    y_fam = z["y_fam"].astype(np.int64)
    fam_names = [str(s) for s in z["fam_names"]]
    excl = y_fam == -1
    normal_idx = fam_names.index("normal")
    is_norm = y_fam == normal_idx
    m_eval = excl | is_norm
    y_glob = (y_fam == -1).astype(int)
    rec = {}

    gate = z["p_gate"] if "p_gate" in z.files else z["p_noul"] if "p_noul" in z.files else None
    if gate is not None:
        try:
            rec["auroc"] = round(auroc(np.asarray(gate, np.float64)[m_eval],
                                       y_glob[m_eval]), 4)
            rec["fpr95"] = round(fpr_at_tpr95(np.asarray(gate, np.float64)[m_eval],
                                              y_glob[m_eval]), 4)
        except ValueError:
            rec["auroc"] = rec["fpr95"] = None

    if "p_choice" in z.files and z["p_choice"].ndim == 2:
        meta = json.loads((dir_ / f"{model}.json").read_text())
        omega = None
        if vx_json is not None and vx_json.exists():
            omega = np.asarray(json.loads(vx_json.read_text())["omega"], np.float64)
        probs = {k: np.asarray(z[k], dtype=np.float64) for k in z.files
                 if k.startswith("p_")}
        cal = _baseline_calibrated(probs, meta["temperatures"],
                                   omega if omega is not None
                                   else np.ones(probs["p_choice"].shape[1]), None)
        pc = cal["p_choice"]
        mp, am = pc.max(1), pc.argmax(1)
        attn_known = [i for i, n in enumerate(fam_names) if n != "normal"]
        rec["sweep"] = {str(t): {
            "abst": round(float((mp[excl] < t).mean()), 4),
            "false_att": round(float(((mp[excl] >= t)
                                      & np.isin(am[excl], attn_known)).mean()), 4)}
            for t in TAUS}
        rec["abst05"] = rec["sweep"]["0.5"]["abst"]
        rec["false_att05"] = rec["sweep"]["0.5"]["false_att"]
        rec["normal_frac"] = round(float((am[excl] == normal_idx).mean()), 4)
        known = ~excl
        rec["macroF1_known"] = round(macro_f1_known(y_fam[known], am[known]), 4)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["unsw", "cic"], required=True)
    args = ap.parse_args()
    scheme = "oficial" if args.source == "unsw" else "strat"
    out = {"source": args.source, "folds": {}}
    pending = []
    for fam in FAMS[args.source]:
        tree_dir = LOAO / f"{args.source}_{scheme}_loao_{fam}"        # run_tabular
        ood_dir = LOAO / f"{args.source}_small_{scheme}_loao_{fam}"   # e2_loao_ood
        vx_json = VX_JSON_BASE / f"{'unsw_small_oficial' if args.source == 'unsw' else 'cic_small_strat'}_loao_{fam}.json"
        recs = {}
        for model in MODELS:
            r = None
            if model in ("xgboost", "rf", "mlp") and tree_dir.exists():
                r = eval_model(tree_dir, model, vx_json)
            elif model in ("logreg", "mahalanobis") and ood_dir.exists():
                r = eval_model(ood_dir, model, None)
            recs[model] = r
            if r is None:
                pending.append(f"{fam}:{model}")
        out["folds"][fam] = recs

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"e2_loao_baselines_{args.source}.json").write_text(json.dumps(out, indent=1))
    md = [f"# E2b — LOAO baselines ({args.source})",
          f"pendiente: {pending if pending else 'nada'}", ""]
    for model in MODELS:
        md += [f"## {model}", "",
               "| fold | gate AUROC | FPR@95 | abst@0.5 | falseAtt@0.5 | normal | macroF1 known |",
               "|---|---|---|---|---|---|---|"]
        for fam, recs in out["folds"].items():
            v = recs.get(model) or {}
            if v is None:
                md.append(f"| {fam} | — pending — | | | | | |")
                continue
            md.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                fam, v.get("auroc"), v.get("fpr95"), v.get("abst05", "—"),
                v.get("false_att05", "—"), v.get("normal_frac", "—"),
                v.get("macroF1_known", "—")))
        md.append("")
    (OUT / f"e2_loao_baselines_{args.source}.md").write_text("\n".join(md) + "\n")
    print(f"[eval] → {OUT}/e2_loao_baselines_{args.source}.md (pending: {pending})")


if __name__ == "__main__":
    main()
