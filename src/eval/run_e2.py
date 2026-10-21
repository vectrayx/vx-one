"""E2 — zero-day (LOAO §6.7) y drift temporal (EMBER).

Para cada fold LOAO (npz con test COMPLETO, y_fam=−1 en la familia
excluida + test_fam_names):
- Noul restringido: AUROC y FPR@95TPR de la familia excluida vs normal
  (¿el detector sigue viendo la familia nunca vista?).
- Choice: cobertura de abstención sobre las filas de la familia excluida
  (¿abstiene como debe o atribuye mal?) + macro-F1 sobre el resto (las
  familias conocidas no deben degradarse).
- Drift EMBER: métricas por familia nativa del test (recall/Unknown-rate),
  contrastadas con los conteos de train (manifest) — narrativa sfone/
  grandoreiro/sinowal ~desaparecidas, berbew creciendo, challenge evasivo.

Uso:
    python -m src.eval.run_e2 --vx results/outputs --out results/e2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import _binary_metrics, fpr_at_tpr, _softmax, vx_calibrated


def _choice_probs(npz: dict, meta: dict, prefix: str = "te") -> tuple[np.ndarray, np.ndarray]:
    """(p_choice calibrada, probs crudas) del split prefix."""
    cal = vx_calibrated(npz, meta)
    return cal[f"{prefix}_p_choice"], cal[f"{prefix}_p_choice_raw"]


def eval_loao_fold(npz_path: str | Path) -> dict:
    npz = dict(np.load(npz_path, allow_pickle=True))
    meta = json.loads(Path(str(npz_path).replace(".npz", ".json")).read_text())
    excl = meta.get("exclude")
    if not excl:
        return {}  # no es fold LOAO
    p_c, _ = _choice_probs(npz, meta)
    y_bin, y_fam = npz["te_y_bin"], npz["te_y_fam"]
    names = np.asarray(npz["te_fam_names"], dtype=object)
    exc_mask = names == excl
    norm_mask = names == "normal"
    p_n = 1.0 / (1.0 + np.exp(-npz["te_noul_logits"].astype(np.float64)
                              / meta["temperatures"]["noul"]))

    # 1) ¿detecta la familia excluida? (uno-vs-normal, detección pura)
    m_en = exc_mask | norm_mask
    detection = _binary_metrics(p_n[m_en], (y_bin[m_en] == 1).astype(int))

    # 2) Choice sobre las filas de la familia excluida: ¿abstiene (τ_abs) o
    # atribuye a una familia equivocada? (ground truth: candidata no existe)
    conf = p_c[exc_mask].max(1)
    pred = p_c[exc_mask].argmax(1)
    cand = meta["candidates"]
    tau = meta.get("tau_abs", 0.5)
    attribution = {
        "tau_abs": tau,
        "n_rows": int(exc_mask.sum()),
        "abstained": float((conf < tau).mean()),  # → unknown (D1, correcto)
        "attributed": float((conf >= tau).mean()),
        "top_wrong_family": (cand[int(np.bincount(pred, minlength=len(cand)).argmax())]
                             if len(pred) else None),
    }

    # 3) familias conocidas intactas: macro-F1 sobre filas NO excluidas
    from sklearn.metrics import f1_score
    known = ~exc_mask & (y_fam >= 0)
    known_f1 = float(f1_score(y_fam[known], p_c[known].argmax(1),
                              average="macro", zero_division=0)) if known.any() else None

    return {"exclude": excl, "noul_on_excluded": detection,
            "choice_on_excluded": attribution, "known_macro_f1": known_f1,
            "m_val": meta.get("m_val")}


def eval_ember_drift(npz_path: str | Path) -> dict | None:
    npz = dict(np.load(npz_path, allow_pickle=True))
    meta = json.loads(Path(str(npz_path).replace(".npz", ".json")).read_text())
    if meta.get("source") != "ember":
        return None
    p_c, _ = _choice_probs(npz, meta)
    y_fam = npz["te_y_fam"]
    cand = meta["candidates"]
    # modo trained_unknown: unknown ES candidato nativo; por familia nativa:
    # recall (¿predice su familia?) y fracción atribuida a unknown
    per_family = {}
    for k, fam in enumerate(cand):
        m = y_fam == k
        if not m.any():
            continue
        pred = p_c[m].argmax(1)
        per_family[fam] = {
            "n_test": int(m.sum()),
            "recall_own": float((pred == k).mean()),
            "to_unknown": float((pred == cand.index("unknown")).mean()
                                if "unknown" in cand else 0.0),
        }
    # Noul global (detector): AUROC + FPR@95TPR
    p_n = 1.0 / (1.0 + np.exp(-npz["te_noul_logits"].astype(np.float64)
                              / meta["temperatures"]["noul"]))
    y_bin = npz["te_y_bin"]
    return {"per_family": per_family,
            "noul": _binary_metrics(p_n, y_bin)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", default="results/outputs")
    ap.add_argument("--out", default="results/e2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loao, drift = {}, {}
    for npz in sorted(Path(args.vx).glob("*.npz")):
        if npz.name.startswith("smoke"):
            continue
        if f := eval_loao_fold(npz):
            loao[npz.stem] = f
        if d := eval_ember_drift(npz):
            drift[npz.stem] = d
    (out / "e2.json").write_text(json.dumps({"loao": loao, "ember_drift": drift},
                                            indent=1, default=str))
    print(f"[e2] {len(loao)} folds LOAO, {len(drift)} corridas EMBER → {out}/e2.json")
    for k, f in loao.items():
        d = f["noul_on_excluded"]
        a = f["choice_on_excluded"]
        print(f"  {k}: excl={f['exclude']} AUROC={d['auroc']:.4f} "
              f"FPR@95={d['fpr@95tpr']:.4f} | abst={a['abstained']:.2f} "
              f"attr={a['attributed']:.2f} | knownF1={f['known_macro_f1']}")


if __name__ == "__main__":
    main()
