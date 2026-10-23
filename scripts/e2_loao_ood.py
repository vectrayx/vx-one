#!/usr/bin/env python
"""E2b — baselines OOD clásicos por fold LOAO (ítem 2 del informe externo).

Reutiliza EXACTAMENTE el protocolo de run_tabular (mismo build_run con
exclude_family → mismos splits/fold/matríz φ). Añade por fold:

  * logreg      : regresión logística binaria (ataque vs normal) entrenada en
                  las familias conocidas (la familia excluida nunca entra);
                  score directo sobre filas test de la familia excluida vs
                  normal. Es "el clasificador binario entrenado con el resto
                  de familias" del informe — hipótesis a confirmar/refutar.
  * mahalanobis : gaussiano una-clase sobre train *normal* (media + cov
                  compartida con shrinkage Ledoit-Wolf); score = −MD
                  (mayor = más normal). El análogo one-class del gate noul.

Salida: results/baselines_loao/<run_id>/{logreg,mahalanobis}.{npz,json}
npz: p_gate (score de ataque), p_noul (−score, alineado al head noul),
     y_bin, y_fam, fam_names, excl_mask — convención de run_tabular para
     que el evaluador de LOAO-baselines sea único.

Uso:
    PYTHONPATH=. python scripts/e2_loao_ood.py --source unsw
    PYTHONPATH=. python scripts/e2_loao_ood.py --source cic --scheme strat
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.models.data import build_run

FAMS = {
    "unsw": ["analysis", "backdoor", "dos", "exploitation", "fuzzing",
             "generic", "recon", "shellcode", "worms"],
    "cic": ["botnet", "bruteforce", "dos_ddos", "exploitation",
            "infiltration", "recon", "webattack"],
}

SEED = 42


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def fpr_at_tpr95(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score)
    y_s = y[order]
    tpr = np.cumsum(y_s) / max(y_s.sum(), 1)
    fpr = np.cumsum(1 - y_s) / max((1 - y_s).sum(), 1)
    idx = np.searchsorted(tpr, 0.95, side="left")
    if idx >= len(fpr):
        return 1.0
    return float(fpr[min(idx, len(fpr) - 1)])


def run_fold(source: str, scheme: str, fam: str, out_root: Path) -> None:
    run_id = f"{source}_small_{scheme}_loao_{fam}"
    outdir = out_root / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    r = build_run(source, "small", scheme, exclude_family=fam)
    xtr = r.train.detach().cpu().numpy().astype(np.float64)
    xte = r.test.detach().cpu().numpy().astype(np.float64)

    cands = list(r.candidates)
    yf_tr = r.train_labels.y_fam.detach().cpu().numpy()
    y_fam = r.test_labels.y_fam.detach().cpu().numpy()
    normal_idx = cands.index("normal")
    y_bin = (y_fam != normal_idx).astype(np.int64)
    fam_names = np.array(cands, dtype=object)

    # --- logreg binaria (el resto de familias; excluida fuera de train)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(xtr)
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    lr.fit(sc.transform(xtr), yf_tr != normal_idx)
    gate_lr = lr.decision_function(sc.transform(xte))       # score ataque

    # --- Mahalanobis una clase sobre train normal
    from sklearn.covariance import LedoitWolf
    xtr_n = xtr[yf_tr == normal_idx]
    mu = xtr_n.mean(axis=0)
    cov = LedoitWolf().fit(xtr_n)
    md_normal = -cov.mahalanobis(xte - mu)                  # mayor = más normal

    excl_idx = cands.index(fam) if fam in cands else -1
    excl = y_fam == excl_idx
    m_eval = (y_fam == normal_idx) | excl
    rows = {}
    for name, score in (("logreg", gate_lr), ("mahalanobis", -md_normal)):
        auc, fpr = auroc(score[m_eval], y_bin[m_eval]), 0.0
        fpr = fpr_at_tpr95(score[m_eval], y_bin[m_eval])
        rows[name] = {"auroc_excl": round(auc, 4), "fpr95_excl": round(fpr, 4),
                      "n_excl_test": int(excl.sum())}
        np.savez_compressed(outdir / f"{name}.npz",
                            p_gate=score.astype(np.float32),
                            p_noul=(-score).astype(np.float32),
                            y_bin=y_bin, y_fam=y_fam, fam_names=fam_names,
                            excl_mask=excl)
        (outdir / f"{name}.json").write_text(json.dumps(
            {"run_id": run_id, "variant": name, **rows[name],
             "fit_s": round(time.time() - t0, 1)}, indent=1))
    print(f"[ood] {run_id}: {rows}  ({time.time()-t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["unsw", "cic"], required=True)
    ap.add_argument("--scheme", default=None, help="(default: oficial/strat)")
    ap.add_argument("--only", default=None, help="un solo fold")
    args = ap.parse_args()
    scheme = args.scheme or ("oficial" if args.source == "unsw" else "strat")
    out_root = Path(__file__).resolve().parent.parent / "results/baselines_loao"
    folds = [args.only] if args.only else FAMS[args.source]
    for fam in folds:
        run_fold(args.source, scheme, fam, out_root)


if __name__ == "__main__":
    main()
