#!/usr/bin/env python
"""E8-LS — label shift explícito bajo drift (externo al ítem 6 del informe):
¿la calibración del día 2 (PortScan) es 'techo = discriminación' o 'falta
corregir el prior'?

Protocolo (misma convención frozen de e8_window_recal/e8_nonparam):
  * Gate = noul, logits te_noul_logits de results/outputs_seed/cic_small_time.npz
    en orden de captura: day1 = Friday-DDos n=138,145 (etiquetas retrasadas
    disponibles), day2 = Friday-PortScan n=286,467 (a evaluar).
  * Mapa base M ajustado SOLO con labels del día 1 (extensible localmente,
    desplegable en producción): temperature / Platt / isotonic sobre day1.
  * Estimadores del prior del día 2 (sin usar labels de day2):
      - BBSE-hard  : modele confusion matrix de day1 con umbral 0.5, q̂=C⁻¹μ.
      - BBSE-soft  : q̂ = (E[r] − E[r|y=0]) / (E[r|y=1] − E[r|y=0]) sobre day1.
      - EM-Saerens : EM sobre los posteriors plug-in de M en day2.
  * Corrección de prior que PRESERVA el ranking (odds re-weighting):
      p'(x) = α r / (α r + (1−α)(1−r)),  α = [q̂2/(1−q̂2)] / [q1/(1−q1)]
  * Referencias de diagnóstico:
      - constante en q̂2 (calibración-in-the-large perfecta, mata ranking)
      - constante en q2 verdadero (cota oracle-prior; SOLO diagnóstico)
  * Sanity day1: los estimadores deben devolver q̂ ≈ q1 en el propio day1.

Salida: results/e8_recal/e8_labelshift.{json,md}

Ejecución: python -m scripts.e8_labelshift
(necesita solo numpy+scipy+sklearn; no GPU, no parquet, no entrenamiento)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parent.parent
NPZ = REPO / "results/outputs_seed/cic_small_time.npz"
OUT = REPO / "results/e8_recal"
DAYS = [("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX", 138145),
        ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX", 286467)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def ece_bin(score: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """ECE 15 bins de anchura igual (convención frozen del repo)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, bins - 1)
    e, n = 0.0, len(y)
    for b in range(bins):
        m = idx == b
        if m.any():
            e += (m.sum() / n) * abs(score[m].mean() - y[m].mean())
    return float(e)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def platt_fit(logit: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a,b) minimizando NLL binaria sobre p=sigmoid(a·logit+b)."""
    z = np.clip(logit, -60, 60)

    def nll(c):
        a, b = c
        p = sigmoid(a * z + b)
        eps = 1e-12
        return -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    r = minimize_scalar(lambda t: nll((1.0, t)), method="bounded", bounds=(-40, 40))
    b0 = r.x
    r2 = minimize_scalar(lambda a: nll((a, b0)), method="bounded", bounds=(1e-6, 60))
    a1 = r2.x
    best = r2.fun
    r3 = minimize_scalar(lambda b: nll((a1, b)), method="bounded", bounds=(-40, 40))
    b1 = r3.x
    r4 = minimize_scalar(lambda a: nll((a, b1)), method="bounded", bounds=(1e-6, 60))
    if r4.fun < best:
        a1, b1 = r4.x, b1
    return a1, b1


def temp_fit(logit: np.ndarray, y: np.ndarray) -> float:
    z = np.clip(logit, -60, 60)

    def nll(logt):
        p = sigmoid(z / np.exp(logt))
        eps = 1e-12
        return -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    r = minimize_scalar(nll, method="bounded", bounds=(-8, 8))
    return float(np.exp(r.x))


def iso_fit(p: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(np.clip(p, 1e-9, 1 - 1e-9), y.astype(float))
    return ir


def bbse_hard(r1: np.ndarray, y1: np.ndarray, r2: np.ndarray) -> float:
    """Confusión C[i,j]=E[1{pred=j}|y=i] en day1; q̂=C⁻¹ μ con μ=mean(pred2)."""
    pred1 = (r1 >= 0.5).astype(int)
    C = np.zeros((2, 2))
    for i in (0, 1):
        m = y1 == i
        if m.any():
            C[i] = [float((pred1[m] == 0).mean()), float((pred1[m] == 1).mean())]
    mu = np.array([(r2 < 0.5).mean(), (r2 >= 0.5).mean()])
    try:
        q = float(np.linalg.solve(C, mu)[1])
    except np.linalg.LinAlgError:
        return float("nan")
    return q


def bbse_soft(r1: np.ndarray, y1: np.ndarray, r2: np.ndarray) -> float:
    a = float(r1[y1 == 1].mean())
    b = float(r1[y1 == 0].mean())
    mu = float(r2.mean())
    if abs(a - b) < 1e-9:
        return float("nan")
    return (mu - b) / (a - b)


def em_pi(posterior_base: np.ndarray, pi0: float | None = None) -> float:
    """EM/Saerens con posterior plug-in r(x)=P_{q1}(y=1|x):
    π ← mean(π r / (π r + (1−π)(1−r))); parte en pi0 (q1), 200 iteraciones."""
    pi = float(np.clip(pi0 if pi0 is not None else posterior_base.mean(),
                       1e-6, 1 - 1e-6))
    r = np.clip(posterior_base, 1e-6, 1 - 1e-6)
    for _ in range(200):
        pos = pi * r
        post = pos / np.maximum(pos + (1 - pi) * (1 - r), 1e-12)
        new = float(post.mean())
        if abs(new - pi) < 1e-10:
            pi = new
            break
        pi = new
    return pi


def prior_odds(p: np.ndarray, q_from: float, q_to: float) -> np.ndarray:
    """Re-weighting de prior con odds ratio (preserva ranking)."""
    eps = 1e-9
    a = (q_to / max(1 - q_to, eps)) / (q_from / max(1 - q_from, eps))
    num = a * p
    return num / np.maximum(num + (1 - a) * (1 - p), 1e-12)


def metrics_row(name: str, p: np.ndarray, y: np.ndarray) -> dict:
    try:
        auc = auroc(p, y)
    except ValueError:
        auc = float("nan")
    return {"variant": name, "auroc": round(auc, 4) if auc == auc else None,
            "ece15": round(ece_bin(np.clip(p, 0, 1), y), 4),
            "brier": round(brier(np.clip(p, 0, 1), y), 4),
            "prev_mean_pred": round(float(np.clip(p, 0, 1).mean()), 4)}


def main() -> None:
    z = np.load(NPZ, allow_pickle=True)
    n1 = DAYS[0][1]
    logits = z["te_noul_logits"].astype(np.float64)
    y = z["te_y_bin"].astype(np.int64)
    va_logit = z["va_noul_logits"].astype(np.float64)
    va_y = z["va_y_bin"].astype(np.int64)

    y1, logit1, y2, logit2 = y[:n1], logits[:n1], y[n1:], logits[n1:]
    p1_raw, p2_raw = sigmoid(logit1), sigmoid(logit2)
    q1, q2 = float(y1.mean()), float(y2.mean())
    print(f"[e8LS] q1(day1)={q1:.4f}  q2_true(day2)={q2:.4f}")

    # ---- mapas base ajustados SOLO con labels day1 (despliegue honesto)
    p1 = sigmoid(logit1)
    T1 = temp_fit(logit1, y1)
    (pl_a, pl_b) = platt_fit(logit1, y1)
    ir1 = iso_fit(p1, y1)
    base = {
        "temp(day1)": (lambda l: sigmoid(l / T1)),
        "platt(day1)": (lambda l: sigmoid(pl_a * np.clip(l, -60, 60) + pl_b)),
        "iso(day1)": (lambda l: ir1.predict(sigmoid(np.clip(l, -60, 60)))),
    }
    # r1 = M(day1), r2 = M(day2) — plug-in posteriors bajo el prior day1
    r1 = {k: f(logit1) for k, f in base.items()}
    r2 = {k: f(logit2) for k, f in base.items()}

    estim = {}
    for k in r2:
        sh, so = bbse_hard(r1[k], y1, r2[k]), bbse_soft(r1[k], y1, r2[k])
        em = em_pi(r2[k], pi0=q1)  # EM/Saerens sobre los posteriors de day2
        estim[k] = {"bbse_hard": sh, "bbse_soft": so, "em": em}

    out = {"protocol": {"npz": str(NPZ), "day1": DAYS[0][0], "day2": DAYS[1][0],
                        "q1": round(q1, 4), "q2_true": round(q2, 4),
                        "base_maps": "day1-only labels (temp/platt/iso)"},
           "prior_estimates": {k: {kk: round(vv, 4) for kk, vv in v.items()}
                               for k, v in estim.items()},
           "days": {}}

    # ---- day1 sanity (prior estable intra-regime)
    rows1 = [metrics_row("static_temp(val)", sigmoid(np.clip(va_logit, -60, 60) / T1), va_y)]
    rows1.append(metrics_row(f"const@q1(oracle)", np.full_like(p1_raw, q1), y1))
    out["days"][DAYS[0][0]] = {"q_true": round(q1, 4), "rows": rows1}

    # ---- day2: cada mapa base {sin corrección, corrección por cada estimador}
    rows = []
    for k in base:
        rows.append(metrics_row(f"{k} sin corrección", r2[k], y2))
        for est, qhat in estim[k].items():
            if not np.isfinite(qhat):
                continue
            qh = float(np.clip(qhat, 1e-6, 1 - 1e-6))
            p_corr = prior_odds(r2[k], q1, qh)
            rows.append(metrics_row(f"{k} prior-corregido[{est}] (q̂={qh:.3f})", p_corr, y2))
            rows.append(metrics_row(f"const@q̂2[{est}] (q̂={qh:.3f})",
                                    np.full_like(p2_raw, qh), y2))
    rows.append(metrics_row("const@q2_true(oracle-prior)", np.full_like(p2_raw, q2), y2))
    rows.append(metrics_row("static_temp(val) — serving actual", sigmoid(np.clip(logit2, -60, 60) / T1), y2))
    rows.append(metrics_row("raw (sin calibrar)", p2_raw, y2))
    out["days"][DAYS[1][0]] = {"q_true": round(q2, 4), "rows": rows}

    brier_const = round(q2 * (1 - q2), 4)
    out["reference"] = {"brier_min_const": brier_const,
                        "ece_min_const": 0.0,
                        "note": "constante en el prior verdadero: ECE→0 y "
                                "Brier=q(1-q); el 'plateau 0.445' del binfit "
                                "se debe a constante en MEDIA DEL SCORE CRUDO "
                                "(p.mean()≈0.0002), no del prior (Q2=0.555)."}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e8_labelshift.json").write_text(json.dumps(out, indent=1))
    md = ["# E8-LS — label shift explícito (PortScan day2)", "",
          f"q1={q1:.4f}  q2_true={q2:.4f}",
          "| mapa base | estimador | q̂2 | ECE15 | Brier | AUROC |", "|---|---|---|---|---|---|"]
    for k in base:
        rows = out["days"][DAYS[1][0]]["rows"]
        for r in rows:
            if r["variant"].startswith(k) or r["variant"] in ("raw (sin calibrar)",):
                est = "none" if "sin corrección" in r["variant"] else ""
                md.append(f"| {r['variant']} | {est} | — | {r['ece15']} | {r['brier']} | {r['auroc']} |")
    (OUT / "e8_labelshift.md").write_text("\n".join(md) + "\n")
    print(f"[e8LS] → {OUT}/e8_labelshift.json + .md")
    for r in out["days"][DAYS[1][0]]["rows"]:
        print(f"  {r['variant']:46s} ece={r['ece15']:.4f} brier={r['brier']:.4f} auroc={r['auroc']}")


if __name__ == "__main__":
    main()
