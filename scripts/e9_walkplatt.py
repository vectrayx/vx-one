#!/usr/bin/env python
"""E9 walk-recal: platt (a·logit+b) ajustado en el día-1 congelado (labels
retrasadas) → aplicado al día-2. Análogo directo de la fila `walkplatt` de
E8, sobre los logits del modelo retrainer. Entrada: npz E9 (--tag).
Salida: results/e9_retrain/e9_walk{tag}.json (merge en e9_vs_static.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e8_window_recal import sigmoid, soft_pred, ece_bin, temp_scale_multi  # noqa: E402
import e7_vendor_api as e7  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results/e9_retrain"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()
base = args.tag and f"_{args.tag}" or ""
# el save del main usa tag directo sin "_": cic_small_time_fridayep2.npz
suffix = args.tag or ""
d = np.load(OUT / f"cic_small_time_friday{suffix}.npz")
j = json.loads((OUT / f"e9_retrain{suffix}.json").read_text())
Tn = float(j["protocol"]["temperatures"]["calib_valday"]["T_noul"])
Tc = float(j["protocol"]["temperatures"]["calib_valday"]["T_choice"])

ln1 = d["te1_noul"].astype(np.float64)  # día-1 E9 (labels retrasadas)
y1 = d["te1_y_bin"].astype(np.int64)
ln2 = d["te2_noul"].astype(np.float64)
y2 = d["te2_y_bin"].astype(np.int64)


def nll(w):
    a, b = w
    p = np.clip(sigmoid(a * ln1 + b), 1e-7, 1 - 1e-7)
    return float(-(y1 * np.log(p) + (1 - y1) * np.log(1 - p)).mean())


(a, b) = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B").x

# walk-temperature cal (E8): temperatura binaria ajustada en día-1
def nll_t(logt):
    T = np.exp(logt)
    p = np.clip(sigmoid(ln1 / T), 1e-7, 1 - 1e-7)
    return float(-(y1 * np.log(p) + (1 - y1) * np.log(1 - p)).mean())


from scipy.optimize import minimize_scalar  # noqa: E402
T1 = float(np.exp(minimize_scalar(nll_t, bounds=(np.log(0.05), np.log(200.)),
                                  method="bounded", options={"xatol": 1e-4}).x))

cal_p1 = sigmoid(ln1 * 1.0)  # t=1: platt solo muta a·logit+b
rows = {
    "platt_fit": {"a": float(a), "b": float(b),
                  "day1_fit_BCE": round(float(nll([a, b])), 4)},
}

for name, gp in (("walkplatt", sigmoid(a * ln2 + b)),
                 ("walktemp", sigmoid(ln2 / T1))):
    rows[name] = {
        "auroc": round(e7.auc_from(y2, gp.astype(np.float64)), 4),
        "ece15": round(ece_bin(gp, y2), 4),
        "brier": round(float(((gp - y2) ** 2).mean()), 4),
    }
    # FPR@95 con el umbral de la MISMA ventana de recal (positivos día-1 en
    # la MISMA escala de la variante — platt usa platt(día-1), temp usa temp)
    if name == "walkplatt":
        gp1 = sigmoid(a * ln1 + b)
    else:
        gp1 = sigmoid(ln1 / T1)
    thr = float(np.quantile(gp1[y1 == 1], 0.95))
    rows[name]["fpr@95"] = round(float((gp[y2 == 0] >= thr).mean()), 4)
    rows[name]["thr@0.95cal"] = round(thr, 4)

# comparación E8 estática walkplatt ya en results/e8_recal/e8_recal.json
rows["T_walk_day1_temp"] = T1
(OUT / f"e9_walk{suffix or ''}.json").write_text(json.dumps(rows, indent=1))
print(json.dumps(rows, indent=1))
