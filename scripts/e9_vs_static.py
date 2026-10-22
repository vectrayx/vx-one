#!/usr/bin/env python
"""E9 vs estático: tabla comparativa día-1/día-2.

  * estático: results/outputs_seed/cic_small_time.npz (run by-day s42, pre-E9)
    → métricas aquí con las MISMAS funciones de E8 y mismo criterio de thr
      (cuantil 0.95 de los positivos de val calibrada).
  * E9: métricas ya guardadas en results/e9_retrain/e9_retrain.json
    (sev MAE calculado ahí con la y_sev real).
Chequeo adicional: empates de score día-2 E9 (¿AUROC 0.8 exacto?).
Salida: results/e9_retrain/e9_vs_static.{md,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
pa.set_memory_pool(pa.system_memory_pool())

from e8_window_recal import temp_scale_binary, temp_scale_multi, sigmoid, soft_pred, ece_bin, ece_multi  # noqa: E402
import e7_vendor_api as e7  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PARQUET = Path("${HOME}/projects/paper/data/processed/cicids2017/cicids2017.parquet")
DDOS_DAY = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX"
OUT = REPO / "results/e9_retrain"
N_CLASSES = 6  # candidate set congelado del artifact

d = np.load(REPO / "results/outputs_seed/cic_small_time.npz", allow_pickle=True)
e9 = np.load(OUT / "cic_small_time_friday.npz")

pt = pq.ParquetFile(PARQUET)
day = np.asarray(pt.read(columns=["day"]).column("day").to_pylist(), dtype=object)
split = np.asarray(pt.read(columns=["split_time"]).column("split_time").to_pylist(), dtype=object)
m_te = split == "test"
print(f"[dbg] total={len(split)} te={int(m_te.sum())} uniq_day={np.unique(day[m_te])[:4]}")
day_te = day[m_te]
m1, m2 = day_te == DDOS_DAY, day_te != DDOS_DAY
assert int(m1.sum()) == 138145 and int(m2.sum()) == 286467, (int(m1.sum()), int(m2.sum()))

Tn_s = float(temp_scale_binary(d["va_noul_logits"], d["va_y_bin"]))
Tc_s = float(temp_scale_multi(d["va_choice_logits"], d["va_y_fam"]))
thr_s = float(np.quantile(
    sigmoid(d["va_noul_logits"].astype(np.float64) / Tn_s)[d["va_y_bin"] == 1], 0.95))


def metrics(gate_p: np.ndarray, thr: float, choice_p: np.ndarray,
            corn: np.ndarray, yb, yf, ys) -> dict:
    ok = sigmoid(corn.astype(np.float64))
    lvl = np.stack([1 - ok[:, 0], ok[:, 0] - ok[:, 1],
                    ok[:, 1] - ok[:, 2], ok[:, 2]], axis=1)
    sev_est = (lvl * np.arange(4)).sum(1)
    return {"n": len(yb), "auroc": round(e7.auc_from(yb, gate_p), 4),
            "ece15": round(ece_bin(gate_p, yb), 4),
            "brier": round(float(((gate_p - yb) ** 2).mean()), 4),
            "fpr@95": round(float((gate_p[yb == 0] >= thr).mean()), 4),
            "macro_f1": round(e7.macro_f1(yf, choice_p.argmax(1), N_CLASSES), 4),
            "ece15_choice": round(ece_multi(choice_p, yf), 4),
            "sev_mae": round(float(np.abs(sev_est - ys).mean()), 4),
            "thr@0.95cal": round(thr, 4)}


res = {"static_T": {"noul": round(Tn_s, 4), "choice": round(Tc_s, 4)},
       "per_day_static": {}}
for day_name, m in (("day1_ddos", m1), ("day2_portscan", m2)):
    gp = sigmoid(d["te_noul_logits"][m].astype(np.float64) / Tn_s).astype(np.float32)
    res["per_day_static"][day_name] = metrics(
        gp, thr_s, soft_pred(d["te_choice_logits"][m], Tc_s),
        d["te_corn_logits"][m], d["te_y_bin"][m].astype(np.int64),
        d["te_y_fam"][m].astype(np.int64), d["te_y_sev"][m].astype(np.int64))

e9j = json.loads((OUT / "e9_retrain.json").read_text())

# ── tabla ──────────────────────────────────────────────────────────────
lines = ["# E9 (per-window retrain) vs corrida estática — mismos evaluadores E8",
         "",
         "| bloque | n | AUROC | ECE15 | Brier | FPR@95 | macroF1 | ECE-ch | sevMAE |",
         "|---|---|---|---|---|---|---|---|---|"]
for day_name, r in res["per_day_static"].items():
    lines.append(f"| {day_name} static | {r['n']} | {r['auroc']} | {r['ece15']} | "
                 f"{r['brier']} | {r['fpr@95']} | {r['macro_f1']} | "
                 f"{r['ece15_choice']} | {r['sev_mae']} |")
for key, r in e9j["per_day"].items():
    lines.append(f"| {key} (e9) | {r['n']} | {r['auroc']} | {r['ece15']} | "
                 f"{r['brier']} | {r['fpr@95']} | {r['macro_f1']} | "
                 f"{r['ece15_choice']} | {r['sev_mae']} |")
md = "\n".join(lines) + "\n"
(OUT / "e9_vs_static.md").write_text(md)
(OUT / "e9_vs_static.json").write_text(json.dumps(res, indent=1))
print(md)

# ── chequeo empates día-2 E9 ──────────────────────────────────────────
y2 = e9["te2_y_bin"].astype(np.int64)
s2 = e9["te2_noul"].astype(np.float64)
p2 = sigmoid(s2)
print(f"scores únicos día-2: {len(np.unique(s2)):,} de {len(s2):,}")
print("hist [0,0.1),…:", np.histogram(p2, bins=10, range=(0, 1))[0].tolist())
print("p entre 0.45 y 0.55 (zona de empate/umbral):",
      int(((p2 > 0.45) & (p2 < 0.55)).sum()))
print("AUROC noul crudo :", round(e7.auc_from(y2, s2), 4),
      "| calibrated:", e9j["per_day"]["day2_portscan[asrun_val]"]["auroc"])
print("fraude de empate:", round(a_ := e7.auc_from(y2, np.round(p2, 3)), 4),
      "(redondeo 3d)") if False else None
# AUROC con empates forzados a 2 decimales para ver robustez
print("AUROC p2 redondeado 2dp:", round(e7.auc_from(y2, np.round(p2, 2)), 4))
print("tasa pos/pos-reclamo día-2:", float(y2.mean()))
