#!/usr/bin/env python
"""E8 — Recalibración por ventana (walk-forward) bajo drift temporal CIC.

Protocolo congelado (clona el régimen del by-day_run):
  * Test del split CIC by-day EN ORDEN DE CAPTURA (verificado: la secuencia
    label_family del parquet te == te_fam_names del npz; el test son EXACTAMENTE
    2 bloques-día: Friday-DDos n=138,145 → Friday-PortScan n=286,467).
  * Para cada día de test d, la pila de calibración W(d) es
      día 1 → val del pipeline (días previos)
      día 2 → los labels del día de test 1 (labels retrasadas al cierre del
              día — el régimen streaming honesto). W más fresca que val en
              1 ventana.
  * Variantes por día:
      static : calibra SOLO con val (como está el paper hoy).
      walk   : calibra con W(d) — pero día-1 es idéntico a static por no
               tener ventana más fresca; solo día-2 difiere.
  * Calibración aplicada:
      gate    : temperatura (NLL binaria, 1-D L-BFGS en log T).
      choice  : temperatura (NLL softmax 6-clases; el macro-F1 del argmax no
                cambia — lo que se recalibra es la confianza).
      umbral  : cuantil 0.95 de los POSITIVOS de W(d) (el mismo criterio
                repo), aplicado a los scores de hoy día.
  * Métricas: AUROC (invariante, sanity), ECE15, Brier, FPR@TPR95 con
    umbral de W, ECE15 choice (multiclase max-prob), macro-F1.

Salida: results/e8_recal/e8_recal.json + e8_recal.md
Ojo runner: no entrena ningún modelo; solo re-calibra heads congeladas.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from scipy.optimize import minimize_scalar

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import e7_vendor_api as e7

NPZ = REPO / "results/outputs_seed/cic_small_time.npz"
PARQUET = Path("${HOME}/projects/paper/data/processed/cicids2017/cicids2017.parquet")
OUT = REPO / "results/e8_recal"


# ── helpers primitivas ────────────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def soft_pred(logits: np.ndarray, T: float) -> np.ndarray:
    s = logits.astype(np.float64) / T
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def ece_bin(score: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = np.maximum(score, 1 - score)
    pred = (score >= 0.5).astype(np.int8)
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += abs(pred[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def ece_multi(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = p.max(1); pred = p.argmax(1); hit = (pred == y).astype(np.float64)
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += abs(hit[m].mean() - conf[m].mean()) * m.mean()
    return float(e)


def platt_binary(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ln = logits.astype(np.float64)
    from scipy.optimize import minimize

    def nll(w):
        a, b = w
        p = np.clip(sigmoid(a * ln + b), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    r = minimize(nll, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def temp_scale_binary(logits: np.ndarray, y: np.ndarray) -> float:
    ln = logits.astype(np.float64)

    def nll(logt: float) -> float:
        p = np.clip(sigmoid(ln / np.exp(logt)), 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(200.)),
                                        method="bounded", options={"xatol": 1e-4}).x))


def temp_scale_multi(logits: np.ndarray, y: np.ndarray) -> float:
    L = logits.astype(np.float64)

    def nll(logt: float) -> float:
        T = np.exp(logt)
        s = L / T
        s -= s.max(1, keepdims=True)
        p = np.exp(s); p /= p.sum(1, keepdims=True)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())

    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(200.)),
                                        method="bounded", options={"xatol": 1e-4}).x))


# ── carga ─────────────────────────────────────────────────────────────────────
def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ, allow_pickle=True)
    va = {k[3:]: np.asarray(z["va_" + k[3:]]) for k in z.files if k.startswith("va_")}
    te = {k[3:]: np.asarray(z["te_" + k[3:]]) for k in z.files if k.startswith("te_")}
    n_te = len(te["noul_logits"])
    assert bool(np.array_equal((va["fam_names"] != "normal").astype(np.int8), va["y_bin"]))
    assert bool(np.array_equal((te["fam_names"] != "normal").astype(np.int8), te["y_bin"]))

    # días de test en el ORDEN DE CAPTURA (los últimos n_te filas del parquet)
    pt = pq.ParquetFile(PARQUET)
    skip = pt.metadata.num_rows - n_te
    days = []
    done = 0
    for batch in pt.iter_batches(batch_size=500_000, columns=['day'], use_threads=False):
        nb = batch.num_rows
        if done + nb <= skip:
            done += nb; continue
        d = batch.column('day').to_pylist()
        if done < skip:
            d = d[skip - done:]
        days.extend(d); done += nb
    assert len(days) == n_te
    days = np.asarray(days, dtype=object)

    blocks = []
    for d in sorted(set(days), key=lambda x: int(np.where(days == x)[0][0])):
        blocks.append(np.where(days == d)[0])
    names = [str(days[b[0]]) for b in blocks]

    # temperatura estática (val), congelada para ambas variantes
    Tn_s = temp_scale_binary(va["noul_logits"], va["y_bin"])
    Tc_s = temp_scale_multi(va["choice_logits"], va["y_fam"])

    rows = []
    prev = None  # bloque te (índices) del día anterior
    for dn, b in zip(names, blocks):
        for variant in ("static", "walk", "walkplatt", "walkacc", "oracle"):
            if variant == "walk" and prev is None:
                continue  # día 1: no hay ventana fresca previa al test
            if variant == "walkacc" and prev is None:
                continue  # day2+ only
            if variant == "walkplatt" and prev is None:
                continue
            if variant == "static":
                src_n, src_y, src_c, src_yf = va["noul_logits"], va["y_bin"], va["choice_logits"], va["y_fam"]
                cal_from = "val"
                Tn, Tc = Tn_s, Tc_s
                gsrc, gsrc_y = va["noul_logits"], va["y_bin"]
                plat = (1.0, 0.0)
            elif variant == "walk":
                si = prev[1]
                src_n = te["noul_logits"][si]; src_y = te["y_bin"][si]
                src_c = te["choice_logits"][si]; src_yf = te["y_fam"][si]
                gsrc, gsrc_y = src_n, src_y
                cal_from = f"day:{prev[0]}"
                Tn = temp_scale_binary(src_n, src_y)
                Tc = temp_scale_multi(src_c, src_yf)
                plat = (1.0, 0.0)
            elif variant == "walkplatt":
                si = prev[1]
                src_n = te["noul_logits"][si]; src_y = te["y_bin"][si]
                src_c = te["choice_logits"][si]; src_yf = te["y_fam"][si]
                gsrc, gsrc_y = src_n, src_y
                cal_from = f"day:{prev[0]} [platt]"
                plat = platt_binary(src_n, src_y)
                Tn = None
                Tc = temp_scale_multi(src_c, src_yf)
            elif variant == "walkacc":
                src_n = np.concatenate([va["noul_logits"], te["noul_logits"][prev[1]]])
                src_y = np.concatenate([va["y_bin"], te["y_bin"][prev[1]]])
                src_c = np.concatenate([va["choice_logits"], te["choice_logits"][prev[1]]])
                src_yf = np.concatenate([va["y_fam"], te["y_fam"][prev[1]]])
                gsrc, gsrc_y = src_n, src_y
                cal_from = "val+day"
                Tn = temp_scale_binary(src_n, src_y)
                Tc = temp_scale_multi(src_c, src_yf)
                plat = (1.0, 0.0)
            elif variant == "oracle":  # calibra con las PROPIAS labels del día (techo)
                src_n = te["noul_logits"][b]; src_y = te["y_bin"][b]
                src_c = te["choice_logits"][b]; src_yf = te["y_fam"][b]
                gsrc, gsrc_y = src_n, src_y
                cal_from = "oracle(day-self)"
                plat = platt_binary(src_n, src_y)
                Tn = None
                Tc = temp_scale_multi(src_c, src_yf)

            if variant == "walkplatt" or variant == "oracle":
                ln = te["noul_logits"][b].astype(np.float64)
                gprob = sigmoid(plat[0] * ln + plat[1])
                thr = float(np.quantile(sigmoid(plat[0] * gsrc.astype(np.float64) + plat[1])[gsrc_y == 1], 0.95))
                Tn_repr = f"a={plat[0]:.2f},b={plat[1]:.2f}"
            else:
                gprob = sigmoid(te["noul_logits"][b].astype(np.float64) / Tn)
                thr = float(np.quantile(sigmoid(gsrc.astype(np.float64) / Tn)[gsrc_y == 1], 0.95))
                Tn_repr = round(Tn, 3)
            fpr = float((gprob[te["y_bin"][b] == 0] >= thr).mean())
            gprob32 = gprob.astype(np.float32)[:len(b)]
            yb = te["y_bin"][b]
            fams_idx = te["y_fam"][b]
            Tc_eff = Tc if variant != "static" else Tc_s
            pc = soft_pred(te["choice_logits"][b], Tc_eff)
            rows.append({
                "day": dn, "variant": variant, "n": len(b), "cal_from": cal_from,
                "auroc": round(e7.auc_from(yb, gprob32), 4),
                "ece15": round(ece_bin(gprob32, yb), 4),
                "brier": round(float(((gprob32 - yb) ** 2).mean()), 4),
                "fpr@95_thrcal": round(fpr, 4),
                "T_noul": Tn_repr,
                "ece15_choice": round(ece_multi(pc, fams_idx), 4),
                "macro_f1": round(e7.macro_f1(fams_idx, pc.argmax(1), te["choice_logits"].shape[1]), 4),
                "thr0.95_from": round(thr, 4),
            })
        prev = (dn, b)

    agg = {}
    for variant in ("static", "walk"):
        rs = [r for r in rows if r["variant"] == variant]
        agg[variant] = {
            "n_days": len(rs),
            "ece15_mean": round(float(np.mean([r["ece15"] for r in rs])), 4),
            "brier_mean": round(float(np.mean([r["brier"] for r in rs])), 4),
            "fpr@95_mean": round(float(np.mean([r["fpr@95_thrcal"] for r in rs])), 4),
            "ece15_choice_mean": round(float(np.mean([r["ece15_choice"] for r in rs])), 4),
        }
    out = {"protocol": {"npz": str(NPZ), "windows": names,
                        "recal": "temperature per window + quantile threshold per window"},
           "per_day": rows, "agg": agg}
    (OUT).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "e8_recal.json", "w"), indent=1)

    # tabla MD legible
    lines = ["| model | window | cal-src | AUROC | ECE15 | Brier | FPR@95(cal) | T_noul | ECE15-ch | macro-F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {r['day'][:24]} | {r['cal_from']} | {r['auroc']} | "
                     f"{r['ece15']} | {r['brier']} | {r['fpr@95_thrcal']} | {r['T_noul']} | "
                     f"{r['ece15_choice']} | {r['macro_f1']} |")
    (OUT / "e8_recal.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
