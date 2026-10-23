#!/usr/bin/env python
"""E5-fix — pierna Phase-2 honesta para la cascada: modelo POR-DÍA.

El paper/cascada original (src/e5/cascade.py, NPZ=cic_small_strat) estima
F2 con el modelo STRATIFICADO: su train contiene filas de TODAS las
familias de TODOS los días (in-distribución por familia). El informe
externo (falla crítica 1) señala que el headline "F2 recupera la zona
gris" corresponde a ese régimen, no al despliegue por día.

Este script re-estima F2 por día con el artefacto BY-DAY
(results/outputs_seed/cic_small_time.npz): train = días previos (split
time, te = bloques viernes en orden de captura), el modelo NUNCA vio las
familias atacantes del viernes. Solo aplicable a los días viernes
(day_idx 6 parcial-test y 7): los días Mon–Thu del modelo por-día son
train (no hay logits de test honestos).

Protocolo idéntico al resto de la cascada (per-5tupla, slack 2/5 s,
independencia documentada). Salida: results/e5/byday_phase2.json + md.
Uso: PYTHONPATH=. python scripts/e5_phase2_byday.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.e5.cascade import (DAY_FILES, ENH, PROCESSED, load_alerts, load_day,
                            match_f1, prf, OUT_DIR)

TIME_NPZ = pathlib.Path("results/outputs_seed/cic_small_time.npz")
STRAT_NPZ = pathlib.Path("results/outputs/cic_small_strat.npz")
OUT = pathlib.Path("results/e5/byday_phase2.json")
MD = pathlib.Path("results/e5/byday_phase2.md")

FAM_MAP = {"BENIGN": "normal", "DoS Hulk": "dos_ddos", "DDoS": "dos_ddos",
           "DoS GoldenEye": "dos_ddos", "DoS slowloris": "dos_ddos",
           "DoS Slowhttptest": "dos_ddos", "PortScan": "recon",
           "FTP-Patator": "bruteforce", "SSH-Patator": "bruteforce",
           "Bot": "botnet", "Infiltration": "infiltration",
           "Heartbleed": "exploitation",
           "Web Attack – Brute Force": "webattack",
           "Web Attack – XSS": "webattack",
           "Web Attack – Sql Injection": "webattack"}


def load_f2_time(day: str) -> dict:
    """Filas test-del-día bajo split_time + probs noul del npz por-día.

    Nota honestidad: para el day_idx 6 (viernes-DDos) solo 138,145 filas
    son test del split por-día; las filas val restantes NO tienen logits
    de un modelo que las excluya → se excluyen del estimador (subconjunto
    documentado). Para day_idx 7 son todas.
    """
    d_idx = DAY_FILES[day]
    tbl = pq.read_table(PROCESSED, columns=["day_idx", "split_time"])
    dcol = np.asarray(tbl.column("day_idx").to_pylist())
    scol = np.asarray(tbl.column("split_time").to_pylist())
    test_all = np.flatnonzero(scol == "test")
    rows = np.flatnonzero(np.isin(scol, ["test"]) & np.isin(dcol, d_idx))
    npz_pos = np.searchsorted(test_all, rows)
    parts = np.load(TIME_NPZ, allow_pickle=True)
    meta = json.loads(TIME_NPZ.with_suffix(".json").read_text())
    T = meta["temperatures"]["noul"]
    p_noul = 1.0 / (1.0 + np.exp(-parts["te_noul_logits"][npz_pos] / T))
    fam_counts = json.loads(TIME_NPZ.with_suffix(".json").read_text())["candidates"]
    y_fam = parts["te_y_fam"][npz_pos]
    return {"fam": [fam_counts[i] if i >= 0 else "otros" for i in y_fam],
            "flag": p_noul >= 0.5, "p": p_noul}


def combined_estimate(f1_flag: np.ndarray, y: np.ndarray, fams: list[str],
                      flows_fam: np.ndarray, f2: dict) -> dict:
    f2_fam = np.array(f2["fam"])
    f2_flag = np.array(f2["flag"])
    y2 = f2_fam != "normal"
    per_fam_recall = {f: (float(f2_flag[f2_fam == f].mean())
                          if (f2_fam == f).sum() >= 10 else None)
                      for f in fams if f != "normal"}
    f2_fpr = float(f2_flag[~y2].mean()) if (~y2).sum() else 0.0
    est_tp = int(f1_flag[y].sum())
    est_fp = int(f1_flag[~y].sum())
    gz = ~f1_flag
    for f, rec in per_fam_recall.items():
        if rec is None:
            continue
        m = gz & (flows_fam == f)
        est_tp += int(round(m.sum() * rec))
    est_fp += int(round((gz & (flows_fam == "normal")).sum() * f2_fpr))
    n_atk, n_ben = int(y.sum()), int((~y).sum())
    est = prf(est_tp, est_fp, n_atk - est_tp, n_ben - est_fp)
    return {"f2_recall_per_family": {k: v for k, v in per_fam_recall.items()},
            "f2_fpr_benign": round(f2_fpr, 4),
            "f2_acc": float(f2_flag.mean()) if len(f2_flag) else 0.0,
            "combined_est": est}


def main() -> None:
    out = {"note": ("Phase-2 honesta: logits del modelo por-día "
                    "(cic_small_time.npz, te=viernes en orden de captura); "
                    "día DDos usa solo sus filas test del split time."),
           "days": {}}
    for day in ("2017-07-06", "2017-07-07"):
        alerts = load_alerts(OUT_DIR / f"eve_{day}.json.gz")
        flows = load_day(day)
        f1_flag, _ = match_f1(flows, alerts)
        n = len(f1_flag)
        fams = sorted(set(flows["fam"]))
        y = np.array([f != "normal" for f in flows["fam"]])
        f1_acc = prf(int((f1_flag & y).sum()), int((f1_flag & ~y).sum()),
                     int((~f1_flag & y).sum()), int((~f1_flag & ~y).sum()))
        by = combined_estimate(f1_flag, y, fams,
                               np.array(flows["fam"]), load_f2_time(day))
        st = load_f2_strat(day)
        if st is not None:
            strat = combined_estimate(f1_flag, y, fams,
                                      np.array(flows["fam"]), st)
        else:
            strat = None
        out["days"][day] = {"n_flows": n, "n_attack": int(y.sum()),
                            "f1": f1_acc, "f2_byday": by, "f2_strat": strat}
        print(f"[e5fix] {day}: F1 R={f1_acc['recall']} | "
              f"byday comb R={by['combined_est']['recall']} "
              f"P={by['combined_est']['precision']} "
              f"acc={by['combined_est']['accuracy']}"
              + (f" | strat comb R={strat['combined_est']['recall']}" if strat else ""))

    OUT.write_text(json.dumps(out, indent=1))
    md = ["# E5 — Phase 2 por-día (honesto) vs strat", "",
          "| día | F1 R | F2(byday) acc | comb(byday) P/R | comb(strat) P/R |",
          "|---|---|---|---|---|"]
    for d, v in out["days"].items():
        b, s = v["f2_byday"], v["f2_strat"]
        md.append(f"| {d} | {v['f1']['recall']} | {b['f2_acc']:.4f} | "
                  f"{b['combined_est']['precision']:.4f}/{b['combined_est']['recall']:.4f} | "
                  + (f"{s['combined_est']['precision']:.4f}/{s['combined_est']['recall']:.4f} |"
                     if s else "— |"))
    MD.write_text("\n".join(md) + "\n")
    print(f"[e5fix] → {OUT}")


def load_f2_strat(day: str):
    """Fase-2 strat (in-distribución) vía src.e5.cascade, NPZ absoluto si hace falta."""
    import src.e5.cascade as casc  # noqa: PLC0415
    if not pathlib.Path(casc.NPZ).exists():
        alt = pathlib.Path("/home/jsantillana/projects/paper") / casc.NPZ
        if alt.exists():
            casc.NPZ = str(alt)
        else:
            return None
    return casc.load_f2(day)


if __name__ == "__main__":
    main()
