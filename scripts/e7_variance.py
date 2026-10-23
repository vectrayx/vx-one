#!/usr/bin/env python
"""E7-var — varianza de las llamadas vendor (informe, ítem 10): N=5 llamadas
por fila sobre la MISMA muestra protocolar (seed 42, 2,021 filas), para ambos
vendors. Mide:
  * dispersión fila-nivel: σ del noul por fila, tasa de desacuerdo del argmax
    choice, σ del severity score;
  * dispersión de métricas: AUROC/ECE/macro-F1/MAE re-computados por repeat
    (5 valores → mean±σ).
Salida: results/e7_vendors/variance_{vendor}.json

Uso:
    PYTHONPATH=. python scripts/e7_variance.py --vendor azure   (~2.6 h)
    PYTHONPATH=. python scripts/e7_variance.py --vendor opencode --n-normal 75 --n-attack 75
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent.parent),
                str(Path(__file__).resolve().parent.parent / "src")]

from scripts.e7_vendor_api import (SEED, NPZ, META, PARQUET, OUT,
                                   stratified_attack_quota, build_state,
                                   build_questions, ask_one, _key_for,
                                   auc_from, ece15, macro_f1, best_fpr_at_tpr95)
import pyarrow.parquet as pq


def load_sample(n_normal: int, n_attack: int):
    tbl = pq.read_table(PARQUET, columns=["x_num", "split", "label_family",
                                          "label_severity", "x_txt_proto",
                                          "x_txt_service", "x_txt_state"])
    meta = json.load(open(META))
    cand = meta["candidates"]
    fam_all = np.asarray(tbl.column("label_family").to_pylist())
    te = np.asarray(tbl.column("split").to_pylist()) == "test"
    fam_te = fam_all[te]
    y_bin = (fam_te != "normal").astype(np.int8)
    sev_te = np.asarray(tbl.column("label_severity").to_pylist())[te].astype(np.int64)
    x = tbl.column("x_num").to_pylist()
    proto = np.asarray(tbl.column("x_txt_proto").to_pylist())[te]
    service = np.asarray(tbl.column("x_txt_service").to_pylist())[te]
    state_c = np.asarray(tbl.column("x_txt_state").to_pylist())[te]
    rng = np.random.default_rng(SEED)
    idx_n = np.where(y_bin == 0)[0]
    idx_a = np.where(y_bin == 1)[0]
    q_att = stratified_attack_quota(fam_te[idx_a], n_attack)
    idx_a_s = idx_a[q_att]
    idx_n_s = rng.choice(idx_n, size=min(n_normal, len(idx_n)), replace=False)
    idx = np.sort(np.r_[idx_n_s, idx_a_s])
    rows = [{"x": x[i], "proto": str(proto[i]), "service": str(service[i]),
             "state": str(state_c[i])} for i in idx]
    return idx, rows, y_bin, sev_te, fam_te, cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", choices=["azure", "opencode", "typesafe"], required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-normal", type=int, default=1010)
    ap.add_argument("--n-attack", type=int, default=1011)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    model = {"azure": "decision-1", "opencode": "jev-1.13-free",
             "typesafe": "jev-1.13.0"}[args.vendor]
    idx, rows, y_bin, sev_te, fam_te, cand = load_sample(args.n_normal, args.n_attack)
    questions = build_questions(cand)
    api_key = _key_for(args.vendor)
    n = len(idx)
    R = args.repeats
    print(f"[var] {args.vendor}: n={n} filas × {R} repeats")

    noul = np.full((R, n), np.nan)
    fam_arg = np.full((R, n), -2, dtype=np.int64)
    sev_sc = np.full((R, n), np.nan)
    fam_idx = {f: i for i, f in enumerate(cand)}

    def work(t):
        r, k = t
        for attempt in range(3):  # tolerante a ráfagas: reintenta el par (r,k)
            try:
                resp, lat, _ = ask_one(args.vendor, model, build_state(rows[k]),
                                       questions, api_key, tries=10)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [var] fila {k} repeat {r} FALLÓ {e!r}", flush=True)
                    return r, k, float("nan"), -1, float("nan")
                time.sleep(30 * (attempt + 1))
        a = resp["answers"]
        g = a["gate"]["noul"]
        probs = a["family"]["probabilities"]
        best, bestp = None, -1.0
        for name, p in probs.items():
            if name in fam_idx and p > bestp:
                best, bestp = fam_idx[name], p
        sv = a["severity"]["score"]
        return r, k, g, best if best is not None else -1, sv

    t0 = time.time()
    tasks = [(r, k) for r in range(R) for k in range(n)]
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for r, k, g, fb, sv in ex.map(work, tasks):
            noul[r, k] = g
            fam_arg[r, k] = fb
            sev_sc[r, k] = sv
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    y = y_bin[idx].astype(int)
    label_idx = np.array([{f: i for i, f in enumerate(cand)}[f] for f in fam_te[idx]])
    per_repeat = []
    for r in range(R):
        m = ~np.isnan(noul[r])
        auc = auc_from(y[m], noul[r][m])
        ece = ece15(noul[r][m], y[m])
        f1 = macro_f1(label_idx[m], fam_arg[r][m], len(cand))
        mae = float(np.abs(sev_sc[r] - sev_te[idx]).mean())
        per_repeat.append({"repeat": r, "auroc": round(auc, 4),
                           "ece15": round(ece, 4), "macro_f1": round(f1, 4),
                           "sev_mae": round(mae, 4),
                           "n_calls_ok": int(m.sum())})
        print(f"  repeat {r}: auroc={auc:.4f} ece={ece:.4f} f1={f1:.4f} mae={mae:.4f}")

    m = ~np.isnan(noul)
    row_std = float(np.nanmean(np.nanstd(noul, axis=0)))
    arg_dis = float(np.mean([len(set(fam_arg[:, k][fam_arg[:, k] >= 0])) > 1
                             for k in range(n)]))
    sev_std = float(np.nanmean(np.nanstd(sev_sc, axis=0)))
    import statistics as st
    agg = {}
    for key in ("auroc", "ece15", "macro_f1", "sev_mae"):
        vals = [p[key] for p in per_repeat]
        agg[key] = {"mean": round(st.mean(vals), 4),
                    "stdev": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
                    "min": min(vals), "max": max(vals)}
    out = {"vendor": args.vendor, "model": model, "repeats": R,
           "n_rows": n, "sample": {"seed": SEED, "stratified": True},
           "row_level": {"mean_row_std_noul": round(row_std, 5),
                         "choice_argmax_disagreement_rate": round(arg_dis, 4),
                         "mean_row_std_severity": round(sev_std, 5)},
           "per_repeat": per_repeat, "metrics_agg": agg,
           "wall_s": round(time.time() - t0, 1)}
    p = OUT / f"variance_{args.vendor}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[var] → {p}")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
