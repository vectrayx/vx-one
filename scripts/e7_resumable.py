#!/usr/bin/env python
"""Runner E7 resumible + rate-limited sobre OpenCode Zen (jev-1.13-free).

Diferencias respecto a e7_vendor_api.py (full-run pago):
- Secuencial con pacing adaptativo y honoring del retry-after; un cooldown
  largo (cooldown de horas del free tier, verificado 2026-10-20) no mata el
  run: duerme hasta el reset y sigue (hasta --max-wait).
- Resultados incrementales en .jsonl (1 línea por fila; flush por línea) →
  resume omitiendo filas ya respondidas. La muestra es determinista (SEED
  42; idx fijo en session_idx.npz), así que el resume es válido.
- Prints unbuffered con progreso cada 25 filas.
- --metrics: recalcula métricas vendor-vs-nuestro desde el .jsonl sin red.

Uso:
    OPENCODE_API_KEY=oc_sk... python scripts/e7_resumable.py
    OPENCODE_API_KEY=oc_sk... python scripts/e7_resumable.py --pace 2
    python scripts/e7_resumable.py --metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import e7_vendor_api as e7  # noqa: E402
from src.eval.metrics import fpr_at_tpr  # noqa: E402

OUT = e7.OUT / "opencode"
JSONL = OUT / "jev_free_partial.jsonl"
IDX_NPZ = OUT / "session_idx.npz"
SEV_P = None  # relleno


def ask(vendor: str, model: str, state: dict, questions: dict, key: str):
    base = e7.VENDORS[vendor]["base"]
    body = json.dumps({"model": model, "state": state, "questions": questions},
                      separators=(",", ":")).encode()
    req = urllib.request.Request(
        f"{base}/v1/systemone", data=body, method="POST",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "vxone-e7/0.1"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read()), (time.perf_counter() - t0) * 1000


def build_sample(n_normal: int, n_attack: int):
    """Muestra idéntica a distinta semilla que e7_vendor_api.py (idx fijos)."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(e7.PARQUET, columns=["x_num", "split", "label_family",
                                             "label_severity", "x_txt_proto",
                                             "x_txt_service", "x_txt_state"])
    fam_all = np.asarray(tbl.column("label_family").to_pylist())
    te = np.asarray(tbl.column("split").to_pylist()) == "test"
    fam_te = fam_all[te]
    proto = np.asarray(tbl.column("x_txt_proto").to_pylist())[te]
    service = np.asarray(tbl.column("x_txt_service").to_pylist())[te]
    state_c = np.asarray(tbl.column("x_txt_state").to_pylist())[te]
    x = tbl.column("x_num").to_pylist()

    npz = np.load(e7.NPZ)
    y_bin = (fam_te != "normal").astype(np.int8)
    assert (npz["te_y_bin"] == y_bin).all(), "alineación npz↔parquet"

    rng = np.random.default_rng(e7.SEED)
    idx_a = np.where(y_bin == 1)[0]
    q_att = e7.stratified_attack_quota(fam_te[idx_a], n_attack)
    idx_a_s = idx_a[q_att]
    idx_n = np.where(y_bin == 0)[0]
    idx_n_s = rng.choice(idx_n, size=min(n_normal, len(idx_n)), replace=False)
    idx = np.sort(np.r_[idx_n_s, idx_a_s])
    rows = [{"x": x[i], "proto": str(proto[i]), "service": str(service[i]),
             "state": str(state_c[i])} for i in idx]
    return idx, rows, fam_te, y_bin


def run(args) -> int:
    meta = json.load(open(e7.META))
    cand = meta["candidates"]
    OUT.mkdir(parents=True, exist_ok=True)

    idx, rows, fam_te, y_bin = build_sample(args.n_normal, args.n_attack)
    np.savez_compressed(IDX_NPZ, idx=idx)
    questions = e7.build_questions(cand, with_unknown=args.with_unknown)
    key = os.environ["OPENCODE_API_KEY"]

    done = {}
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["k"]] = r
    todo = [k for k in range(len(idx)) if k not in done]
    print(f"filas: {len(idx)} | hechas: {len(done)} | por hacer: {len(todo)}",
          flush=True)
    if not todo:
        print("todo resuelto; usa --metrics", flush=True)
        return 0

    consec_err = 0
    t0 = time.time()
    with open(JSONL, "a") as fh:
        for j, k in enumerate(todo):
            s = e7.build_state(rows[k])
            resp = None; lat = None
            backoff = 2.0
            for attempt in range(20):
                try:
                    resp, lat = ask("opencode", "jev-1.13-free", s, questions, key)
                    break
                except urllib.error.HTTPError as ex:
                    ra_h = ex.headers.get("retry-after")
                    ra = None
                    if ra_h:
                        try:
                            ra = float(ra_h)
                        except ValueError:
                            ra = None
                    print(f"  HTTP {ex.code} retry-after={ra} "
                          f"(consecutivos {consec_err + 1}) fila {j+1}/{len(todo)}",
                          flush=True)
                    if ex.code == 429 and ra and ra > 600:
                        if ra > args.max_wait:
                            raise RuntimeError(
                                f"cooldown {ra:.0f}s > max-wait {args.max_wait}; "
                                "aborto limpio (re-run para resumir)")
                        print(f"  cooldown largo: duermo {ra/60:.1f} min", flush=True)
                        time.sleep(ra + 2)
                        continue
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 90)
                except Exception:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 90)
            if resp is None:
                raise RuntimeError("attempts agotados; aborto limpio (resume ok)")
            consec_err = 0
            ans = resp["answers"]
            rec = {"k": k,
                   "gate": ans.get("gate", {}).get("noul"),
                   "family": ans.get("family", {}).get("probabilities"),
                   "sev": ans.get("severity", {}).get("probabilities"),
                   "lat_ms": round(lat, 1),
                   "tok": resp["usage"]["input_tokens"],
                   "model": resp.get("model")}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (j + 1) % 25 == 0:
                print(f"  {j+1}/{len(todo)} ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
            time.sleep(args.pace)
    print("RUN COMPLETE", flush=True)
    return 0


def metrics(no_op_codekey: bool = True) -> dict:
    npz_idx = np.load(IDX_NPZ)["idx"]
    meta = json.load(open(e7.META))
    cand = meta["candidates"]; Tn = meta["temperatures"]["noul"]; Ts = meta["temperatures"]["score"]
    fam_idx = {f: i for i, f in enumerate(cand)}

    npz = np.load(e7.NPZ)
    # reconstruye etiquetas de la muestra desde el PROPIO session_idx:
    import pyarrow.parquet as pq
    tbl = pq.read_table(e7.PARQUET, columns=["split", "label_family", "label_severity"])
    fam_all = np.asarray(tbl.column("label_family").to_pylist())
    te = np.asarray(tbl.column("split").to_pylist()) == "test"
    fam_te_all = fam_all[te]
    sev_all = np.asarray(tbl.column("label_severity").to_pylist())[te]
    fam_te = fam_te_all[npz_idx]
    y = (fam_te != "normal").astype(np.int8)
    sev_true = sev_all[npz_idx].astype(np.int64)

    n = len(npz_idx)
    n_gate = np.full(n, np.nan)
    fam_p = np.zeros((n, len(cand)))
    sev_p = np.zeros((n, 4))
    lats, toks = [], []
    if JSONL.exists():
        for line in JSONL.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = r["k"]
            n_gate[k] = r["gate"]
            for name_, p in (r["family"] or {}).items():
                if name_ in fam_idx:
                    fam_p[k, fam_idx[name_]] = p
            for lv, p in (r["sev"] or {}).items():
                sev_p[k, int(lv)] = p
            lats.append(r["lat_ms"]); toks.append(r["tok"])

    src = np.load(e7.NPZ)
    idx = npz_idx
    our_gate = 1 / (1 + np.exp(-src["te_noul_logits"][idx] / Tn))
    our_choice = src["te_choice_logits"][idx].argmax(1)
    our_sev = None
    sl = src["te_corn_logits"][idx]
    if sl.ndim == 2 and sl.shape[1] == 3:
        okp = 1 / (1 + np.exp(-sl / Ts))
        lvl = np.zeros((len(idx), 4))
        lvl[:, 0] = 1 - okp[:, 0]
        lvl[:, 1] = okp[:, 0] - okp[:, 1]
        lvl[:, 2] = okp[:, 1] - okp[:, 2]
        lvl[:, 3] = okp[:, 2]
        our_sev = (lvl * np.arange(4)).sum(1)
    label_idx = np.array([fam_idx[f] for f in fam_te[:n]])

    n_gate_ = np.isfinite(n_gate)
    sub = n_gate_
    res = {
        "vendor": "opencode", "model": "jev-1.13-free",
        "rows_done": int(sub.sum()), "rows_total": n,
        "complete": bool(sub.sum() == n),
        "gate": {
            "vendor": {"auroc": e7.auc_from(y[sub], n_gate[sub]),
                       "fpr@95": fpr_at_tpr(n_gate[sub], y[sub], 0.95),
                       "ece15": e7.ece15(n_gate[sub], y[sub])},
            "ours": {"auroc": e7.auc_from(y[sub], our_gate[sub]),
                     "fpr@95": fpr_at_tpr(our_gate[sub], y[sub], 0.95),
                     "ece15": e7.ece15(our_gate[sub], y[sub])}},
        "choice": {"vendor_macro_f1": e7.macro_f1(label_idx[sub],
                                                  fam_p[sub].argmax(1), len(cand)),
                   "our_macro_f1": e7.macro_f1(label_idx[sub], our_choice[sub],
                                               len(cand))},
        "severity": {"vendor_mae": float(np.abs(sev_p @ np.arange(4) - sev_true).mean())
                                     if sub.any() else None,
                     "our_mae": float(np.abs(our_sev - sev_true).mean())},
        "latency_ms": {"p50": float(np.percentile(lats, 50)) if lats else None,
                       "p95": float(np.percentile(lats, 95)) if lats else None},
        "tokens": {"total": int(sum(toks)),
                   "usd_if_paid": sum(toks) * 0.042 / 1e6},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "metrics_partial.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-normal", type=int, default=1000)
    ap.add_argument("--n-attack", type=int, default=1000)
    ap.add_argument("--with-unknown", action="store_true")
    ap.add_argument("--pace", type=float, default=1.2,
                    help="segundos entre requests (default 1.2 ≈ 50 RPM)")
    ap.add_argument("--max-wait", type=float, default=26 * 3600)
    ap.add_argument("--metrics", action="store_true",
                    help="sin red: recalcula desde el .jsonl")
    args = ap.parse_args()
    if args.metrics:
        print(json.dumps(metrics(), indent=1))
    else:
        sys.exit(run(args))
