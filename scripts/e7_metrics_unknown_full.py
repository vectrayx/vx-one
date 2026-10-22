#!/usr/bin/env python
"""Métricas E7-long: run preguntas CON unknown sobre el sample COMPLETO
(n=2,021 = mismas filas de la run principal → comparación PAIRED).

Input: results/e7_vendors/opencode/unkfull/e7_unkfull_*.jsonl (dedupe por k,
meta k=-1 excluida). Salida: vendor_jev_free_unkfull.json (estructura espejo
del vendor_jev_free_ablation.json) + comparación main-run en las mismas
filas (jev_free_partial.jsonl).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import e7_vendor_api as e7  # noqa: E402
from src.eval.metrics import fpr_at_tpr  # noqa: E402

UNK = REPO / "results/e7_vendors/opencode/unkfull"
OUT = UNK / "vendor_jev_free_unkfull.json"

meta = json.load(open(e7.META))
cand = meta["candidates"]
Tn = meta["temperatures"]["noul"]; Ts = meta["temperatures"]["score"]
fam_idx = {f: i for i, f in enumerate(cand)}

npz = np.load(e7.NPZ)
tbl = pq.read_table(e7.PARQUET, columns=["split", "label_family", "label_severity"])
fam_all = np.asarray(tbl.column("label_family").to_pylist())
te = np.asarray(tbl.column("split").to_pylist()) == "test"
fam_te_all = fam_all[te]
sev_all = np.asarray(tbl.column("label_severity").to_pylist())[te]

# ── leer unkfull (dedupe por k, última escritura) ─────────────────────
recs = {}
for p in sorted(UNK.glob("e7_unkfull_*.jsonl")):
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["k"] < 0:            # meta line (fue pedida por error)
            continue
        recs[int(r["k"])] = r
idx = np.array(sorted(recs), dtype=np.int64)
y = (fam_te_all[idx] != "normal").astype(np.int8)
sev_true = sev_all[idx].astype(np.int64)
n = len(idx)
fams_true = fam_te_all[idx]

n_gate = np.full(n, np.nan)
fam_p = np.zeros((n, len(cand)))
unk_p = np.full(n, np.nan)
sev_p = np.zeros((n, 4))
lats, toks = [], []
pos = {int(k): j for j, k in enumerate(idx)}
for k, r in recs.items():
    j = pos[k]
    n_gate[j] = r["gate"]
    for name_, p in (r["family"] or {}).items():
        if name_ == "unknown":
            unk_p[j] = p
        elif name_ in fam_idx:
            fam_p[j, fam_idx[name_]] = p
    for lv, p in (r["sev"] or {}).items():
        sev_p[j, int(lv)] = p
    lats.append(r["lat_ms"]); toks.append(r["tok"])

# renormaliza fam_p sobre los candidatos si el vendor repartió en unknown
s = fam_p.sum(1, keepdims=True)
fam_p = fam_p / np.clip(s, 1e-12, None)

src = np.load(e7.NPZ)
our_gate = 1 / (1 + np.exp(-src["te_noul_logits"][idx] / Tn))
our_choice = src["te_choice_logits"][idx].argmax(1)
sl = src["te_corn_logits"][idx]
okp = 1 / (1 + np.exp(-sl / Ts))
lvl = np.stack([1 - okp[:, 0], okp[:, 0] - okp[:, 1],
                okp[:, 1] - okp[:, 2], okp[:, 2]], 1)
our_sev = (lvl * np.arange(4)).sum(1)
label_idx = np.array([fam_idx[f] for f in fams_true])

# predicción vendor con Candidatos+unknown como clase extra
vendor_argmax = fam_p.argmax(1)
vendor_with_unk = np.where(unk_p > fam_p.max(1), len(cand), vendor_argmax)

labels_with_unk = np.array([fam_idx[f] if f in fam_idx else len(cand)
                            for f in fams_true])
k_all = len(cand) + 1

res = {"vendor": "opencode", "model": "jev-1.13-free",
       "rows_done": int(n), "rows_total": 2021, "sample": "main (1000 normal + 1021 attack, seed 42) = PAIRED con la run principal",
       "choice_unknown_option": {
           "vendor_macro_f1_all_rows": round(e7.macro_f1(
               labels_with_unk, vendor_with_unk, k_all), 4),
           "vendor_macro_f1_known_rows": round(e7.macro_f1(
               label_idx, vendor_argmax, len(cand)), 4),
           "our_macro_f1": round(e7.macro_f1(label_idx, our_choice, len(cand)), 4),
           "unknown_rate_overall": round(float(np.mean(vendor_with_unk == len(cand))), 6),
           "unknown_rate_attack": round(float(np.mean((vendor_with_unk == len(cand))[y == 1])), 6),
           "unknown_rate_normal": round(float(np.mean((vendor_with_unk == len(cand))[y == 0])), 6),
           "unknown_prob_mean_attack": round(float(np.nanmean(unk_p[y == 1])), 4),
           "unknown_prob_mean_normal": round(float(np.nanmean(unk_p[y == 0])), 4),
       },
       "gate": {"vendor": {"auroc": round(e7.auc_from(y, n_gate), 4),
                           "fpr@95": round(fpr_at_tpr(n_gate, y, 0.95), 4),
                           "ece15": round(e7.ece15(n_gate, y), 4)},
                "ours": {"auroc": round(e7.auc_from(y, our_gate), 4),
                         "fpr@95": round(fpr_at_tpr(our_gate, y, 0.95), 4),
                         "ece15": round(e7.ece15(our_gate, y), 4)}},
       "severity": {"vendor_mae": round(float(np.abs(sev_p @ np.arange(4) - sev_true).mean()), 4),
                    "our_mae": round(float(np.abs(our_sev - sev_true).mean()), 4)},
       "latency_ms": {"p50": round(float(np.percentile(lats, 50)), 1),
                      "p95": round(float(np.percentile(lats, 95)), 1)},
       "tokens": {"total": int(sum(toks)), "usd_if_paid": round(sum(toks) * 0.042 / 1e6, 4)}}

# comparación paired con la run principal (mismas filas, sin unknown)
main = {}
murl = REPO / "results/e7_vendors/opencode/jev_free_partial.jsonl"
for line in murl.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    main[int(r["k"])] = r
idx_m = np.array(sorted(main), dtype=np.int64)
inter, ai, ci = np.intersect1d(idx, idx_m, return_indices=True)
y_i = y[ci]
fam_p_m = np.zeros((len(inter), len(cand)))
for j, k in enumerate(inter):
    for name_, p in (main[int(k)]["family"] or {}).items():
        if name_ in fam_idx:
            fam_p_m[j, fam_idx[name_]] = p
gate_m = np.array([main[int(k)]["gate"] for k in inter])
our_gate_m = 1 / (1 + np.exp(-src["te_noul_logits"][inter] / Tn))
fams_i = fams_true[ci]
label_i = np.array([fam_idx[f] for f in fams_i])
res["paired_vs_main"] = {
    "n_rows_paired": int(len(inter)),
    "gate_vendor_auroc": round(e7.auc_from(y_i, gate_m), 4),
    "gate_vendor_auroc_unkfull": round(e7.auc_from(y_i, n_gate[ci]), 4),
    "gate_our_auroc": round(e7.auc_from(y_i, our_gate_m), 4),
    "vendor_macro_f1_main": round(e7.macro_f1(label_i, fam_p_m.argmax(1), len(cand)), 4),
    "vendor_macro_f1_unkfull": res["choice_unknown_option"]["vendor_macro_f1_known_rows"],
    "notes": "mismas filas (paired); la única diferencia es la opción unknown en family"}

OUT.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
