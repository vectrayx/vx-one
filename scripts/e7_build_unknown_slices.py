#!/usr/bin/env python
"""Construye slices para E7-long con unknown (n=2,021, mismas filas de la
run principal pero questions family con opción `unknown`).

- idx/rows: mismos de la run principal (build_sample re-fijado seed 42,
  assert alineación npz↔parquet).
- meta line k=-1 con questions con unknown (e7_vendor_api.build_questions).
- slices: rest[i::N] entre los archivos de salida slice_00..NN.jsonl.

Uso: /tmp/opencode/venv/bin/python scripts/e7_build_unknown_slices.py [--out DIR] [--n 13]
Salida: /tmp/opencode/slices_unknown/slice_XX.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import e7_vendor_api as e7        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/opencode/slices_unknown")
ap.add_argument("--n", type=int, default=13)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# Selección ligera (idéntica a build_sample de e7_resumable: mismo orden de
# llamadas rng — quota usa SU propio rng con SEED y la elección de normals
# también; sin leer x_num completo).
import pyarrow.parquet as pq
import numpy as np

tbl = pq.read_table(e7.PARQUET, columns=["split", "label_family",
                                         "label_severity"])
fam_all = np.asarray(tbl.column("label_family").to_pylist())
te = np.asarray(tbl.column("split").to_pylist()) == "test"
fam_te = fam_all[te]
y_bin = (fam_te != "normal").astype(np.int8)

npz = np.load(e7.NPZ)
assert (npz["te_y_bin"] == y_bin).all(), "alineación npz↔parquet"

rng = np.random.default_rng(e7.SEED)
idx_a = np.where(y_bin == 1)[0]
q_att = e7.stratified_attack_quota(fam_te[idx_a], 1000)
idx_a_s = idx_a[q_att]
idx_n = np.where(y_bin == 0)[0]
idx_n_s = rng.choice(idx_n, size=min(1000, len(idx_n)), replace=False)
idx = np.sort(np.r_[idx_n_s, idx_a_s])
print(f"sample: {len(idx)} filas (test idx {idx.min()}..{idx.max()})", flush=True)

# rows solo de las filas seleccionadas (take de 2,021 — barato)
tsel = pq.read_table(e7.PARQUET, columns=["x_num", "x_txt_proto",
                                          "x_txt_service", "x_txt_state"]
                     ).take(idx)
proto = tsel.column("x_txt_proto").to_pylist()
service = tsel.column("x_txt_service").to_pylist()
state_c = tsel.column("x_txt_state").to_pylist()
x = tsel.column("x_num").to_pylist()          # 2,021×77 — trivial
rows = [{"x": xi, "proto": str(pr), "service": str(sv), "state": str(st)}
        for xi, pr, sv, st in zip(x, proto, service, state_c)]

meta = json.load(open(e7.META))
questions = e7.build_questions(meta["candidates"], with_unknown=True)
meta_line = json.dumps({"k": -1, "questions": questions})

lines = [meta_line]
for j in range(len(idx)):
    state = e7.build_state(rows[j])
    lines.append(json.dumps({"k": int(idx[j]), "state": state}))

# slices: las filas en orden, reparte i::n
rest = lines[1:]
for i in range(args.n):
    p = out / f"slice_{i:02d}.jsonl"
    with open(p, "w") as fh:
        fh.write(meta_line + "\n")
        for k in range(i, len(rest), args.n):
            fh.write(rest[k] + "\n")
    print(f"  {p.name}: {sum(1 for _ in open(p)) - 1} filas", flush=True)
total = sum(sum(1 for _ in open(out / f"slice_{i:02d}.jsonl")) - 1 for i in range(args.n))
print(f"TOTAL filas: {total} (esperado {len(idx)})", flush=True)
assert total == len(idx)
print("SLICES OK")
