"""E4 — latencia de decisión (§7 del spec): batch-1 en CPU con 1 hilo.

Protocolo: el triage perimetral decide EN LÍNEA (batch 1, flujo a flujo), así
que la métrica es p50/p95/p99 de la ruta COMPLETA de inferencia —
φ (feature stage) + forward del tronco + cabezas + de-tilt/gates — sin
contar la carga del parquet. Se mide en el CPU local (peor caso absoluto, sin
GPU) y, opcionalmente, en Modal L4 para el punto GPU. La primera medición se
descarta (warmup de allocator/JIT).

Uso:
    python -m src.eval.run_e4 --artifact results/artifacts/unsw_small_oficial.pt \
        --n 2000 --out results/e4/
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np


def bench_engine(artifact: Path, n: int) -> dict:
    import torch

    torch.set_num_threads(1)  # peor caso: 1 hilo, batch 1
    from src.models.engine import VXOneEngine

    eng = VXOneEngine(str(artifact))
    d_num = eng.ckpt["config"]["d_num"]
    rng = np.random.default_rng(42)
    # features en la escala que φ produce (post-clip ±8): lo que entra al net
    x = rng.normal(0.0, 1.5, size=(1, d_num)).astype(np.float32)
    # x_txt de predict_batch es texto CRUDO (el motor encodea); None → UNK,
    # misma ruta de cómputo que una fila real
    x_txt = {k: [None] for k in (eng.vocab or {})}

    # warmup (se descarta)
    for _ in range(50):
        eng.predict_batch(x, x_txt)

    t = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        eng.predict_batch(x, x_txt)
        t.append((time.perf_counter_ns() - t0) / 1000.0)  # µs
    t.sort()
    p = lambda q: t[min(len(t) - 1, int(q * len(t)))]
    return {"n": n, "p50_us": round(statistics.median(t), 1),
            "p95_us": round(p(0.95), 1), "p99_us": round(p(0.99), 1),
            "max_us": round(t[-1], 1), "mean_us": round(statistics.mean(t), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/e4")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[e4] {os.cpu_count()} hilos visibles, midiendo con 1 (peor caso) "
          f"batch=1 n={args.n} sobre {args.artifact}")
    r = bench_engine(Path(args.artifact), args.n)
    r["artifact"] = args.artifact
    r["threads"] = 1
    r["batch"] = 1
    r["host"] = "local-cpu"
    (out / f"e4_{Path(args.artifact).stem}.json").write_text(json.dumps(r, indent=1))
    print(f"[e4] p50={r['p50_us']}µs p95={r['p95_us']}µs p99={r['p99_us']}µs "
          f"max={r['max_us']}µs → {out}")


if __name__ == "__main__":
    main()
