"""E4 — decision latency (§7 of the spec): batch-1 on CPU with 1 thread.

Protocol: perimeter triage decides ONLINE (batch 1, flow by flow), thus
that the metric is p50/p95/p99 of the FULL inference path —
φ (feature stage) + trunk forward + heads + de-tilt/gates — without
count the load of the parquet. It is measured in the local CPU (absolute worst case, without
GPU) and optionally in Modal L4 for the GPU point. The first measurement is
discard (allocator/JIT warmup).

Use:
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
    # features on the scale that φ produces (post-clip ±8): what enters the net
    x = rng.normal(0.0, 1.5, size=(1, d_num)).astype(np.float32)
    # x_txt of predict_batch is RAW text (engine encodes); None → UNK,
    # same compute path as a real row
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
