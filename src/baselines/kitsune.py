"""Kitsune (Mirsky et al., NDSS 2018) adaptado a flujos — baseline
no-supervisado obligatorio (spec §5.3, lit review: 'baseline conceptual').

Adaptación documentada (los datasets son flujos, no paquetes):
- φ PROPIO: el FeatureStage compartido se fitted sobre TODO train (57%
  ataques en UNSW) y deja al benigno fuera de centro — hostil para un
  modelo benign-only (señal invertida verificada: AUROC(s,y)=0.38). Como
  exige un despliegue no-supervisado real, Kitsune re-fitea φ (misma
  mediana+IQR+log1p) SOLO sobre sus filas de entrenamiento benignas.
- Feature bagging por correlación (algoritmo del paper §III-B): greedy que
  agrupa las features más correlacionadas hasta r por miembro, M = ⌈d/r⌉.
- Autoencoder embudo r → ⌈0.75r⌉ → ⌈0.5625r⌉ → r, ReLU; entrenamiento
  SOLO con filas benignas de train (no-supervisado: nunca ve y_bin=1).
- Score por muestra = RMSE máximo del ensemble (el del original).
- Calibración §5.3: Platt afín p = σ((s−b)/T) fitted en val con Logistic
  Regression sobre s (misma familia de 2 parámetros; w=1/T, b=−w₀/T).
  El npz guarda la salida de Platt (su 'raw'); T_json = 1 (ya calibrado).
- Solo detección (Noul-equivalente): Kitsune no atribuye familias —
  esa es exactamente la diferenciación del paper (sin tipado, sin
  abstención, sin semántica probabilística nativa).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from src.models.data import DATA, _read_x_num, _registry, _frozen_candidates
from src.models.feature_stage import FeatureStage

SEED = 42
N_MEMBERS = 9          # tamaño del ensemble (default del original)
EPOCHS = 30
BATCH = 8192
LR = 1e-3


def correlation_bagging(d: int, m: int, x_benign: torch.Tensor) -> list[list[int]]:
    """Greedy del original: partiendo del par más correlacionado, cada
    grupo crece con la feature de mayor |corr| media con el grupo."""
    sub = x_benign[torch.randperm(len(x_benign))[:100_000]]
    c = np.abs(np.corrcoef(sub.numpy(), rowvar=False))
    np.fill_diagonal(c, 0.0)
    c = np.nan_to_num(c)
    r = int(np.ceil(d / m))
    groups: list[list[int]] = []
    free = set(range(d))
    while free:
        i, j = np.unravel_index(np.argmax(c), c.shape)
        g = [int(i)] if i in free else [int(next(iter(free)))]
        if j in free and j != g[0]:
            g.append(int(j))
        free -= set(g)
        while len(g) < r and free:
            best = max(free, key=lambda f: c[f, g].mean())
            g.append(int(best))
            free.remove(best)
        groups.append(g)
    return groups


class AE(nn.Module):
    def __init__(self, r: int) -> None:
        super().__init__()
        h1, h2 = max(4, int(np.ceil(0.75 * r))), max(2, int(np.ceil(0.5625 * r)))
        self.net = nn.Sequential(nn.Linear(r, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def rmse(net: AE, x: torch.Tensor, bs: int = 65536) -> np.ndarray:
    """RMSE de reconstrucción por muestra (score del miembro)."""
    out = np.empty(len(x), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs]
            e = (xb - net(xb)) ** 2
            out[i:i + bs] = e.mean(1).sqrt().numpy()
    return out


def fit_kitsune(source: str, scheme: str, exclude_family: str | None = None,
                log=print) -> dict:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    path = DATA / ("unsw_nb15/unsw_nb15.parquet" if source == "unsw"
                   else "cicids2017/cicids2017.parquet")
    split_col = "split" if source == "unsw" else f"split_{scheme}"
    tbl = pq.read_table(path, columns=[split_col, "label_family"])
    splt = np.asarray(tbl.column(split_col).to_pylist())
    fam_all = np.asarray(tbl.column("label_family").to_pylist())
    y_all = (fam_all != "normal").astype(np.int64)
    x_all = _read_x_num(path)
    m_tr = splt == "train"
    if exclude_family is not None:
        m_tr &= fam_all != exclude_family
    counts = {f: int((fam_all[m_tr] == f).sum())
              for f in _registry(source)["native_families"]}
    candidates = _frozen_candidates(source, counts)
    rid = f"{source}_{scheme}" + (f"_loao_{exclude_family}" if exclude_family else "")

    # φ de Kitsune: fitted SOLO en las filas benignas de su train (§ arriba)
    x_ben_raw = x_all[m_tr & (y_all == 0)]
    stage = FeatureStage.fit(x_ben_raw.astype(np.float64))
    log(f"[kitsune] {rid}: d={x_all.shape[1]} benign_tr={len(x_ben_raw)} "
        f"(φ re-fitted en benigno)")
    del x_ben_raw

    imap = {c: i for i, c in enumerate(candidates)}
    fam_id = np.array([imap.get(f, -1) for f in fam_all], dtype=np.int64)

    def u(split: str) -> tuple[torch.Tensor, np.ndarray]:
        m = splt == split
        if exclude_family is not None and split != "test":
            m &= fam_all != exclude_family
        uu, _ = stage.transform(x_all[m])
        return torch.from_numpy(uu), y_all[m], fam_id[m]

    x_tr, yb_tr, _ = u("train")
    x_va, y_va, _ = u("val")
    x_te, y_te, yf_te = u("test")
    x_ben = x_tr[yb_tr == 0]
    d = x_tr.shape[1]

    groups = correlation_bagging(d, N_MEMBERS, x_ben)
    nets = []
    for gi, g in enumerate(groups):
        net = AE(len(g))
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        sub = x_ben[:, g]
        n = len(sub)
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            tot = 0.0
            for s in range(0, n, BATCH):
                idx = perm[s:s + BATCH]
                xb = sub[idx]
                loss = ((xb - net(xb)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            if ep in (0, EPOCHS // 2, EPOCHS - 1):
                log(f"[kitsune] member {gi} r={len(g)} epoch {ep + 1} "
                    f"mse={tot / n:.5f}")
        nets.append((net, g))

    def ensemble_score(x: torch.Tensor) -> np.ndarray:
        scores = [rmse(net, x[:, g]) for net, g in nets]
        return np.maximum.reduce(scores)  # el score del original

    s_va = ensemble_score(x_va)
    s_te = ensemble_score(x_te)

    # Platt afín §5.3: LogisticRegression sobre s (w·s+w0 ⇒ T=1/w, b=−w0/w)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(s_va.reshape(-1, 1), y_va)
    w, w0 = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    T, b = 1.0 / w, -w0 / w
    log(f"[kitsune] Platt: T={T:.4f} b={b:.4f} "
        f"(s_va mal={s_va[y_va == 1].mean():.4f} "
        f"ben={s_va[y_va == 0].mean():.4f})")

    p_te = 1.0 / (1.0 + np.exp(-(s_te - b) * w))  # = σ((s−b)/T)
    return {"rid": rid, "p_noul": p_te.astype(np.float32),
            "y_bin": y_te, "y_fam": yf_te, "candidates": candidates,
            "T": 1.0, "b": b, "fit_s": time.time() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["unsw", "cic"])
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--exclude-family", default=None)
    ap.add_argument("--out", default="results/baselines")
    args = ap.parse_args()
    scheme = args.scheme or ("oficial" if args.source == "unsw" else "strat")
    res = fit_kitsune(args.source, scheme, args.exclude_family)
    out = Path(args.out) / res["rid"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "kitsune.npz", p_noul=res["p_noul"],
                        y_bin=res["y_bin"], y_fam=res["y_fam"])
    (out / "kitsune.json").write_text(json.dumps({
        "run_id": res["rid"], "model": "kitsune",
        "candidates": res["candidates"],
        "temperatures": {"noul": 1.0, "choice": 1.0},
        "platt_b": res["b"], "fit_s": round(res["fit_s"], 1)}, indent=2))
    print(f"[kitsune] {res['rid']} → {out}/kitsune.npz "
          f"({res['fit_s']:.0f}s)")


if __name__ == "__main__":
    main()
