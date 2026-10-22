#!/usr/bin/env python3
"""Protocolo conformal mínimo (advisory §2(d)) — EMBER sobre Noul, sin
entrenamiento de modelos nuevos: los logits del checkpoint Small ya están
en npz; el clasificador de dominios es logístico SGD streaming (memoria
plana, float32, nunca float64 completo — colgado de RAM el 2026-10-20).

- Calibración split-conformal en val EMBER (w46–51; subsample n=120k, seed 42):
  score s(x,y) = 1 − p_cal(y|x); Q̂ = ⌈(n+1)(1−α)⌉-ésimo mínimo; α=0.1.
- Por semana de test w∈{52..63}: coverage global y por clase (benign/malicious)
  con IC 95% Clopper–Pearson exacto.
- Variante weighted: clasificador de dominios logístico val-pool vs semana w
  (datasets separados train/eval 2/3 vs 1/3); odds w(x) = P(test|x)/P(val|x)
  en filas de calibración, clipeadas a [0.1,10]; cuantíl ponderado
  (Tibshirani et al. 2019) sobre los mismos scores → coverage weighted.
- Drift: min-BER del clasificador en its held-out half → d_TV ≥ 1−2·BER
  (certificado con estimación held-out, no in-sample).

Resultados en results/e4_conformal/conformal_weeks.{json,md}.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# El pool nativo (jemalloc) retiene VA tras liberar → con RLIMIT_AS es
# fatal. Pool de sistema: cada buffer se munmap-ea al liberarse.
pa.set_memory_pool(pa.system_memory_pool())
print("[arrow] pool =", pa.default_memory_pool().backend_name, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.data import ember_shards

OLD = Path("${HOME}/projects/paper/results/outputs")
OUT = Path("results/e4_conformal")
OUT.mkdir(parents=True, exist_ok=True)

NPZ = "ember_small_temporal"
ALPHA = 0.1
N_CAL = 120_000
N_CLF_VAL = 60_000        # pool val para el dominio-clf (2/3 train, 1/3 eval)
W0_TEST = 52
N_WEEKS = 12
LIST_SIZE = 2568
EPOCHS = 20
BATCH = 8192


def cp_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """IC exacto de Clopper–Pearson para k/n."""
    if k == 0:
        return (0.0, stats.beta.ppf(1 - (1 - conf) / 2, 1, n))
    if k == n:
        return (stats.beta.ppf((1 - conf) / 2, n, 1), 1.0)
    return (stats.beta.ppf((1 - conf) / 2, k, n - k + 1),
            stats.beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))


def read_x(path: Path, local_idx: np.ndarray | None = None) -> np.ndarray:
    """x_num crudo [N, 2568] float32 (mismo orden que el loader del repo).
    Con local_idx (ordenado asc) devuelve solo esas filas."""
    x = pq.read_table(path, columns=["x_num"]).column("x_num")
    flat = np.asarray(x.combine_chunks().flatten().to_numpy(zero_copy_only=False),
                      dtype=np.float32).reshape(-1, LIST_SIZE)
    return flat if local_idx is None else flat[local_idx]


def take_val_rows(idx: np.ndarray) -> np.ndarray:
    """Filas de val por índices globales (0..359999), leyendo 1 shard a la
    vez. Requiere idx ordenado Asc (valida)."""
    assert (np.diff(idx) > 0).all()
    shards = ember_shards("val")
    out = np.empty((len(idx), LIST_SIZE), dtype=np.float32)
    gi = 0
    for sh, path in enumerate(shards):
        lo, hi = sh * 60000, (sh + 1) * 60000
        j = np.searchsorted(idx, [lo, hi - 1e-6])
        sel = idx[j[0]:j[1]] - lo
        if len(sel):
            out[gi:gi + len(sel)] = read_x(path, sel)
            gi += len(sel)
    assert gi == len(idx)
    return out


def pooled_meanstd(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """media/std combinada de dos bloques f32, sin concatenarlos."""
    n = len(a) + len(b)
    m = (a.sum(0) + b.sum(0)) / n
    v = ((a - m) ** 2).sum(0) + ((b - m) ** 2).sum(0)
    sd = np.sqrt(v / n + 1e-12)
    return m, sd


def fit_domain_logistic(Xa: np.ndarray, Xb: np.ndarray
                        ) -> tuple[dict, SGDClassifier, np.ndarray, np.ndarray]:
    """Clasificador dominios logístico (SGD log-loss) entre bloque A (val=0)
    y bloque B (semana=1), 2/3 train 1/3 eval held-out. Estandariza
    escribiendo en bloques preallocados (sin copias extra). Devuelve métr
    held-out, el clf (fitted TRAIN 2/3) y (mu, sd)."""
    n_tr = (2 * len(Xa)) // 3
    rng = np.random.default_rng(11)
    pa = rng.permutation(len(Xa))
    pb = rng.permutation(len(Xb))
    tr_a, ev_a = pa[:n_tr], pa[n_tr:]
    tr_b, ev_b = pb[:n_tr], pb[n_tr:]
    n_ev = len(Xa) + len(Xb) - 2 * n_tr
    Xtr = np.empty((2 * n_tr, LIST_SIZE), dtype=np.float32)
    Xev = np.empty((n_ev, LIST_SIZE), dtype=np.float32)
    ytr = np.concatenate([np.zeros(n_tr, np.int8), np.ones(n_tr, np.int8)])
    yev = np.concatenate([np.zeros(len(Xa) - n_tr, np.int8),
                          np.ones(len(Xb) - n_tr, np.int8)])
    mu, sd = pooled_meanstd(Xa[tr_a], Xb[tr_b])
    sd[sd < 1e-6] = 1.0
    off = 0
    for D, j in ((Xtr, tr_a), ):
        dst = D[off:off + len(j)]
        np.subtract(Xa[j], mu, out=dst)
        np.divide(dst, sd, out=dst)
        off += len(j)
    dst = Xtr[off:off + len(tr_b)]
    np.subtract(Xb[tr_b], mu, out=dst)
    np.divide(dst, sd, out=dst)
    off = 0
    for D, j in ((Xev, ev_a), ):
        dst = D[off:off + len(j)]
        np.subtract(Xa[j], mu, out=dst)
        np.divide(dst, sd, out=dst)
        off += len(j)
    dst = Xev[off:off + len(ev_b)]
    np.subtract(Xb[ev_b], mu, out=dst)
    np.divide(dst, sd, out=dst)
    Xtr = np.ascontiguousarray(Xtr)
    clf = SGDClassifier(loss="log_loss", alpha=1e-5,
                        learning_rate="optimal", random_state=7)
    o = np.random.default_rng(7).permutation(len(ytr))
    if o[0] >= n_tr:                       # primera clase debe incluir ambas
        for k in range(len(o)):
            if o[k] < n_tr:
                o[[0, k]] = o[[k, 0]]
                break
    clf.partial_fit(Xtr[o[:BATCH]], ytr[o[:BATCH]], classes=np.arange(2))
    r2 = np.random.default_rng(13)
    for _ in range(EPOCHS):
        o = r2.permutation(len(ytr))
        for st in range(0, len(ytr), BATCH):
            sl = o[st:st + BATCH]
            clf.partial_fit(Xtr[sl], ytr[sl])
    # métr held-out
    ps = np.empty(len(Xev), dtype=np.float32)
    for st in range(0, len(Xev), 32768):
        ps[st:st + 32768] = clf.predict_proba(Xev[st:st + 32768])[:, 1]
    fpr, tpr, _ = roc_curve(yev, ps)
    ber = float(np.min(0.5 * (fpr + (1.0 - tpr))))
    return {"ber": ber,
            "auroc": float(roc_auc_score(yev, ps)),
            "n_train": int(len(ytr)), "n_eval": int(len(yev))}, \
        clf, mu, sd


def score_cif(clf: SGDClassifier, mu: np.ndarray, sd: np.ndarray,
              X: np.ndarray) -> np.ndarray:
    """P(bloque B | x) por trozos, en f32."""
    out = np.empty(len(X), dtype=np.float32)
    for st in range(0, len(X), 16384):
        Z = (X[st:st + 16384] - mu) / sd
        out[st:st + 16384] = clf.predict_proba(Z)[:, 1]
    return out


def weighted_quantile(scores: np.ndarray, w: np.ndarray, level: float) -> float:
    """Q̂ ponderado (test con peso 1; Tibshirani et al. 2019): menor s_(k)
    tal que (Σ_{i≤k} w_i + 1) / (1 + Σ w_i) ≥ level."""
    order = np.argsort(scores)
    s, ww = scores[order], w[order]
    cw = np.cumsum(ww)
    total = 1.0 + ww.sum()
    k = int(np.searchsorted(cw, level * total - 1.0, side="left"))
    return float(s[min(k, len(s) - 1)])


def main() -> None:
    t0 = time.time()
    npz = dict(np.load(OLD / f"{NPZ}.npz", allow_pickle=True))
    meta = json.loads((OLD / f"{NPZ}.json").read_text())
    T = float(meta["temperatures"]["noul"])

    p_va = 1.0 / (1.0 + np.exp(-npz["va_noul_logits"].astype(np.float64) / T))
    p_te = 1.0 / (1.0 + np.exp(-npz["te_noul_logits"].astype(np.float64) / T))
    y_va = npz["va_y_bin"].astype(int)
    y_te = npz["te_y_bin"].astype(int)

    # subsets de val: calibración y pool del dominio-clf DISJUNTOS
    rng = np.random.default_rng(42)
    cal_idx = np.sort(rng.choice(len(y_va), N_CAL, replace=False))
    mask = np.ones(len(y_va), dtype=bool)
    mask[cal_idx] = False
    rng2 = np.random.default_rng(43)
    pool_cands = np.where(mask)[0]
    clf_val_idx = np.sort(rng2.choice(pool_cands, N_CLF_VAL, replace=False))

    print("[ember] extrayendo features de val (2 pasadas por shard)...",
          flush=True)
    X_cal = take_val_rows(cal_idx)
    X_pool = take_val_rows(clf_val_idx)
    p_cal = p_va[cal_idx]
    y_cal = y_va[cal_idx]
    s_cal = np.where(y_cal == 1, 1.0 - p_cal, p_cal)          # s(x, y_true)
    q_hat = float(np.sort(s_cal)[
        int(np.ceil((N_CAL + 1) * (1 - ALPHA))) - 1])
    cov_cal = np.where(y_cal == 1, p_cal >= 1.0 - q_hat, p_cal <= q_hat)
    del p_va, y_va, npz

    def covered_at(q: float, ndx: np.ndarray) -> np.ndarray:
        p, y = p_te[ndx], y_te[ndx]
        return np.where(y == 1, p >= 1.0 - q, p <= q)

    weeks_te = np.concatenate([
        np.full(60000, w, dtype=np.int64)
        for w in range(W0_TEST, W0_TEST + N_WEEKS)
    ])

    rows = []
    te_shards = ember_shards("test")
    for wi, w in enumerate(range(W0_TEST, W0_TEST + N_WEEKS)):
        ndx = np.where(weeks_te == w)[0]
        cov_u = covered_at(q_hat, ndx)
        row: dict = {
            "week": int(w), "n": int(len(ndx)),
            "coverage_unw": float(cov_u.mean()),
            "cp95": cp_interval(int(cov_u.sum()), int(len(ndx))),
        }
        for c, name in ((1, "malicious"), (0, "benign")):
            m = y_te[ndx] == c
            ck = cov_u[m]
            row[f"coverage_unw_{name}"] = float(ck.mean())
            row[f"cp95_{name}"] = cp_interval(int(ck.sum()), int(m.sum()))
        print(f"[ember] w{w} cov_unw={row['coverage_unw']:.4f} "
              f"(mal {row['coverage_unw_malicious']:.4f}, "
              f"ben {row['coverage_unw_benign']:.4f})", flush=True)

        print(f"[ember] w{w}: domain classifier...", flush=True)
        X_w = read_x(te_shards[wi])
        domm, clf, mu, sd = fit_domain_logistic(X_pool, X_w)
        row.update(domm, dtv_lb=1.0 - 2.0 * domm["ber"])
        print(f"[ember] w{w}: BER={domm['ber']:.4f} "
              f"AUC={domm['auroc']:.4f} dTV_lb={row['dtv_lb']:.4f}",
              flush=True)
        # odds en filas de calibración: P(test|x)/P(val|x) = p/(1−p)
        g_cal = score_cif(clf, mu, sd, X_cal).astype(np.float64)
        wgt = np.clip(g_cal / np.clip(1.0 - g_cal, 1e-12, None), 0.1, 10.0)
        q_w = weighted_quantile(s_cal, wgt, 1 - ALPHA)
        cov_wt = covered_at(q_w, ndx)
        row.update({
            "q_weighted": q_w, "q_unweighted": q_hat,
            "coverage_wt": float(cov_wt.mean()),
            "coverage_wt_malicious": float(cov_wt[y_te[ndx] == 1].mean()),
            "coverage_wt_benign": float(cov_wt[y_te[ndx] == 0].mean()),
        })
        print(f"[ember] w{w}: q_wt={q_w:.4f} cov_wt={row['coverage_wt']:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        rows.append(row)
        del clf, mu, sd
        gc.collect()
        json.dump(
            {"config": {"head": f"Noul (cal T={T:.4f})", "alpha": ALPHA,
                        "n_cal": N_CAL, "score": "1 - p_cal(y|x)",
                        "cal_weeks": "46-51 (subsample, seed 42)",
                        "q_unweighted": q_hat,
                        "weights": "logistic domain clf val-pool vs week, "
                                   "odds P(test|x)/P(val|x) clip [0.1,10]",
                        "ber": "held-out (2/3 train, 1/3 eval)"},
             "calibration_coverage": float(cov_cal.mean()),
             "weeks": rows},
            open(OUT / "conformal_weeks.json", "w"), indent=2)

    md = ["# Conformal sobre Noul, EMBER (advisory §2(d))\n",
          f"α={ALPHA}; calibración split-conformal en w46–51 (n={N_CAL:,}, "
          f"seed 42); Q̂={q_hat:.4f}; **coverage en calibración "
          f"{float(cov_cal.mean()):.4f}** (referencia de intercambiabilidad "
          "nombrada).\n",
          "| sem | cov unp. (IC95) | cov unp. mal | cov unp. ben | d_TV ≥ 1−2BER "
          "| BER held-out | cov wt | cov wt mal | cov wt ben |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lo, hi = r["cp95"]
        md.append(
            f"| {r['week']} | {r['coverage_unw']:.4f} [{lo:.4f},{hi:.4f}] "
            f"| {r['coverage_unw_malicious']:.4f} "
            f"| {r['coverage_unw_benign']:.4f} "
            f"| {r['dtv_lb']:.4f} | {r['ber']:.4f} "
            f"| {r['coverage_wt']:.4f} "
            f"| {r['coverage_wt_malicious']:.4f} "
            f"| {r['coverage_wt_benign']:.4f} |")
    md.append(
        "\nLectura (scoped del advisory): coverage EMPÍRICO con IC "
        "Clopper–Pearson por semana — NO una garantía distribution-free. "
        "La cota de drift certifica magnitud (d_TV ≥ 1−2·BER, BER medido "
        "held-out); el anclaje citable es Barber et al. 2023 (gap ≈ 2·d_TV "
        "con pesos conocidos) y la variante weighted es mitigación empírica, "
        "nunca garantía exacta. Challenge set: fuera de esta tabla (estrés "
        "sin benignos, se reporta separado).")
    open(OUT / "conformal_weeks.md", "w").write("\n".join(md) + "\n")
    print(f"OK -> {OUT} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
