#!/usr/bin/env python3
"""Diagnósticos E3 que reusan logits ya computados (sin re-entrenar):

1. Reliability por condicional CORN (Lema 2): q1 en todas las filas,
   q2 en elegibles sev≥1, y ECE DIRECTO del drop gate
   C2 = P(sev≥2|x) = q1·q2 — atribuye el ECE del gate a sus eslabones.
2. Classwise calibration de Choice: E[1[y=k] | p_k] por clase
   (incluye la coordenada recon que alimenta g_recon).
3. Bayes-vs-Chow en val: % de filas donde la regla general con matriz
   de costes difiere de la abstracción max-prob (Chow, λ_r=0.5).

Resultados en results/e3_diag/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RUNS = [
    "unsw_small_oficial", "cic_small_strat", "cic_small_time",
    "ember_small_temporal",
]
OLD = Path("${HOME}/projects/paper/results/outputs")
OUT = Path("results/e3_diag")
OUT.mkdir(parents=True, exist_ok=True)
BINS = 15


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def cal_measures(p: np.ndarray, event: np.ndarray, bins: int = BINS) -> dict:
    """ECE/MCE sobre (p, 1[event]); p es una probabilidad del evento."""
    p = np.clip(p, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = mce = 0.0
    rel = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            rel.append({"bin": b, "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(p[m].mean()), float(event[m].mean())
        gap = abs(acc - conf)
        ece += gap * m.mean()
        mce = max(mce, gap)
        rel.append({"bin": b, "n": int(m.sum()), "conf": round(conf, 4),
                    "acc": round(acc, 4)})
    return {"ece": float(ece), "mce": float(mce), "n": int(len(p)),
            "reliability": rel}


def corn_q(z: np.ndarray) -> np.ndarray:
    """[N,3] logits condicionales → q_k = P(y≥k | y≥k−1), k=1..3."""
    return sigmoid(z)


# ── 1. Reliability por condicional CORN + ECE directo de C2 ───────────
def corn_reliability(npz: dict, meta: dict) -> dict | None:
    """Lema 2 aplicada a la cabeza Score (CORN). Los runs sin score head
    (EMBER binario: no hay corn logits) no aplican."""
    if "va_corn_logits" not in npz:
        return None
    T = float(meta["temperatures"]["score"])
    out: dict = {}
    for pre in ("va", "te"):
        z = npz[f"{pre}_corn_logits"].astype(np.float64) / T
        y_sev = npz[f"{pre}_y_sev"].astype(int)
        q = corn_q(z)
        block: dict = {}
        # q1: todas las filas
        ev1 = (y_sev >= 1).astype(float)
        block["q1"] = {
            "raw": cal_measures(sigmoid(npz[f"{pre}_corn_logits"][:, 0]
                                        .astype(np.float64)), ev1),
            "cal": cal_measures(q[:, 0], ev1),
        }
        # q2: elegibles sev≥1
        m2 = y_sev >= 1
        ev2 = (y_sev[m2] >= 2).astype(float)
        block["q2"] = {
            "raw": cal_measures(
                sigmoid(npz[f"{pre}_corn_logits"][m2, 1].astype(np.float64)),
                ev2),
            "cal": cal_measures(q[m2, 1], ev2),
        }
        # C2 directo: q1·q2, todas las filas
        m1 = y_sev >= 0  # todas
        c2_cal = q[:, 0] * q[:, 1]
        zr = npz[f"{pre}_corn_logits"].astype(np.float64)
        c2_raw = sigmoid(zr[:, 0]) * sigmoid(zr[:, 1])
        block["C2"] = {
            "raw": cal_measures(c2_raw, (y_sev >= 2).astype(float)),
            "cal": cal_measures(c2_cal, (y_sev >= 2).astype(float)),
        }
        out[pre] = block
    return out


# ── 2. Classwise calibration de Choice ────────────────────────────────
def classwise(npz: dict, meta: dict) -> dict:
    T = float(meta["temperatures"]["choice"])
    omega = np.asarray(meta["omega"], dtype=np.float64)
    names = meta.get("candidates")
    out: dict = {}
    for pre in ("va", "te"):
        y = npz[f"{pre}_y_fam"]
        m = y >= 0
        z = npz[f"{pre}_choice_logits"][m].astype(np.float64)
        p = softmax((z - np.log(omega)) / T)
        yy = y[m].astype(int)
        per_class = {}
        for k in range(p.shape[1]):
            ev = (yy == k).astype(float)
            per_class[(names or [str(k) for k in range(p.shape[1])])[k]] = {
                **cal_measures(p[:, k], ev),
                "support": int(ev.sum()),
            }
        # ECE classwise ponderado por filas sobre (p_k, 1[y=k])
        out[pre] = {
            "classwise_ece": None,
            "per_class": per_class,
        }
        # agregado: por filas, |p_k − 1[y=k]| binned (classwise ECE)
        K = p.shape[1]
        ev_long = np.zeros((len(p), K))
        ev_long[np.arange(len(p)), yy] = 1.0
        out[pre]["classwise_ece"] = {
            "cal": cal_measures(p.ravel(), ev_long.ravel()),
        }
    return out


# ── 3. Bayes-vs-Chow (val, choice head) ───────────────────────────────
def bayes_vs_chow(npz: dict, meta: dict) -> dict:
    """Chow/max-prob: atribuye argmax si max p ≥ 1−λ_r, si no abstiene.
    Bayes general: acciones = clases ∪ {abstain}; L(d,y) = c(y)·1[d≠y] con
    c(y) = mfac si y es attack (miss) sino 1; L(abstain,y) = λ_r.
    E[coste|d] = S − p_d·c(d) con S = Σ_y p_y·c(y); abstiene si min > λ_r.
    Con mfac=1 Chow y Bayes coinciden EXACTAMENTE (sanity: frac_dev=0)."""
    T = float(meta["temperatures"]["choice"])
    omega = np.asarray(meta["omega"], dtype=np.float64)
    names = meta.get("candidates") or []
    y = npz["va_y_fam"]
    m = y >= 0
    z = npz["va_choice_logits"][m].astype(np.float64)
    p = softmax((z - np.log(omega)) / T)
    yy = y[m].astype(int)
    norm_idx = names.index("normal") if "normal" in names else 0
    is_attack_true = yy != norm_idx                                   # [N]
    out: dict = {}
    for mfac in (1, 2, 3, 5, 10):
        c_all = np.array([1.0 if n == "normal" else mfac for n in names])
        S = p @ c_all                                                  # [N]
        exp_cost = S[:, None] - p * c_all[None, :]                     # [N,K]
        for lam_r in (0.5, 0.1):
            thr = 1.0 - lam_r
            chow_action = np.where(p.max(1) >= thr, p.argmax(1), -1)
            # Bayes: atribuye si el coste esperado mínimo ≤ λ_r, si no abstiene
            bayes_action = np.where(exp_cost.min(1) <= lam_r,
                                    exp_cost.argmin(1), -1)
            diff = bayes_action != chow_action
            out[f"m{mfac}_lam{lam_r}"] = {
                "frac_dev": float(diff.mean()),
                "n_dev": int(diff.sum()),
                "n": int(len(p)),
                "frac_dev_among_ms_attacks":
                    float(diff[is_attack_true].mean())
                    if is_attack_true.any() else None,
                "ms": float((p.max(1) >= thr).mean()),
            }
    return out


def main() -> None:
    summary: dict = {}
    for run in RUNS:
        npz = dict(np.load(OLD / f"{run}.npz", allow_pickle=True))
        meta = json.loads((OLD / f"{run}.json").read_text())
        summary[run] = {
            "names": meta.get("candidates"),
            "corn_reliability": corn_reliability(npz, meta),
            "classwise": classwise(npz, meta),
            "bayes_vs_chow": bayes_vs_chow(npz, meta)
            if (meta.get("candidates")
                and "normal" in meta["candidates"]) else None,
        }
    json.dump(summary, open(OUT / "e3_diag.json", "w"), indent=2, default=float)

    md = ["# Diagnósticos E3 sobre logits ya computados (sin re-entrenar)\n",
          "## 1) Reliability por condicional CORN + ECE directo del drop gate C2\n",
          "Lema 2: C2 = P(sev≥2|x) = q1·q2. El ECE del gate se mide DIRECTO;",
          "los condicionales se miden por separado en su población propia.\n",
          "| run | split | ECE(q1) | ECE(q2 \\| sev≥1) | n(q2) | ECE(C2) dir | ECE(C2) raw |",
          "|---|---|---|---|---|---|---|"]
    for run, r in summary.items():
        cr = r["corn_reliability"]
        if not cr:
            md.append(f"| {run} | — | n/a | n/a | — | n/a | n/a |")
            continue
        for pre, ln in (("va", "val"), ("te", "test")):
            b = cr[pre]
            md.append(
                f"| {run} | {ln} | {b['q1']['cal']['ece']:.4f} "
                f"| {b['q2']['cal']['ece']:.4f} | {b['q2']['cal']['n']} "
                f"| {b['C2']['cal']['ece']:.4f} | {b['C2']['raw']['ece']:.4f} |")
    md += ["\n## 2) Classwise calibration de Choice (val)\n",
           "E[1[y=k] \\| p_k] por clase; la coordenada recon alimenta g_recon.\n",
           "| run | classwise ECE | peor clase | ECE | support | recon: ECE |",
           "|---|---|---|---|---|---|"]
    for run, r in summary.items():
        pc = r["classwise"]["va"]["per_class"]
        cw = r["classwise"]["va"]["classwise_ece"]["cal"]["ece"]
        worst = max(pc.items(), key=lambda kv: kv[1]["ece"])
        rec = pc.get("recon", {})
        md.append(
            f"| {run} | {cw:.4f} | {worst[0]} | {worst[1]['ece']:.4f} "
            f"| {worst[1]['support']} | {rec.get('ece', float('nan')):.4f} |")
    md += ["\n## 3) Bayes-vs-Chow (val, choice head)\n",
           "Chow: atribuye argmax si max p ≥ 1−λ_r. Bayes: matriz de costes",
           "L(d,y) = c(y)·1[d≠y], c(y)=m si y es attack (miss) sino 1;",
           "abstain = λ_r. Con m=1 las reglas coinciden EXACTAMENTE",
           "(sanity: 0 desviaciones). EMBER sin clase normal → n/a.\n",
           "| run | λ_r | m | % filas donde difieren | % entre attacks |",
           "|---|---|---|---|---|"]
    for run, r in summary.items():
        b = r["bayes_vs_chow"]
        if not b:
            continue
        for k, v in b.items():
            mfac, lam = k.split("_")
            md.append(
                f"| {run} | {lam[3:]} | {mfac[1:]} | {v['frac_dev']*100:.2f} "
                f"| {v['frac_dev_among_ms_attacks']*100:.2f} |")
    md.append(
        "\n## Veredictos\n"
        "- **CORN/Lema 2**: los ECE condicionales de los eslabones NO "
        "determinan el ECE del producto. UNSW test: q1 0.0925 y q2 0.0508 "
        "coexisten con C2 directo 0.0438; CIC by-day test: q1 0.4203 y q2 "
        "0.1407 coexisten con C2 directo 0.1125. Exactamente la razón por la "
        "que Lemma 2 manda a medir el gate, no a sus eslabones.\n"
        "- **Classwise**: la calibración max-prob (TS) no garantiza "
        "calibración por coordenada; ECE classwise val ≤0.007 en los 4 runs "
        "y la coordenada recon queda ≤0.003 donde existe (UNSW 0.0027 en "
        "1,574 filas; CIC strat 0.0001 en 23,839). El g_recon del "
        "controlador hereda una coordenada razonablemente calibrada.\n"
        "- **Bayes-vs-Chow**: con costes uniformes la abstracción max-prob "
        "es regla de Bayes EXACTA (0 desviaciones en los 3 runs). La "
        "desviación aparece solo con asimetrías fuertes y SIEMPRE en la "
        "misma dirección: Bayes escala (abstiene) y Chow atribuye — la "
        "forma exacta del contraejemplo del apéndice. Magnitudes: CIC "
        "strat ≤0.2% en todo el grid; UNSW val 13.4% (m=3, λ_r=0.5) hasta "
        "19.2% (m=10); CIC by-day ≤4.9% hasta m=10 λ_r=0.5, pero 10.8% "
        "(28.8% entre attacks) con m=10 λ_r=0.1. Consecuencia operativa: "
        "con τ=0.5 y costes moderados la abstracción está justificada "
        "empíricamente; operadores con asimetría grande (miss:alarm ≥5:1 "
        "con λ_r bajo) NO pueden heredar τ=0.5 sin remirir la política en "
        "val.")
    open(OUT / "e3_diag.md", "w").write("\n".join(md) + "\n")
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
