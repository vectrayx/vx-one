"""Métricas E1/E3 computadas local desde npz (sin re-entrenar).

E1 (spec F0/F2): AUROC, FPR@95TPR (micro + macro one-vs-rest), PR-AUC,
macro-F1 Choice sin abstención, cobertura/accuracy con abstención τ_abs,
MAE score (1−MAE/3), macro-F1 ddos.
E3 (§5.3): ECE 15 bins, MCE, Brier, NLL — RAW y CALIBRADO — para Noul y
Choice (max-prob binario/multiclase), + datos de reliability diagram.

VX-One: npz de dump_outputs (logits crudos) + json meta (T, ω, τ) →
calibradas recomputadas con el MISMO contrato del engine (de-tilt→T).
Baselines: npz de run_tabular (probs crudas) + json (T) → calibrated =
f(raw, T); de-tilt choice con la ω de la corrida VX correspondiente
(mismos pesos de entrenamiento, Prop. 2 simétrica).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.losses import corn_pmf

ECE_BINS = 15


# ── utilidades base ────────────────────────────────────────────────────
def fpr_at_tpr(scores: np.ndarray, y: np.ndarray, tpr_target: float = 0.95) -> float:
    """FPR al menor umbral con TPR ≥ target (estándar en detección)."""
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresh = np.quantile(pos, 1 - tpr_target)
    return float((neg >= thresh).mean())


def _binary_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return {
        "auroc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "fpr@95tpr": fpr_at_tpr(p, y),
    }


def _calibration_measures(p_conf: np.ndarray, correct: np.ndarray,
                          bins: int = ECE_BINS) -> dict:
    """ECE/MCE + datos del reliability diagram sobre (conf, correct)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p_conf, edges[1:-1]), 0, bins - 1)
    ece = mce = 0.0
    rel = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            rel.append({"bin": b, "n": 0, "conf": None, "acc": None})
            continue
        conf, acc = float(p_conf[m].mean()), float(correct[m].mean())
        gap = abs(acc - conf)
        ece += gap * m.mean()
        mce = max(mce, gap)
        rel.append({"bin": b, "n": int(m.sum()), "conf": conf, "acc": acc})
    return {"ece": float(ece), "mce": float(mce), "reliability": rel}


def noul_block(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray) -> dict:
    """E1+E3 para Noul (detección binaria). Para ECE/MCE la CONFIANZA es la
    de la clase predicha: conf = max(p, 1−p) (p directo confundiría un
    benigno con p≈0 correcto como gap ~1)."""
    p_conf_raw = np.maximum(p_raw, 1.0 - p_raw)
    p_conf_cal = np.maximum(p_cal, 1.0 - p_cal)
    correct_raw = (p_raw >= 0.5) == (y == 1)
    correct_cal = (p_cal >= 0.5) == (y == 1)
    nll_raw = float(-(y * np.log(np.clip(p_raw, 1e-12, 1))
                      + (1 - y) * np.log(np.clip(1 - p_raw, 1e-12, 1))).mean())
    nll_cal = float(-(y * np.log(np.clip(p_cal, 1e-12, 1))
                      + (1 - y) * np.log(np.clip(1 - p_cal, 1e-12, 1))).mean())
    return {
        "e1": {**_binary_metrics(p_cal, y),
               "auroc_raw": _binary_metrics(p_raw, y)["auroc"]},
        "e3": {
            "raw": {**_calibration_measures(p_conf_raw, correct_raw),
                    "brier": float(((p_raw - y) ** 2).mean()), "nll": nll_raw},
            "cal": {**_calibration_measures(p_conf_cal, correct_cal),
                    "brier": float(((p_cal - y) ** 2).mean()), "nll": nll_cal},
        },
    }


def _mc_conf_correct(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    return conf, correct


def _mc_nll_brier(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    onehot = np.eye(p.shape[1])[y]
    brier = float(((p - onehot) ** 2).sum(1).mean())
    return nll, brier


def choice_block(p_raw: np.ndarray, p_cal: np.ndarray, y: np.ndarray,
                 tau_abs: float = 0.5,
                 candidates: list[str] | None = None) -> dict:
    """E1+E3 para Choice (multiclase, sin abstención + política τ_abs)."""
    K = p_cal.shape[1]
    from sklearn.metrics import f1_score

    pred_cal = p_cal.argmax(1)
    f1s = f1_score(y, pred_cal, labels=list(range(K)), average=None,
                   zero_division=0)
    names = candidates or [str(i) for i in range(K)]
    per_class_f1 = {names[k]: round(float(v), 4) for k, v in enumerate(f1s)}
    conf_raw, corr_raw = _mc_conf_correct(p_raw, y)
    conf_cal, corr_cal = _mc_conf_correct(p_cal, y)
    nll_raw, brier_raw = _mc_nll_brier(p_raw, y)
    nll_cal, brier_cal = _mc_nll_brier(p_cal, y)
    out = {
        "e1": {
            "macro_f1": float(f1_score(y, pred_cal, average="macro",
                                       zero_division=0)),
            "per_class_f1": per_class_f1,
            "abstention": {
                "tau_abs": tau_abs,
                "coverage": float((conf_cal >= tau_abs).mean()),
                "accuracy_on_covered": float(
                    corr_cal[conf_cal >= tau_abs].mean())
                if (conf_cal >= tau_abs).any() else float("nan"),
                "accuracy_when_unknown": float("nan")
                if (conf_cal < tau_abs).sum() == 0 else float("nan"),
            },
        },
        "e3": {
            "raw": {**_calibration_measures(conf_raw, corr_raw),
                    "brier": brier_raw, "nll": nll_raw},
            "cal": {**_calibration_measures(conf_cal, corr_cal),
                    "brier": brier_cal, "nll": nll_cal},
        },
    }
    return out


def score_block(pi_cal: np.ndarray, y_sev: np.ndarray) -> dict:
    """E1 score: ŝ = Σ k·π_k, MAE, 1−MAE/3."""
    sev_hat = (pi_cal * np.arange(pi_cal.shape[1])).sum(1)
    mae = float(np.abs(sev_hat - y_sev).mean())
    return {"mae": mae, "score_m": 1.0 - mae / 3.0}


# ── reconstrucción de probs calibradas desde npz+json ──────────────────
def vx_calibrated(npz: dict, meta: dict) -> dict:
    """Aplica el contrato del engine a los logits crudos: Noul σ(z/T);
    Choice de-tilt z−log ω → softmax(z/T); Score CORN recomuesto; ddos igual
    que choice. Devuelve probs crudas y calibradas."""
    T = meta["temperatures"]
    omega = np.asarray(meta["omega"], dtype=np.float64)
    out = {}
    for prefix in ("te", "va"):
        if f"{prefix}_noul_logits" not in npz:
            continue
        z_n = npz[f"{prefix}_noul_logits"].astype(np.float64)
        z_c = (npz[f"{prefix}_choice_logits"].astype(np.float64)
               - np.log(omega))
        out[f"{prefix}_p_noul_raw"] = 1.0 / (1.0 + np.exp(-z_n))
        out[f"{prefix}_p_noul"] = 1.0 / (1.0 + np.exp(-z_n / T["noul"]))
        out[f"{prefix}_p_choice_raw"] = _softmax(
            npz[f"{prefix}_choice_logits"].astype(np.float64))
        out[f"{prefix}_p_choice"] = _softmax(z_c / T["choice"])
        if f"{prefix}_corn_logits" in npz:
            out[f"{prefix}_pi_raw"] = corn_pmf(torch_tensor(
                npz[f"{prefix}_corn_logits"].astype(np.float64)))[1].numpy()
            out[f"{prefix}_pi"] = corn_pmf(torch_tensor(
                npz[f"{prefix}_corn_logits"].astype(np.float64)
                / T["score"]))[1].numpy()
        if f"{prefix}_ddos_logits" in npz:
            om_d = np.asarray(meta["omega_ddos"], dtype=np.float64)
            z_d = (npz[f"{prefix}_ddos_logits"].astype(np.float64)
                   - np.log(om_d))
            out[f"{prefix}_p_ddos_raw"] = _softmax(
                npz[f"{prefix}_ddos_logits"].astype(np.float64))
            out[f"{prefix}_p_ddos"] = _softmax(z_d / T["ddos"])
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def torch_tensor(a: np.ndarray):
    import torch
    return torch.tensor(a)


# ── evaluador de una corrida VX ────────────────────────────────────────
def eval_vx_run(npz_path: str | Path) -> dict:
    npz = dict(np.load(npz_path, allow_pickle=True))
    meta = json.loads(Path(str(npz_path).replace(".npz", ".json")).read_text())
    cal = vx_calibrated(npz, meta)
    tau = meta.get("tau_abs", 0.5)
    res: dict = {"run_id": meta.get("artifact", str(npz_path)),
                 "scale": meta.get("scale"), "source": meta.get("source"),
                 "split_scheme": meta.get("split_scheme"),
                 "exclude": meta.get("exclude")}
    for prefix in ("te", "va"):
        if f"{prefix}_p_noul" not in cal:
            continue
        y_bin = npz[f"{prefix}_y_bin"]
        blk = {"noul": noul_block(cal[f"{prefix}_p_noul_raw"],
                                  cal[f"{prefix}_p_noul"], y_bin)}
        y_fam = npz[f"{prefix}_y_fam"]
        fam_mask = y_fam >= 0
        if fam_mask.any():
            blk["choice"] = choice_block(
                cal[f"{prefix}_p_choice_raw"][fam_mask],
                cal[f"{prefix}_p_choice"][fam_mask], y_fam[fam_mask], tau,
                candidates=meta.get("candidates"))
        else:
            blk["choice"] = None
        if f"{prefix}_pi" in cal:
            blk["score"] = score_block(cal[f"{prefix}_pi"], npz[f"{prefix}_y_sev"])
        if f"{prefix}_p_ddos" in cal:
            m = npz[f"{prefix}_ddos_mask"].astype(bool)
            from sklearn.metrics import f1_score

            blk["ddos"] = {
                "macro_f1": float(f1_score(
                    npz[f"{prefix}_y_ddos"][m], cal[f"{prefix}_p_ddos"][m].argmax(1),
                    average="macro", zero_division=0)),
            }
        res[prefix] = blk
    return res
