"""Tabular baselines (spec F2 §5.3, anti-strawman): XGBoost, Random Forest
and MLP with the SAME protocol as VX-One.

Symmetry with the engine (everything that VX-One receives, the baselines receive):
- Same inputs: u_r of the FeatureStage (d+1) + one-hot of the fields
texts with the SAME vocab fitted in train (the branch of embeddings
learned is NOT replicated — that edge is measured by E6-A1).
- Same weights ω (§3.5) via sample_weight in Choice/ddos. for being
trained with the same tilt, apply the SAME analytical de-tilt
z − log ω before T (fit_choice_temperature; Prop. 2 symmetric).
- Score WITHOUT weights (symmetrical with CORN D3) and nominal softmax 4 classes —
the ordinal-vs-nominal comparison is part of the design (F2 §3.4).
- Noul without weights (Prop. 1).
- Same calibration: 1 T per head, same NLL criterion, same val,
same search engine (src.models.calibration). Trees only emit probs →
logits = log p (strictly monotonic, same argmax).
- Same stopping criterion M_val (§6.4) with ŝ = Σ k·π_k (identical decode).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn

from src.models.calibration import (fit_choice_temperature,
                                    fit_noul_temperature,
                                    fit_temperature_scalar)

SEED = 42


# ── features: u_r + one-hot textual (same inputs as the engine) ───────
def baseline_features(stage, x_num: np.ndarray,
                      txt: dict[str, list] | None,
                      vocabs: dict | None) -> np.ndarray:
    u, _ = stage.transform(x_num)
    blocks = [u.astype(np.float32)]
    if txt and vocabs:
        for k, vocab in vocabs.items():
            ids = np.asarray(vocab.encode(txt[k]), dtype=np.int64)
            onehot = np.zeros((len(ids), vocab.size), dtype=np.float32)
            onehot[np.arange(len(ids)), ids] = 1.0
            blocks.append(onehot)
    return np.concatenate(blocks, axis=1)


# ── adapter probs → logits (trees do not expose logits) ───────────
def _probs_to_logits(p: np.ndarray, binary: bool) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    if binary:
        return np.log(p / (1 - p))
    return np.log(p)


@dataclass
class HeadCalibration:
    """Temperatures fitted in val (§5.3). score_T None in EMBER."""
    noul_T: float
    choice_T: float
    score_T: float | None = None
    ddos_T: float | None = None


def calibrate_heads(val: dict[str, np.ndarray], omega: torch.Tensor,
                    omega_ddos: torch.Tensor | None = None, *,
                    fam_keep: np.ndarray | None = None,
                    sev_keep: np.ndarray | None = None) -> HeadCalibration:
    """val brings p_noul/p_choice/p_score/p_ddos (probs) + y_*. Same criterion
NLL, same val and same search engine as VX-One (§5). fam_keep/sev_keep:
boolean masks to exclude classful rows from calibration
absent in train (scheme time); p_ddos should arrive pre-trimmed at
ddos mask rows (gather tolerates misaligned silently)."""
    fm = val["fam_mask"] if fam_keep is None else (val["fam_mask"] & fam_keep)
    cal = HeadCalibration(
        noul_T=fit_noul_temperature(
            torch.tensor(_probs_to_logits(val["p_noul"], binary=True)),
            torch.tensor(val["y_bin"])),
        choice_T=fit_choice_temperature(
            torch.tensor(_probs_to_logits(val["p_choice"], binary=False)),
            torch.tensor(val["y_fam"][fm]), omega),
    )
    if "p_score" in val:
        m = np.ones(len(val["y_sev"]), dtype=bool) if sev_keep is None else sev_keep
        z_s = torch.tensor(_probs_to_logits(val["p_score"][m], binary=False))
        y_s = torch.tensor(val["y_sev"][m])
        cal.score_T = fit_temperature_scalar(
            lambda T: -torch.log_softmax(z_s / T, 1)
            .gather(1, y_s.unsqueeze(1)).mean().item())
    if "p_ddos" in val and omega_ddos is not None:
        cal.ddos_T = fit_choice_temperature(
            torch.tensor(_probs_to_logits(val["p_ddos"], binary=False)),
            torch.tensor(val["y_ddos"]), omega_ddos)
    return cal


# ── sample weights ω normalizadas a media 1 ────────────────────────────
def _sample_weights(y: np.ndarray, omega: np.ndarray) -> np.ndarray:
    w = omega[y]
    return w * (len(y) / w.sum())


# ── XGBoost / RF: one model per head ─────────────────────────────────
_XGB_PARAMS = {
    "n_estimators": 500, "max_depth": 8, "learning_rate": 0.1,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "tree_method": "hist", "n_jobs": -1, "random_state": SEED,
    "eval_metric": "logloss",
}


def fit_xgboost(x_tr: np.ndarray, y_tr: np.ndarray,
                w_tr: np.ndarray | None, task: str) -> object:
    """task: 'binary' | 'multiclass' (num_class is derived from y). The valley
reserve for calibration — without internal early stopping (§5.3)."""
    import xgboost as xgb

    params = dict(_XGB_PARAMS)
    if task == "binary":
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = int(y_tr.max()) + 1
    model = xgb.XGBClassifier(**params)
    model.fit(x_tr, y_tr, sample_weight=w_tr)
    return model


def fit_rf(x_tr: np.ndarray, y_tr: np.ndarray,
           w_tr: np.ndarray | None, task: str = "multiclass") -> object:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=SEED)
    model.fit(x_tr, y_tr, sample_weight=w_tr)
    return model


def predict_tree(model, x: np.ndarray, binary: bool) -> np.ndarray:
    """Tree probs → logits-compatible (1D in binary, [N,K] in multi)."""
    p = model.predict_proba(x)
    if binary:
        return p[:, 1].astype(np.float64)
    return p.astype(np.float64)


# ── Tabular MLP: a multi-head network (standard BN→ReLU, not LN) ──────────
class MLPBaseline(nn.Module):
    def __init__(self, d_in: int, K: int, has_score: bool,
                 width: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.1),
        )
        self.noul = nn.Linear(width, 1)
        self.choice = nn.Linear(width, K)
        self.has_score = has_score
        if has_score:
            self.score = nn.Linear(width, 4)  # nominal (vs CORN of VX-One)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        out = {"noul": self.noul(h).squeeze(-1), "choice": self.choice(h)}
        if self.has_score:
            out["score"] = self.score(h)
        return out


def _m_val_mlp(o: dict[str, np.ndarray], has_score: bool) -> float:
    """M_val mirror of train._m_val (§6.4) with nominal score: ŝ = Σ k·π_k."""
    p_mal = 1.0 / (1.0 + np.exp(-o["noul"]))
    fam_mask = o["y_fam"] >= 0
    m = [
        roc_auc_score(o["y_bin"], p_mal)
        if len(np.unique(o["y_bin"])) > 1 else 0.5,
        f1_score(o["y_fam"][fam_mask], np.argmax(o["choice"], 1)[fam_mask],
                 average="macro", zero_division=0),
    ]
    if has_score:
        pi = np.exp(o["score"] - o["score"].max(1, keepdims=True))
        pi /= pi.sum(1, keepdims=True)
        sev_hat = (pi * np.arange(4)).sum(axis=1)
        m.append(1.0 - float(np.abs(sev_hat - o["y_sev"]).mean()) / 3.0)
    return float(np.mean(m))


@torch.no_grad()
def _mlp_forward(net: MLPBaseline, x: np.ndarray, y_bin, y_fam, y_sev,
                 device: str, bs: int = 8192) -> dict[str, np.ndarray]:
    was_training = net.training
    net.eval()
    outs: dict[str, list[np.ndarray]] = {}
    for i in range(0, len(x), bs):
        o = net(torch.tensor(x[i:i + bs], dtype=torch.float32, device=device))
        for k, v in o.items():
            outs.setdefault(k, []).append(v.float().cpu().numpy())
    flat = {k: np.concatenate(v) for k, v in outs.items()}
    flat.update(y_bin=np.asarray(y_bin), y_fam=np.asarray(y_fam))
    if y_sev is not None:
        flat["y_sev"] = np.asarray(y_sev)
    if was_training:
        net.train()
    return flat


def train_mlp(x_tr: np.ndarray, y_bin: np.ndarray, y_fam: np.ndarray,
              omega: np.ndarray, y_sev: np.ndarray | None,
              x_va: np.ndarray, y_bin_va: np.ndarray, y_fam_va: np.ndarray,
              y_sev_va: np.ndarray | None, *,
              epochs: int = 30, batch_size: int = 512, lr: float = 1e-3,
              device: str = "cpu", width: int = 256, patience: int = 4,
              seed: int = SEED, log=print) -> MLPBaseline:
    """MLP multi-head with ω in Choice (weighted CE), Noul/Score without weights,
early stop on the SAME M_val (fam_mask −1 excluded from Choice)."""
    torch.manual_seed(seed)
    K = int(y_fam.max()) + 1
    net = MLPBaseline(x_tr.shape[1], K, has_score=y_sev is not None,
                      width=width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    x_t = torch.tensor(x_tr, dtype=torch.float32, device=device)
    yb_t = torch.tensor(y_bin, dtype=torch.float32, device=device)
    yf_t = torch.tensor(y_fam, dtype=torch.int64, device=device)
    ys_t = (torch.tensor(y_sev, dtype=torch.int64, device=device)
            if y_sev is not None else None)
    fm_t = yf_t >= 0  # rows without family (−1) outside the Choice CE
    n = len(x_t)
    w_fam = torch.zeros(n, dtype=torch.float32, device=device)
    if bool(fm_t.any()):
        w_fam[fm_t] = torch.tensor(_sample_weights(y_fam[fm_t], omega),
                                   dtype=torch.float32)

    best_m, bad, best_state = -1.0, 0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            o = net(x_t[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                o["noul"], yb_t[idx])
            fi = fm_t[idx]
            if bool(fi.any()):
                w_i = w_fam[idx][fi]
                loss = loss + (nn.functional.cross_entropy(
                    o["choice"][fi], yf_t[idx][fi], reduction="none")
                    .mul(w_i).sum() / w_i.sum())
            if ys_t is not None:
                loss = loss + nn.functional.cross_entropy(o["score"], ys_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        va = _mlp_forward(net, x_va, y_bin_va, y_fam_va, y_sev_va, device)
        mv = _m_val_mlp(va, y_sev is not None)
        if mv > best_m:
            best_m, bad = mv, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad >= patience:
            log(f"[mlp] epoch {ep + 1}/{epochs} M_val={mv:.4f} bad={bad}")
        if bad >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net
