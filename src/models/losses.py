"""VX-One engine losses (spec F2 §3).

- Noul: BCE without weighting (Prop. 1: weighting introduces a shift
additive log(α+/α−) in the logit that temperature scaling does not reverse).
- Choice: weighted CE with ω_k = K·n_k^{-1/2}/Σ_j n_j^{-1/2} (mean 1);
the de-tilt z − log ω is from INFERENCE (Prop. 2), not from the loss.
- Score: CORN (Shi, Cao & Raschka 2021) — BCE conditional by task k=1..3
over eligible rows and ≥ k−1, valid pmf by construction.
- Multi-task: homeostatic uncertainty from Kendall (2018) modified:
s_q learnable init 0 + activity mask a_q per batch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


# ── class weights (frozen formula §3.5) ───────────────────────────
def class_weights(train_counts: np.ndarray | list[int]) -> np.ndarray:
    """ω_k = K·n_k^{-1/2} / Σ_j n_j^{-1/2}; average 1 over the K classes."""
    n = np.asarray(train_counts, dtype=np.float64)
    assert (n > 0).all(), f"conteo de train 0: {n} — el candidate set (D12) debe excluir clases sin filas"
    inv_sqrt = n ** -0.5
    return (len(n) * inv_sqrt / inv_sqrt.sum()).astype(np.float32)


# ── Noul (§3.2) ────────────────────────────────────────────────────────
def noul_loss(logits: torch.Tensor, y_bin: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, y_bin.float())


# ── Choice (§3.3) ──────────────────────────────────────────────────────
def choice_loss(logits: torch.Tensor, y_idx: torch.Tensor,
                omega: torch.Tensor) -> torch.Tensor:
    """CE ponderada: −Σ ω_{y_i}·log softmax(z_i)[y_i] / Σ ω_{y_i}."""
    logp = nn.functional.log_softmax(logits, dim=1)
    w = omega[y_idx]
    return -(w * logp.gather(1, y_idx.unsqueeze(1)).squeeze(1)).sum() / w.sum()


# ── Score CORN (§3.4) ──────────────────────────────────────────────────
def corn_pmf(corn_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,3] conditional logits → (cumulative C [B,4] with C0=1, π pmf [B,4])."""
    q = torch.sigmoid(corn_logits)                      # q_k = P(y ≥ k | y ≥ k−1)
    C = torch.cat(
        [torch.ones_like(q[:, :1]), torch.cumprod(q, dim=1)], dim=1
    )                                                   # [C0..C3]
    pi = torch.cat(
        [C[:, :3] - C[:, 1:], C[:, 3:]], dim=1
    )                                                   # [π0..π3]
    return C, pi


def corn_loss(corn_logits: torch.Tensor, y_sev: torch.Tensor) -> torch.Tensor:
    """Sum of the 3 conditional ECBs, each one normalized by its rows
eligible and ≥ k−1 (average reduction per task, not overall)."""
    total = corn_logits.new_zeros(())
    for k in (1, 2, 3):
        m = y_sev >= (k - 1)
        if not m.any():
            continue
        target = (y_sev[m] >= k).float()
        total = total + nn.functional.binary_cross_entropy_with_logits(
            corn_logits[m, k - 1], target
        )
    return total


def corn_expected(pi: torch.Tensor) -> torch.Tensor:
    """ŝ = Σ_k k·π_k ∈ [0,3] (typed decoding of F0 §2.3)."""
    levels = torch.arange(4, device=pi.device, dtype=pi.dtype)
    return (pi * levels.unsqueeze(0)).sum(dim=1)


# ── Multi-task Kendall homeostatic modified (§3.6) ──────────────────
class KendallMultiTask(nn.Module):
    """L = Σ_q a_q·[e^{−s_q}·L̃_q + s_q]; s_q init 0 → λ_q starts at 1.
a_q = 0 when the batch has no rows eligible for head q
(the term +s_q does not contribute either: without activity s_q should not be derived)."""

    def __init__(self, head_names: tuple[str, ...]) -> None:
        super().__init__()
        self.head_names = tuple(head_names)
        self.s = nn.ParameterDict({q: nn.Parameter(torch.zeros(())) for q in head_names})

    def forward(
        self, losses: dict[str, torch.Tensor], active: dict[str, bool]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = None
        lambdas: dict[str, float] = {}
        for q in self.head_names:
            lambdas[q] = float(torch.exp(-self.s[q]).detach())
            if not active.get(q, False):
                continue
            term = torch.exp(-self.s[q]) * losses[q] + self.s[q]
            total = term if total is None else total + term
        if total is None:  # batch without any active head (should not occur)
            total = sum(losses.values()) if losses else torch.zeros(())
        return total, lambdas
