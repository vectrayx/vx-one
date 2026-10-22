"""Temperature calibration (spec F2 §5): 1 scalar per head, fitted
in val (never test), NLL criterion (proper scoring rule; the ECE is reported
but does not participate in the selection).

Deterministic search: log-uniform grid of 200 points in [0.05, 20] +
refinement by golden section in the best range up to tolerance
relative 1e-3.

- Noul: NLL of σ(z/T) against and binary.
- Choice: de-tilt z − log ω FIRST (part of contract, Prop. 2), then
NLL of softmax(z'/T) against the true class.
- Score CORN: a single T scales the three conditional logits and the pmf is
recomposes; NLL of π^(T) at the true level.
"""

from __future__ import annotations

import numpy as np
import torch

GRID_LO, GRID_HI, GRID_N = 0.05, 20.0, 200
GOLDEN_TOL = 1e-3

_PHI = (np.sqrt(5.0) - 1.0) / 2.0  # 1/φ ≈ 0.618


def _nll_binary(logits: torch.Tensor, y: torch.Tensor, T: float) -> float:
    z = logits / T
    return float(torch.nn.functional.binary_cross_entropy_with_logits(
        z, y.float()).item())


def _nll_multiclass(z_detailed: torch.Tensor, y: torch.Tensor, T: float) -> float:
    logp = torch.log_softmax(z_detailed / T, dim=1)
    return float(-logp.gather(1, y.unsqueeze(1)).mean().item())


def _golden_minimize(f, lo: float, hi: float) -> float:
    """Golden section over [lo, hi] up to relative tolerance 1e-3."""
    a, b = lo, hi
    c = b - _PHI * (b - a)
    d = a + _PHI * (b - a)
    fc, fd = f(c), f(d)
    while (b - a) / max(a, 1e-12) > GOLDEN_TOL and (b - a) > 1e-9:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _PHI * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _PHI * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return float(x)


def fit_temperature_scalar(f_nll) -> float:
    """Grid log-uniform 200 points in [GRID_LO, GRID_HI] + gold section."""
    grid = np.logspace(np.log10(GRID_LO), np.log10(GRID_HI), GRID_N)
    vals = np.array([f_nll(float(T)) for T in grid])
    i = int(np.argmin(vals))
    lo = grid[max(i - 1, 0)]
    hi = grid[min(i + 1, GRID_N - 1)]
    return _golden_minimize(f_nll, lo, hi)


def fit_noul_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    return fit_temperature_scalar(lambda T: _nll_binary(logits, y, T))


def fit_choice_temperature(logits: torch.Tensor, y: torch.Tensor,
                           omega: torch.Tensor) -> float:
    """Analytical de-tilt BEFORE searching for T (order matters, §5.2)."""
    z_tilted = logits - torch.log(omega).unsqueeze(0)
    return fit_temperature_scalar(lambda T: _nll_multiclass(z_tilted, y, T))


def fit_corn_temperature(corn_logits: torch.Tensor, y_sev: torch.Tensor) -> float:
    """Common T over the 3 conditionals; NLL of the recomposed pmf."""
    def f(T: float) -> float:
        from .losses import corn_pmf
        _, pi = corn_pmf(corn_logits / T)
        pi = pi.clamp_min(1e-12)
        return float(-torch.log(pi.gather(1, y_sev.unsqueeze(1))).mean().item())
    return fit_temperature_scalar(f)
