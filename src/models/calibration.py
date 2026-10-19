"""Calibración por temperatura (spec F2 §5): 1 escalar por cabeza, fitted
en val (nunca test), criterio NLL (proper scoring rule; el ECE se reporta
pero no participa en la selección).

Búsqueda determinista: grid log-uniforme de 200 puntos en [0.05, 20] +
refinamiento por sección dorada en el mejor intervalo hasta tolerancia
relativa 1e-3.

- Noul: NLL de σ(z/T) contra y binaria.
- Choice: de-tilt z − log ω PRIMERO (parte del contrato, Prop. 2), luego
  NLL de softmax(z'/T) contra la clase verdadera.
- Score CORN: un único T escala los tres logits condicionales y la pmf se
  recompone; NLL de π^(T) en el nivel verdadero.
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
    """Sección dorada sobre [lo, hi] hasta tolerancia relativa 1e-3."""
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
    """Grid log-uniforme 200 puntos en [GRID_LO, GRID_HI] + sección dorada."""
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
    """De-tilt analítico ANTES de la búsqueda de T (orden importa, §5.2)."""
    z_tilted = logits - torch.log(omega).unsqueeze(0)
    return fit_temperature_scalar(lambda T: _nll_multiclass(z_tilted, y, T))


def fit_corn_temperature(corn_logits: torch.Tensor, y_sev: torch.Tensor) -> float:
    """T común sobre las 3 condicionales; NLL de la pmf recompuesta."""
    def f(T: float) -> float:
        from .losses import corn_pmf
        _, pi = corn_pmf(corn_logits / T)
        pi = pi.clamp_min(1e-12)
        return float(-torch.log(pi.gather(1, y_sev.unsqueeze(1))).mean().item())
    return fit_temperature_scalar(f)
