"""Calibración por temperatura (spec F2 §5): recuperación de T conocido,
determinismo, orden de-tilt→T (§5.2), T conjunta CORN."""

import numpy as np
import pytest
import torch

from src.models.calibration import (GRID_HI, GRID_LO, fit_choice_temperature,
                                    fit_corn_temperature,
                                    fit_noul_temperature, fit_temperature_scalar)
from src.models.losses import class_weights, corn_pmf


def _binary_sample(n, T_true, seed):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=n)                       # logits "verdaderos"
    y = (rng.random(n) < 1 / (1 + np.exp(-z0))).astype(np.float32)
    return torch.tensor(z0 * T_true, dtype=torch.float32), torch.tensor(y)


def test_noul_recovers_known_T():
    logits, y = _binary_sample(60_000, T_true=1.7, seed=0)
    T = fit_noul_temperature(logits, y)
    assert T == pytest.approx(1.7, rel=0.02)


def test_noul_T_within_grid_bounds():
    logits, y = _binary_sample(5_000, T_true=0.3, seed=1)
    T = fit_noul_temperature(logits, y)
    assert GRID_LO <= T <= GRID_HI


def test_calibration_deterministic():
    logits, y = _binary_sample(10_000, T_true=2.0, seed=2)
    assert fit_noul_temperature(logits, y) == fit_noul_temperature(logits, y)


def test_noul_recovers_underconfident_T():
    # T_true < 1: modelo demasiado suave → T óptimo < 1 (sharpening)
    logits, y = _binary_sample(60_000, T_true=0.5, seed=3)
    assert fit_noul_temperature(logits, y) == pytest.approx(0.5, rel=0.03)


def test_choice_detilt_before_T():
    """z = log π + log ω (logits con tilt exacto) → tras de-tilt, T*=1.
    Sin de-tilt la búsqueda sobre z/T no devolvería 1 en general."""
    torch.manual_seed(4)
    n = 40_000
    pi = torch.softmax(torch.randn(n, 5) * 1.5, dim=1)
    omega = torch.tensor(class_weights([900, 300, 90, 30, 10]))
    y = torch.multinomial(pi, 1).squeeze(1)
    z = torch.log(pi) + torch.log(omega).unsqueeze(0)
    T = fit_choice_temperature(z, y, omega)
    assert T == pytest.approx(1.0, rel=0.02)


def test_corn_joint_T_recovered():
    torch.manual_seed(5)
    n = 60_000
    pi = torch.softmax(torch.randn(n, 4) * 1.2, dim=1)
    y = torch.multinomial(pi, 1).squeeze(1)
    C = 1.0 - torch.cumsum(pi, dim=1)[:, :3]      # C_k = P(y ≥ k), k=1..3
    q = torch.cat([C[:, 0:1], C[:, 1:2] / C[:, 0:1].clamp_min(1e-9),
                   C[:, 2:3] / C[:, 1:2].clamp_min(1e-9)], dim=1)  # q1,q2,q3
    T_true = 1.4
    logits = torch.logit(q.clamp(1e-6, 1 - 1e-6)) * T_true
    T = fit_corn_temperature(logits, y)
    assert T == pytest.approx(1.4, rel=0.03)


def test_golden_section_refines_grid():
    # mínimo fino entre puntos del grid: T*=1.23 con curva cuadrática aguda
    logits, y = _binary_sample(80_000, T_true=1.23, seed=6)
    assert fit_noul_temperature(logits, y) == pytest.approx(1.23, rel=0.02)
