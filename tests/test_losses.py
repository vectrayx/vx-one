"""Losses y pesos (spec F2 §3): ω sqrt media-1 contra tablas exactas §3.5,
CORN pmf válida + loss por tareas elegibles, Kendall con máscaras."""

import numpy as np
import pytest
import torch

from src.models.losses import (KendallMultiTask, choice_loss, class_weights,
                               corn_expected, corn_loss, corn_pmf, noul_loss)

# ── ω: tablas exactas de la spec §3.5 (n → ω, 4 decimales) ────────────
UNSW_TABLE = {
    "normal": (47_600, 0.2067), "dos": (10_424, 0.4417), "recon": (8_917, 0.4776),
    "exploitation": (28_384, 0.2677), "fuzzing": (15_456, 0.3628),
    "analysis": (1_700, 1.0939), "backdoor": (1_484, 1.1708),
    "shellcode": (963, 1.4534), "worms": (111, 4.2808), "generic": (34_000, 0.2446),
}
CIC_TABLE = {
    "normal": (1_591_168, 0.0102), "dos_ddos": (266_482, 0.0249),
    "recon": (111_251, 0.0386), "bruteforce": (9_684, 0.1307),
    "webattack": (1_526, 0.3292), "botnet": (1_376, 0.3467),
    "infiltration": (25, 2.5723), "exploitation": (8, 4.5473),
}
CIC_DDOS_TABLE = {
    "normal": (376_380, 0.3489), "tcp_flood": (89_598, 0.7150),
    "http_dos": (169_034, 0.5206), "slowloris": (7_850, 2.4156),
}
EMBER_TABLE = {
    "berbew": (222_302, 0.4750), "cosmu": (39_526, 1.1265),
    "wacatac": (37_658, 1.1541), "sfone": (43_720, 1.0711),
    "upatre": (42_294, 1.0890), "grandoreiro": (40_698, 1.1101),
    "glupteba": (38_180, 1.1462), "expiro": (33_382, 1.2258),
    "sinowal": (34_488, 1.2059), "flystudio": (32_344, 1.2453),
    "unknown": (2_195_408, 0.1511),
}


@pytest.mark.parametrize("table", [UNSW_TABLE, CIC_TABLE, CIC_DDOS_TABLE, EMBER_TABLE],
                         ids=["unsw", "cic", "cic_ddos", "ember"])
def test_class_weights_match_spec_tables(table):
    counts = [v[0] for v in table.values()]
    w = class_weights(counts)
    expected = np.array([v[1] for v in table.values()], dtype=np.float64)
    np.testing.assert_allclose(w, expected, atol=5e-5)
    np.testing.assert_allclose(w.mean(), 1.0, atol=1e-6)  # media 1 (§3.5)


def test_class_weights_rejects_zero_count():
    with pytest.raises(AssertionError):
        class_weights([10, 0, 5])


def test_class_weights_monotone():
    w = class_weights([1000, 100, 10])
    assert w[0] < w[1] < w[2]  # más rara → más peso


# ── Noul (§3.2, Prop. 1) ───────────────────────────────────────────────
def test_noul_loss_is_plain_bce():
    z = torch.tensor([1.0, -1.0, 2.0])
    y = torch.tensor([1, 0, 1])
    expected = torch.nn.functional.binary_cross_entropy_with_logits(z, y.float())
    assert torch.allclose(noul_loss(z, y), expected)


# ── Choice (§3.3) ──────────────────────────────────────────────────────
def test_choice_loss_weighted_reference():
    z = torch.tensor([[2.0, 0.0, -1.0], [0.5, 1.5, 0.0]])
    y = torch.tensor([0, 1])
    omega = torch.tensor([2.0, 1.0, 0.5])
    logp = torch.log_softmax(z, dim=1)
    ref = -(omega[y] * logp.gather(1, y.unsqueeze(1)).squeeze(1)).sum() / omega[y].sum()
    assert torch.allclose(choice_loss(z, y, omega), ref)


# ── CORN (§3.4) ────────────────────────────────────────────────────────
def test_corn_pmf_valid_distribution():
    torch.manual_seed(0)
    logits = torch.randn(64, 3) * 3
    C, pi = corn_pmf(logits)
    np.testing.assert_allclose(pi.sum(1).detach().numpy(), 1.0, atol=1e-6)
    assert (pi >= -1e-7).all()
    # acumuladas consistentes: C_k = Σ_{j≥k} π_j
    np.testing.assert_allclose(C[:, 1].numpy(), pi[:, 1:].sum(1).numpy(), atol=1e-6)


def test_corn_pmf_matches_manual():
    logits = torch.tensor([[0.5, -1.0, 2.0]])
    q = torch.sigmoid(logits)
    C0, C1, C2, C3 = 1.0, q[0, 0], q[0, 0] * q[0, 1], q[0, 0] * q[0, 1] * q[0, 2]
    pi_ref = [C0 - C1, C1 - C2, C2 - C3, C3]
    _, pi = corn_pmf(logits)
    np.testing.assert_allclose(pi[0].detach().numpy(), pi_ref, atol=1e-7)


def test_corn_loss_eligible_masks():
    # referencia manual de la suma de 3 BCE condicionales (media por tarea)
    corn_logits = torch.tensor([[1.0, 0.5, -0.5],
                                [-2.0, 1.0, 0.3],
                                [0.2, 0.1, 0.9]])
    y = torch.tensor([3, 1, 0])
    total = 0.0
    for k in (1, 2, 3):
        m = y >= (k - 1)
        if m.any():
            tgt = (y[m] >= k).float()
            total += torch.nn.functional.binary_cross_entropy_with_logits(
                corn_logits[m, k - 1], tgt).item()
    assert abs(corn_loss(corn_logits, y).item() - total) < 1e-6


def test_corn_expected_in_range():
    torch.manual_seed(1)
    _, pi = corn_pmf(torch.randn(32, 3) * 2)
    s = corn_expected(pi)
    assert (s >= 0).all() and (s <= 3).all()


# ── Kendall homeostático (§3.6) ────────────────────────────────────────
def test_kendall_init_lambda_one():
    k = KendallMultiTask(("noul", "choice", "score"))
    losses = {"noul": torch.tensor(1.0), "choice": torch.tensor(2.0),
              "score": torch.tensor(3.0)}
    total, lambdas = k(losses, {"noul": True, "choice": True, "score": True})
    assert abs(total.item() - (1 + 2 + 3)) < 1e-6  # s=0 → λ=1, L+s=L
    assert lambdas == {"noul": 1.0, "choice": 1.0, "score": 1.0}


def test_kendall_inactive_head_excluded():
    k = KendallMultiTask(("noul", "ddos"))
    losses = {"noul": torch.tensor(1.0)}  # ddos sin filas en el batch
    total, lambdas = k(losses, {"noul": True, "ddos": False})
    assert abs(total.item() - 1.0) < 1e-6  # el término s_ddos tampoco aporta


def test_kendall_homeostasis_pulls_loss_down():
    # con s_q>0 la cabeza cara recibe λ<1 pero paga +s: la masa no se evapora
    k = KendallMultiTask(("a",))
    with torch.no_grad():
        k.s["a"].fill_(1.0)
    total, lambdas = k({"a": torch.tensor(10.0, requires_grad=True)}, {"a": True})
    assert abs(total.item() - (torch.exp(torch.tensor(-1.0)).item() * 10.0 + 1.0)) < 1e-6
    assert abs(lambdas["a"] - np.exp(-1.0)) < 1e-6


# ── Prop. 2: de-tilt recupera el posterior ─────────────────────────────
def test_de_tilt_recovers_posterior():
    """z* = log π + log ω (óptimo del riesgo ponderado, §3.3 Prop. 2) →
    softmax(z − log ω) = π."""
    torch.manual_seed(3)
    pi = torch.softmax(torch.randn(8, 5), dim=1)
    omega = class_weights([500, 50, 5, 1, 2000])
    z_star = torch.log(pi) + torch.log(torch.tensor(omega)).unsqueeze(0)
    p = torch.softmax(z_star - torch.log(torch.tensor(omega)).unsqueeze(0), dim=1)
    np.testing.assert_allclose(p.detach().numpy(), pi.numpy(), atol=1e-6)
