"""Conteo de parámetros contra la tabla §2.3 de la spec (exacto) + forward."""

import pytest
import torch

from src.models.network import SCALES, HeadSpec, VXOneNet

# (escala, fuente, d_num, vocab_sizes, K, ddos, has_score, total esperado §2.3)
VOCAB_UNSW = {"proto": 133 + 2, "service": 13 + 2, "state": 9 + 2}  # slots +2

CASES = [
    ("nano",  "unsw",  39,   VOCAB_UNSW, 10, 0, True,  97_230),
    ("nano",  "cic",   77,   None,       8,  4, True,  88_608),
    ("nano",  "ember", 2568, None,       11, 0, False, 95_572),
    ("small", "unsw",  39,   VOCAB_UNSW, 10, 0, True,  913_806),
    ("small", "cic",   77,   None,       8,  4, True,  865_456),
    ("small", "ember", 2568, None,       11, 0, False, 977_676),
    ("base",  "unsw",  39,   VOCAB_UNSW, 10, 0, True,  5_878_350),
    ("base",  "cic",   77,   None,       8,  4, True,  5_644_304),
    ("base",  "ember", 2568, None,       11, 0, False, 6_424_332),
    # variante split_time (D12): CIC sin ddos y K=6 → −(4w+4) −(2w+2)
    ("nano",  "cic",   77,   None,       6,  0, True,  87_258),
]


@pytest.mark.parametrize("scale,src,d,vocabs,K,ddos,score,expected", CASES,
                         ids=[f"{c[0]}-{c[1]}-K{c[4]}" for c in CASES])
def test_param_table_exact(scale, src, d, vocabs, K, ddos, score, expected):
    net = VXOneNet(src, scale, HeadSpec(n_candidates=K, ddos_candidates=ddos,
                                        has_score=score), d, vocab_sizes=vocabs)
    assert net.n_parameters() == expected


def test_budgets():
    assert SCALES["nano"]["L"] == 2 and SCALES["small"]["L"] == 3 and SCALES["base"]["L"] == 4


def test_forward_shapes():
    torch.manual_seed(0)
    net = VXOneNet("cic", "nano", HeadSpec(n_candidates=8, ddos_candidates=4), 77)
    net.eval()
    u = torch.randn(16, 78)  # d+1 = 78 (flag incluido)
    out = net(u)
    assert out["noul"].shape == (16,)
    assert out["choice"].shape == (16, 8)
    assert out["corn_logits"].shape == (16, 3)
    assert out["ddos"].shape == (16, 4)


def test_forward_ember_no_score():
    net = VXOneNet("ember", "nano", HeadSpec(n_candidates=11, has_score=False), 2568)
    out = net(torch.randn(4, 2569))
    assert "corn_logits" not in out and "ddos" not in out
    assert out["choice"].shape == (4, 11)


def test_corn_init_biases():
    net = VXOneNet("unsw", "nano", HeadSpec(n_candidates=10), 39)
    torch.testing.assert_close(net.corn_b.detach(), torch.tensor([1.0, 0.0, -1.0]))


def test_text_branch_concat():
    torch.manual_seed(0)
    vocabs = {"proto": 10, "service": 8, "state": 6}
    net = VXOneNet("unsw", "nano", HeadSpec(n_candidates=10), 39, vocab_sizes=vocabs)
    slots = {f: torch.zeros(5, dtype=torch.long) for f in vocabs}
    out = net(torch.randn(5, 40), slots)
    assert out["choice"].shape == (5, 10)
