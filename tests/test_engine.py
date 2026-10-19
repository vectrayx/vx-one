"""Engine (spec F2 §8): artefacto self-contained, invariantes tipados,
abstención τ_abs, gates derivados (D9), round-trip exacto."""

import numpy as np
import pytest
import torch

from src.models.engine import VXOneEngine, save_artifact
from src.models.feature_stage import FeatureStage
from src.models.losses import corn_pmf
from src.models.network import HeadSpec, VXOneNet
from src.models.vocab import TextVocab

D = 6
VOCABS = {"proto": 4, "service": 3, "state": 3}  # V+2 por campo


def _synthetic_data(n=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, D))
    # conteos de tokens: proto 2→size 4, service 1→3, state 1→3 (= VOCABS)
    txt = {"proto": ["tcp" if i % 2 else "udp" for i in range(n)],
           "service": ["http"] * n, "state": ["fin"] * n}
    return x, txt


def _train_tiny_net(K=4, ddos=0, score=True, seed=7):
    torch.manual_seed(seed)
    x, txt = _synthetic_data()
    stage = FeatureStage.fit(x[:48])
    vocab = {k: TextVocab.fit(k, txt[k][:48]) for k in VOCABS}
    net = VXOneNet("unsw", "nano", HeadSpec(n_candidates=K, ddos_candidates=ddos,
                                            has_score=score), D, vocab_sizes=VOCABS)
    return net, stage, vocab


def _save(net, stage, vocab, path, K=4, mode="abstain", ddos=None, score=True,
          tau_abs=0.5):
    torch.manual_seed(11)
    omega = torch.tensor([1.0, 1.0, 1.0, 1.0])[:K]
    if omega.shape[0] != K:
        omega = torch.ones(K)
    save_artifact(path, net, run_meta={
        "source_key": "unsw_nb15_v1",
        "feature_stage": stage, "vocab": vocab,
        "candidates": ["normal", "recon", "dos", "generic"][:K],
        "omega": omega,
        "omega_ddos": ddos,
        "ddos_candidates": (["normal", "tcp_flood", "http_dos", "slowloris"]
                            if ddos is not None else None),
        "mode": mode,
        "tau_abs": tau_abs,
        "config": {"scale": "nano", "split_scheme": "oficial"},
        "manifest": {"run_id": "test"},
    }, temperatures={"noul": 1.1, "choice": 0.9,
                    **({"score": 1.3} if score else {}),
                    **({"ddos": 1.2} if ddos is not None else {})},
        d_num=D)


def test_roundtrip_forward_exact(tmp_path):
    net, stage, vocab = _train_tiny_net()
    art = tmp_path / "art.pt"
    _save(net, stage, vocab, art)
    eng = VXOneEngine(art)
    x, txt = _synthetic_data(16, seed=1)
    u, _ = stage.transform(x)
    slots = {k: torch.tensor(vocab[k].encode(txt[k])) for k in vocab}
    net.eval()
    with torch.no_grad():
        ref = net(torch.tensor(u), slots)
    dec = eng.predict_batch(x, txt)
    p_ref = torch.sigmoid(ref["noul"] / 1.1)
    np.testing.assert_allclose(
        [d.noul["is_attack"] for d in dec], p_ref.numpy(), atol=1e-6)


def test_choice_probs_sum_to_one_over_natives(tmp_path):
    net, stage, vocab = _train_tiny_net(K=4)
    art = tmp_path / "art.pt"
    _save(net, stage, vocab, art, K=4)
    eng = VXOneEngine(art)
    x, txt = _synthetic_data(32, seed=2)
    for dec in eng.predict_batch(x, txt):
        probs = dec.choice["family"].probs
        assert set(probs) == set(eng.candidates)          # solo nativos
        assert abs(sum(probs.values()) - 1.0) < 1e-5
        assert dec.choice["family"].candidate in eng.candidates or \
               dec.choice["family"].candidate == "unknown"


def test_abstention_tau(tmp_path):
    net, stage, vocab = _train_tiny_net(K=4)
    x, txt = _synthetic_data(32, seed=3)
    # τ_abs=1.01 → nadie supera el umbral → todos unknown
    art_hi = tmp_path / "hi.pt"
    _save(net, stage, vocab, art_hi, K=4, tau_abs=1.01)
    eng = VXOneEngine(art_hi)
    dec = eng.predict_batch(x, txt)
    assert all(d.choice["family"].candidate == "unknown" for d in dec)
    # τ_abs=0 → siempre candidato nativo
    art_lo = tmp_path / "lo.pt"
    _save(net, stage, vocab, art_lo, K=4, tau_abs=0.0)
    eng = VXOneEngine(art_lo)
    dec = eng.predict_batch(x, txt)
    assert all(d.choice["family"].candidate in eng.candidates for d in dec)


def test_trained_unknown_mode(tmp_path):
    # EMBER: unknown es candidato entrenado; sin Score
    torch.manual_seed(9)
    net = VXOneNet("ember", "nano", HeadSpec(n_candidates=5, has_score=False), D)
    stage = FeatureStage.fit(_synthetic_data()[0][:48])
    art = tmp_path / "ember.pt"
    omega = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.5])
    save_artifact(art, net, run_meta={
        "source_key": "ember2024_v1", "feature_stage": stage, "vocab": None,
        "candidates": ["berbew", "cosmu", "wacatac", "sinowal", "unknown"],
        "omega": omega, "omega_ddos": None, "mode": "trained_unknown",
        "config": {"scale": "nano", "split_scheme": "temporal"},
        "manifest": {"run_id": "test-ember"},
    }, temperatures={"noul": 1.0, "choice": 1.0}, d_num=D)
    eng = VXOneEngine(art)
    x, _ = _synthetic_data(16, seed=4)
    dec = eng.predict_batch(x, None)
    assert all(d.score is None for d in dec)
    assert all(d.noul["is_malicious"] == d.noul["is_malicious"] for d in dec)
    assert "g_recon" not in dec[0].policy and "g_drop" not in dec[0].policy
    # probs suman 1 incluyendo unknown como nativo
    assert abs(sum(dec[0].choice["family"].probs.values()) - 1.0) < 1e-5
    assert "unknown" in eng.candidates


def test_gates_derived_from_calibrated_outputs(tmp_path):
    net, stage, vocab = _train_tiny_net(K=4)
    art = tmp_path / "art.pt"
    _save(net, stage, vocab, art, K=4)
    eng = VXOneEngine(art)
    x, txt = _synthetic_data(64, seed=5)
    for d in eng.predict_batch(x, txt):
        p = d.noul["is_attack"]
        assert d.policy["g_attack"] == (p >= 0.5)
        pi = np.array(list(d.score.probs.values()))
        assert d.policy["g_drop"] == ((pi[2] + pi[3]) >= 0.5)
        assert d.policy["g_recon"] == (d.choice["family"].probs.get("recon", 0.0) >= 0.5)


def test_score_sum_and_range(tmp_path):
    net, stage, vocab = _train_tiny_net(K=4)
    art = tmp_path / "art.pt"
    _save(net, stage, vocab, art, K=4)
    eng = VXOneEngine(art)
    x, txt = _synthetic_data(32, seed=6)
    for d in eng.predict_batch(x, txt):
        assert abs(sum(d.score.probs.values()) - 1.0) < 1e-5
        assert 0.0 <= d.score.expected <= 3.0


def test_predict_single_row(tmp_path):
    net, stage, vocab = _train_tiny_net(K=4)
    art = tmp_path / "art.pt"
    _save(net, stage, vocab, art, K=4)
    eng = VXOneEngine(art)
    x, txt = _synthetic_data(4, seed=8)
    d = eng.predict(x[0], {k: v[0] for k, v in txt.items()})
    assert d.source == "unsw" and "family" in d.choice


def test_engine_has_ddos_head(tmp_path):
    net, stage, vocab = _train_tiny_net(K=4, ddos=4)
    art = tmp_path / "art.pt"
    omega_d = torch.tensor([0.5, 1.0, 1.5, 2.0])
    _save(net, stage, vocab, art, K=4, ddos=omega_d)
    eng = VXOneEngine(art)
    x, txt = _synthetic_data(8, seed=10)
    d = eng.predict_batch(x, txt)[0]
    dv = d.choice["ddos_vector"]
    assert set(dv.probs) == {"normal", "tcp_flood", "http_dos", "slowloris"}
    assert abs(sum(dv.probs.values()) - 1.0) < 1e-5
