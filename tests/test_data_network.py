"""Integración con parquets canónicos (spec F2 §0.1): conteos/ω contra las
tablas exactas §3.5, splits, LOAO, D12 (candidate set por corrida)."""

import numpy as np
import pytest
import torch

from src.models.data import build_network_run
from src.models.losses import class_weights

UNSW_COUNTS = {"normal": 47_600, "dos": 10_424, "recon": 8_917,
               "exploitation": 28_384, "shellcode": 963, "backdoor": 1_484,
               "analysis": 1_700, "fuzzing": 15_456, "generic": 34_000,
               "worms": 111}  # orden congelado del registro (§3.1)
CIC_COUNTS = {"normal": 1_591_168, "dos_ddos": 266_482, "recon": 111_251,
              "bruteforce": 9_684, "webattack": 1_526, "botnet": 1_376,
              "infiltration": 25, "exploitation": 8}


@pytest.mark.slow
def test_unsw_run_counts_and_omega():
    run = build_network_run("unsw", "nano", "oficial")
    assert run.meta["train_counts"] == UNSW_COUNTS
    assert run.candidates == list(UNSW_COUNTS)
    w = class_weights(list(UNSW_COUNTS.values()))
    np.testing.assert_allclose(run.omega.numpy(), w, rtol=1e-6)
    # splits oficiales
    assert run.train.shape[0] == 149_039
    assert run.val.shape[0] == 26_302
    assert run.test.shape[0] == 82_332
    assert run.train.shape[1] == 40  # d=39 + flag
    # vocab UNSW: proto 133 tokens → 135 slots (§2.3)
    assert run.vocab["proto"].size == 135
    assert run.vocab["service"].size == 15
    assert run.vocab["state"].size == 11
    assert run.heads.n_candidates == 10


@pytest.mark.slow
def test_cic_strat_counts_and_ddos():
    run = build_network_run("cic", "nano", "strat")
    assert run.meta["train_counts"] == CIC_COUNTS
    assert run.candidates == list(CIC_COUNTS)
    assert run.heads.ddos_candidates == 4
    assert run.ddos_candidates == ["normal", "tcp_flood", "http_dos", "slowloris"]
    # ω del subconjunto ddos (§3.5): 376,380/89,598/169,034/7,850
    m = run.train_labels.ddos_mask
    counts = [int((run.train_labels.y_ddos[m] == i).sum()) for i in range(4)]
    assert counts == [376_380, 89_598, 169_034, 7_850]
    np.testing.assert_allclose(run.omega_ddos.numpy(),
                               class_weights(counts), rtol=1e-6)


@pytest.mark.slow
def test_cic_time_d12_drops_recon_botnet():
    run = build_network_run("cic", "nano", "time")
    # §3.3 D12: recon y botnet sin filas en train temporal
    assert "recon" not in run.candidates and "botnet" not in run.candidates
    assert run.heads.ddos_candidates == 0  # sin head ddos en split_time
    assert run.heads.n_candidates == 6


@pytest.mark.slow
def test_loao_excludes_family_from_train_and_val():
    run = build_network_run("unsw", "nano", "oficial", exclude_family="worms")
    assert "worms" not in run.candidates
    assert run.heads.n_candidates == 9
    # train y val sin worms (111 y 19 filas del censo §3.5); test completo §6.7
    assert run.train.shape[0] == 149_039 - 111
    assert run.val.shape[0] == 26_302 - 19
    assert run.test.shape[0] == 82_332
    # las 44 filas de worms en test quedan como sentinela -1 (E2 las mide)
    assert int((run.test_labels.y_fam == -1).sum()) == 44
    assert int((run.test_fam_names == "worms").sum()) == 44
    # el resto de test mapea a candidatos válidos
    assert int((run.test_labels.y_fam >= 0).sum()) == 82_332 - 44


def test_network_run_labels_consistency():
    run = build_network_run("unsw", "nano", "oficial")
    # y_bin = 1[family != normal]
    normal_idx = run.candidates.index("normal")
    np.testing.assert_array_equal(
        (run.train_labels.y_fam != normal_idx).numpy(),
        run.train_labels.y_bin.numpy())
