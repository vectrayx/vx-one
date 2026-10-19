"""Smoke E2E en CPU: parquet UNSW sintético mínimo → build_run → train_run
(2 epochs) → artefacto §8.2 → VXOneEngine predict con invariantes."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.models.data import build_run
from src.models.engine import VXOneEngine

D = 12
N_TRAIN, N_VAL, N_TEST = 1600, 320, 320
FAMILIES = ["normal", "dos", "recon", "exploitation", "generic"]
TXT = {"proto": ["tcp", "udp", None], "service": ["http", "-", None],
       "state": ["FIN", "INT", "ECO"]}


def _make_unsw_tiny(tmp_path):
    rng = np.random.default_rng(0)
    def block(n, split):
        x = rng.normal(size=(n, D))
        x[rng.random(x.shape) < 0.02] = np.nan  # flag de fila
        fam = rng.choice(FAMILIES, size=n, p=[0.6, 0.15, 0.1, 0.1, 0.05])
        sev = rng.integers(0, 4, size=n).astype(np.int64)
        return pa.table({
            "x_num": pa.FixedSizeListArray.from_arrays(
                pa.array(x.reshape(-1), type=pa.float64()), D),
            "x_txt_proto": pa.array([TXT["proto"][i % 3] for i in range(n)]),
            "x_txt_service": pa.array([TXT["service"][i % 3] for i in range(n)]),
            "x_txt_state": pa.array([TXT["state"][i % 3] for i in range(n)]),
            "split": pa.array([split] * n),
            "label_family": pa.array(fam),
            "label_full": pa.array(fam),
            "label_severity": pa.array(sev),
        })
    d = tmp_path / "unsw_nb15"
    d.mkdir()
    tbl = pa.concat_tables([block(N_TRAIN, "train"), block(N_VAL, "val"),
                            block(N_TEST, "test")])
    pq.write_table(tbl, d / "unsw_nb15.parquet")


def test_end_to_end_smoke(tmp_path, monkeypatch):
    _make_unsw_tiny(tmp_path)
    import src.models.data as data_mod
    monkeypatch.setattr(data_mod, "DATA", tmp_path)

    run = build_run("unsw", "nano", "oficial")
    assert run.candidates == FAMILIES and run.train.shape == (N_TRAIN, D + 1)

    out = tmp_path / "art.pt"
    from src.models.train import train_run
    res = train_run(run, out, device="cpu", amp="off", epochs=2,
                    batch_size=256, log=lambda *_: None)

    assert out.exists()
    assert res["best_epoch"] >= 0 and 0.0 <= res["m_val"] <= 1.0

    eng = VXOneEngine(out)
    assert eng.candidates == FAMILIES and eng.mode == "abstain"
    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, D))
    dec = eng.predict_batch(x, {k: [TXT[k][i % 3] for i in range(16)] for k in TXT})
    assert len(dec) == 16
    d0 = dec[0]
    assert abs(sum(d0.choice["family"].probs.values()) - 1.0) < 1e-5
    assert d0.score is not None and 0 <= d0.score.expected <= 3
    assert set(d0.policy) == {"g_attack", "g_drop", "g_recon"}
    # manifiesto del artefacto
    import torch
    ckpt = torch.load(out, weights_only=False)
    man = ckpt["manifest"]
    assert man["candidates"] == FAMILIES and man["seed"] == 42
    assert set(man["temperatures"]) == {"noul", "choice", "score"}
