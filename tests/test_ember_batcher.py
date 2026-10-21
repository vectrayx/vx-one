"""EmberBatcher (§6.2): shards existentes, drop_last en train, eval
secuencial, determinismo de permutación por epoch."""

from pathlib import Path

import numpy as np
import pytest

from src.models.data import DATA, EmberBatcher, ember_shards

EMBER_DIR = DATA / "ember2024"


def _ember_available() -> bool:
    return EMBER_DIR.exists() and len(ember_shards("train")) > 0


pytestmark = pytest.mark.skipif(not _ember_available(), reason="sin shards EMBER locales")


def test_shard_lists_well_formed():
    tr, va, te = ember_shards("train"), ember_shards("val"), ember_shards("test")
    assert len(tr) == 46 and len(va) == 6 and len(te) == 12
    assert all(p.exists() for p in tr[:3] + va[:2] + te[:2])


@pytest.mark.slow
def test_single_shard_batches_deterministic():
    from src.models.feature_stage import FeatureStage

    # stage de juguete sobre una lectura pequeña: usamos el shard val 46
    p = ember_shards("val")[0]
    import pyarrow.parquet as pq

    tbl = pq.read_table(p, columns=["x_num"])
    import numpy as np

    arr = tbl.column("x_num").combine_chunks()
    flat = arr.flatten().to_numpy(zero_copy_only=False).reshape(-1, arr.type.list_size)
    stage = FeatureStage.fit(flat[:5000].astype(np.float64))

    run = type("R", (), {})()  # stub con lo que EmberBatcher usa
    run.feature_stage = stage
    from src.state.state import SOURCE_REGISTRY
    run.candidates = list(SOURCE_REGISTRY["ember2024_v1"]["native_families"])

    b = EmberBatcher(run)
    u, y_fam, y_bin = b._load_shard(p)
    assert u.shape[0] == flat.shape[0] and u.shape[1] == 2569  # d+1 flag
    assert set(y_fam.tolist()) <= set(range(11))  # 11 candidatos S6
    assert set(y_bin.tolist()) <= {0, 1}

    # train con rng determinista: mismo epoch → mismos batches
    g1 = list(b._iter_shard_batches(p, 512, np.random.default_rng(42)))
    g2 = list(b._iter_shard_batches(p, 512, np.random.default_rng(42)))
    assert len(g1) == len(g2) == flat.shape[0] // 512  # drop_last
    np.testing.assert_array_equal(g1[0][0].numpy(), g2[0][0].numpy())

    # eval: sin drop_last, orden secuencial
    ge = list(b._iter_shard_batches(p, 512, None))
    assert len(ge) == -(-flat.shape[0] // 512)
