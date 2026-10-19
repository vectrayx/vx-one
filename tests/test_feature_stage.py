"""FeatureStage (spec F2 §2.1/D6): mediana de no-finitos + flag de fila,
signed log1p, escala robusta (mediana/IQR de train), clip ±8."""

import numpy as np
import pytest

from src.models.feature_stage import FeatureStage

CLIP = 8.0


def test_fit_robust_stats_on_train():
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 5.0, size=(10_000, 3))
    x[:, 2] *= 1e6  # escala muy distinta por columna
    st = FeatureStage.fit(x)
    assert st.impute_median.shape == (3,)
    # center ≈ mediana de log1p, scale ≈ IQR de log1p
    v = np.sign(x) * np.log1p(np.abs(x))
    med = np.median(v, axis=0)
    iqr = np.percentile(v, 75, axis=0) - np.percentile(v, 25, axis=0)
    np.testing.assert_allclose(st.center, med, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(st.scale, iqr, rtol=1e-3, atol=1e-9)


def test_transform_math_and_flag():
    x = np.array([[1.0, np.nan, -4.0],
                  [np.inf, 2.0, 0.0],
                  [3.0, 1.0, 8.0]])
    st = FeatureStage.fit(x[:2])  # stats solo de train
    u, r = st.transform(x)
    assert u.shape == (3, 4) and r.shape == (3,)          # d+1 flag
    # flag: filas con algún no-finito
    np.testing.assert_array_equal(r, [1.0, 1.0, 0.0])
    # columna 0: train=[1, inf] → mediana de no-finitos = 1
    # valor 3 → v = sign(3)*log1p(2) → escalado por train IQR
    v1 = np.log1p(3.0)
    expected = (v1 - st.center[0]) / st.scale[0]
    assert abs(u[2, 0] - min(max(expected, -CLIP), CLIP)) < 1e-6


def test_clip_bounds():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1e9, size=(500, 2))  # colas enormes
    st = FeatureStage.fit(x[:400])
    u, _ = st.transform(x)
    assert np.abs(u).max() <= CLIP + 1e-6


def test_imputation_uses_train_median():
    x_tr = np.array([[1.0], [3.0], [5.0]])
    st = FeatureStage.fit(x_tr)
    x_te = np.array([[np.nan]])
    u, r = st.transform(x_te)
    v = np.log1p(3.0)  # mediana de train, no de test
    np.testing.assert_allclose(u[0, 0], (v - st.center[0]) / st.scale[0])


def test_serialization_roundtrip():
    rng = np.random.default_rng(2)
    st = FeatureStage.fit(rng.normal(size=(200, 5)))
    st2 = FeatureStage.from_dict(st.to_dict())
    for k in ("impute_median", "center", "scale"):
        np.testing.assert_array_equal(getattr(st, k), getattr(st2, k))
    x = rng.normal(size=(10, 5))
    np.testing.assert_array_equal(st.transform(x)[0], st2.transform(x)[0])


def test_constant_column_no_div0():
    x = np.ones((100, 2))
    st = FeatureStage.fit(x)
    u, _ = st.transform(x)
    assert np.isfinite(u).all()
