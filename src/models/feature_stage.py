"""Feature stage φ (spec F2 §4.1): sanitización + normalización train-only.

Pipeline por columna j (stats SOLO de train, aplicado igual a train/val/test):
1. No-finitos: NaN/±inf → mediana de los finitos de train (fallback 0.0);
   flag de fila r_i = 1[la fila tenía algún no-finito].
2. Signed log1p: v = sign(x̃)·log1p(|x̃|).
3. Robust scale: u = (v − med_j) / max(IQR_j, 1e-8), stats sobre la columna
   ya transformada de train. Columna constante → v−med = 0 → u = 0.
4. Clip ±8.
5. Entrada del modelo: [u ; r] ∈ R^{d+1} (el flag es la dim extra).

Stats en float64, aplicación en float32. Serializado en el checkpoint:
impute_median f32[d], center f32[d], scale f32[d], clip 8.0.
"""

from __future__ import annotations

import numpy as np

CLIP = 8.0
IQR_EPS = 1e-8


class FeatureStage:
    """Ajusta (train) y aplica (todos) el pipeline φ. Stats en float64."""

    def __init__(
        self,
        impute_median: np.ndarray,
        center: np.ndarray,
        scale: np.ndarray,
        clip: float = CLIP,
    ) -> None:
        d = impute_median.shape[0]
        assert center.shape == (d,) and scale.shape == (d,)
        self.impute_median = np.asarray(impute_median, dtype=np.float32)
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.clip = float(clip)

    @classmethod
    def fit(cls, x_train: np.ndarray) -> "FeatureStage":
        """x_train: float64 [N, d] crudo (con NaN/inf tal cual)."""
        assert x_train.ndim == 2
        n, d = x_train.shape
        impute = np.empty(d, dtype=np.float64)
        center = np.empty(d, dtype=np.float64)
        scale = np.empty(d, dtype=np.float64)
        for j in range(d):
            col = x_train[:, j]
            finite = col[np.isfinite(col)]
            impute[j] = float(np.median(finite)) if finite.size else 0.0
            x̃ = np.where(np.isfinite(col), col, impute[j])
            v = np.sign(x̃) * np.log1p(np.abs(x̃))
            med = float(np.median(v))
            q75, q25 = np.percentile(v, [75, 25])
            iqr = q75 - q25
            center[j] = med
            scale[j] = max(iqr, IQR_EPS)
        return cls(impute, center, scale, CLIP)

    def transform(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x crudo [N, d] (float64 o float32) → (u float32 [N, d+1], None).

        Devuelve la matriz [u ; r] con el flag de fila como última columna.
        """
        x64 = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(x64)
        x̃ = np.where(finite, x64, self.impute_median.astype(np.float64))
        v = np.sign(x̃) * np.log1p(np.abs(x̃))
        u = (v - self.center.astype(np.float64)) / self.scale.astype(np.float64)
        u = np.clip(u, -self.clip, self.clip)
        r = (~finite).any(axis=1).astype(np.float32)
        out = np.empty((x64.shape[0], x64.shape[1] + 1), dtype=np.float32)
        out[:, :-1] = u.astype(np.float32)
        out[:, -1] = r
        return out, r

    # ── serialización ────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "impute_median": self.impute_median,
            "center": self.center,
            "scale": self.scale,
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStage":
        return cls(d["impute_median"], d["center"], d["scale"], d["clip"])
