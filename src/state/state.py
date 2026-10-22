"""VectrayX-One typed state — spec F0 §1.

State Compiler Contract: All raw telemetry is compiled to a
dense `State` object before touching the System One engine. The dimensions
(`d_num`) and `x_txt` keys are frozen by source in
`data/SCHEMA.md`; this module validates against that record.

Honesty NOTE: `label_*` only exists for training/evaluation.
In production the compiler never assigns them (None).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

TelemetrySource = Literal[
    "unsw_nb15_v1",       # split oficial training/testing-set
    "cicids2017_v1",      # MachineLearningCVE (79 cols; header verificado 2026-10-16)
    "ember2024_v1",       # features vectorizadas thrember — 2568 dims congeladas (SCHEMA §6)
]

SplitHint = Literal["train", "val", "test", "unassigned"]  # unassigned: the final split is decided by materialize.py

MAX_TXT_LEN = 256  # spec F0 §1: text fields truncated to 256 chars

# Source log: dims and frozen keys in data/SCHEMA.md v1.0.
# UNSW and CIC-MLCVE verified against real headers (2026-10-16); EMBER pending.
SOURCE_REGISTRY: dict[TelemetrySource, dict] = {
    "unsw_nb15_v1": {
        # 39 numeric of the real header: 45 cols = id + proto/service/state (x_txt)
        # + 39 numeric + attack_cat + label. (Corrected 2026-10-16: the "43"
        # original had id + 3 textual; build_canonical's assert caught it.)
        "d_num": 39,
        "x_txt_keys": ("proto", "service", "state"),
        "native_families": (
            "normal", "dos", "recon", "exploitation", "shellcode",
            "backdoor", "analysis", "fuzzing", "generic", "worms",
        ),
    },
    "cicids2017_v1": {
        # MachineLearningCVE: 78 features del CSV menos 'Fwd Header Length.1'
        # (columna duplicada — bug original del dataset). Sin IPs/Protocol/
        # timestamp: those live in the enhanced parquet (SCHEMA S1), which will have
        # your OWN log entry if used as primary source.
        "d_num": 77,
        "x_txt_keys": ("protocol",),  # None in MLCVE; the compiler leaves it at None
        "native_families": (
            "normal", "dos_ddos", "recon", "bruteforce", "webattack",
            "botnet", "infiltration", "exploitation",
        ),
    },
    "ember2024_v1": {
        # Feature version 3 (thrember 0.1.0, PEFeatureExtractor — verificado del
        # source code 2026-10-16): 12 features by default =
        # 7+256+256+34+177+224+1282+129+74+33+8+88 = 2568. Sin rama textual.
        #
        # Frozen Heads (decision S6, 2026-10-16): Noul is_malicious +
        # Choice choice_family = top-10 familias AV del train + "unknown".
        # Order = descending frequency of the COMPLETE TRAIN CENSUS
        # (3,120,000 lines; logs/ember_family_census.log, SCHEMA §3.4):
        # berbew 256,810 / cosmu 60,786 / wacatac 45,118 / sfone 44,216 /
        # upatre 43,954 / grandoreiro 40,728 / glupteba 39,814 / expiro
        # 35,434 / sinowal 34,560 / flystudio 34,262 = 635,682 malicious
        # (40.7%); corte 10→11 limpio (34,262 vs 17,974). Unknown absorbe
        # benign + malicious without family (188,682 = 12.1%) + the tail of
        # ~46,590 families. No head score (no GT severity). Lesson:
        # 4 week placeholder had 3 families out (eggnog/gepys/
        # urelas were gusts of Sept-Oct 2023) — census BEFORE materializing.
        "d_num": 2568,
        "x_txt_keys": (),
        "native_families": (
            "berbew", "cosmu", "wacatac", "sfone", "upatre",
            "grandoreiro", "glupteba", "expiro", "sinowal", "flystudio",
            "unknown",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class State:
    """Dense state object consumed by the decision engine (F0 §1)."""

    telemetry_source: TelemetrySource
    x_num: Sequence[float]            # longitud == SOURCE_REGISTRY[source]["d_num"]; ndarray float32 permitido (EMBER: 2568 dims × 3.85M filas)
    x_txt: dict[str, str | None]      # claves == x_txt_keys; valores ≤ MAX_TXT_LEN
    split_hint: SplitHint
    time: float | None = None         # epoch s, for time/window splits v0.2
    # --- training/assessment only; None in production ---
    label_family: str | None = None   # canonical family (SCHEMA §3) or "unknown"
    label_severity: int | None = None # rubrik 0-3 (SCHEMA §4)
    label_full: str | None = None     # native tag of the dataset (audit)
    label_binary: int | None = None   # binary Noul target (EMBER2024); None if derived from label_family

    def validate(self) -> None:
        """Fail-fast: any deviation from the schema is a pipeline bug."""
        reg = SOURCE_REGISTRY[self.telemetry_source]
        if reg["d_num"] is not None and len(self.x_num) != reg["d_num"]:
            raise ValueError(
                f"{self.telemetry_source}: x_num tiene {len(self.x_num)} dims, "
                f"el schema congela {reg['d_num']} (data/SCHEMA.md)"
            )
        expected_keys = set(reg["x_txt_keys"])
        if set(self.x_txt) != expected_keys:
            raise ValueError(
                f"{self.telemetry_source}: claves x_txt {sorted(self.x_txt)} != "
                f"congeladas {sorted(expected_keys)}"
            )
        for k, v in self.x_txt.items():
            if v is not None and len(v) > MAX_TXT_LEN:
                raise ValueError(f"x_txt[{k!r}] excede {MAX_TXT_LEN} chars — truncar en el compilador")


def truncate_txt(value: str | None) -> str | None:
    """Truncates a textual field to the contract boundary (it should never burst)."""
    if value is None:
        return None
    v = value.strip()
    return v[:MAX_TXT_LEN] if v else None
