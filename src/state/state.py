"""Estado tipado de VectrayX-One — spec F0 §1.

Contrato del compilador de estado: toda telemetría cruda se compila a un
objeto `State` denso antes de tocar el motor System One. Las dimensiones
(`d_num`) y las claves de `x_txt` están congeladas por fuente en
`data/SCHEMA.md`; este módulo valida contra ese registro.

NOTA de honestidad: `label_*` solo existe para entrenamiento/evaluación.
En producción el compilador nunca los asigna (None).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

TelemetrySource = Literal[
    "unsw_nb15_v1",       # split oficial training/testing-set
    "cicids2017_v1",      # MachineLearningCVE (79 cols; header verificado 2026-10-16)
    "ember2024_v1",       # features vectorizadas thrember — 2568 dims congeladas (SCHEMA §6)
]

SplitHint = Literal["train", "val", "test", "unassigned"]  # unassigned: el split final lo decide materialize.py

MAX_TXT_LEN = 256  # spec F0 §1: campos de texto truncados a 256 chars

# Registro de fuentes: dims y claves congeladas en data/SCHEMA.md v1.0.
# UNSW y CIC-MLCVE verificados contra headers reales (2026-10-16); EMBER pendiente.
SOURCE_REGISTRY: dict[TelemetrySource, dict] = {
    "unsw_nb15_v1": {
        # 39 numéricas del header real: 45 cols = id + proto/service/state (x_txt)
        # + 39 numéricas + attack_cat + label. (Corregido 2026-10-16: el "43"
        # original contaba id + 3 textuales; lo pilló el assert de build_canonical.)
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
        # timestamp: esos viven en el parquet enhanced (SCHEMA S1), que tendrá
        # su PROPIA entrada de registro si se usa como fuente primaria.
        "d_num": 77,
        "x_txt_keys": ("protocol",),  # None en MLCVE; el compilador lo deja en None
        "native_families": (
            "normal", "dos_ddos", "recon", "bruteforce", "webattack",
            "botnet", "infiltration", "exploitation",
        ),
    },
    "ember2024_v1": {
        # Feature version 3 (thrember 0.1.0, PEFeatureExtractor — verificado del
        # código fuente 2026-10-16): 12 features por defecto =
        # 7+256+256+34+177+224+1282+129+74+33+8+88 = 2568. Sin rama textual.
        #
        # Heads congelados (decisión S6, 2026-10-16): Noul is_malicious +
        # Choice choice_family = top-10 familias AV del train + "unknown".
        # Orden = frecuencia descendente del CENSO DEL TRAIN COMPLETO
        # (3,120,000 líneas; logs/ember_family_census.log, SCHEMA §3.4):
        # berbew 256,810 / cosmu 60,786 / wacatac 45,118 / sfone 44,216 /
        # upatre 43,954 / grandoreiro 40,728 / glupteba 39,814 / expiro
        # 35,434 / sinowal 34,560 / flystudio 34,262 = 635,682 malicious
        # (40.7%); corte 10→11 limpio (34,262 vs 17,974). Unknown absorbe
        # benignos + malicious sin familia (188,682 = 12.1%) + la cola de
        # ~46,590 familias. Sin head Score (no hay GT severidad). Lección:
        # el placeholder de 4 semanas tuvo 3 familias fuera (eggnog/gepys/
        # urelas eran ráfagas de sept-oct 2023) — censo ANTES de materializar.
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
    """Objeto de estado denso que consume el motor de decisión (F0 §1)."""

    telemetry_source: TelemetrySource
    x_num: Sequence[float]            # longitud == SOURCE_REGISTRY[source]["d_num"]; ndarray float32 permitido (EMBER: 2568 dims × 3.85M filas)
    x_txt: dict[str, str | None]      # claves == x_txt_keys; valores ≤ MAX_TXT_LEN
    split_hint: SplitHint
    time: float | None = None         # epoch s, para splits por tiempo / ventana v0.2
    # --- solo entrenamiento/evaluación; None en producción ---
    label_family: str | None = None   # familia canónica (SCHEMA §3) o "unknown"
    label_severity: int | None = None # rubrik 0-3 (SCHEMA §4)
    label_full: str | None = None     # etiqueta nativa del dataset (auditoría)
    label_binary: int | None = None   # objetivo Noul binario (EMBER2024); None si se deriva de label_family

    def validate(self) -> None:
        """Fail-fast: cualquier desviación del schema es un bug del pipeline."""
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
    """Trunca un campo textual al límite del contrato (nunca debe reventar)."""
    if value is None:
        return None
    v = value.strip()
    return v[:MAX_TXT_LEN] if v else None
