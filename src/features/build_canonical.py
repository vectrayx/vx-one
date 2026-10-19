"""Compilador dataset → `State` (F1).

Regla dura (data/SCHEMA.md §1): el mapeo columna→canónico es un diccionario
EXPLÍCITO y el compilador FALLA si falta una columna esperada — sin
heurísticas silenciosas. Si un dataset cambia de header, queremos el error
aquí, no un modelo entrenado con columnas desalineadas.

Hoy implementado: UNSW-NB15 (split oficial, lista), CIC-IDS2017
MachineLearningCVE (generador por chunks — header verificado 2026-10-16) y
EMBER2024 (líneas jsonl → State con thrember PEFeatureExtractor, 2568 dims
congeladas — SCHEMA §3.4; heads S6: Noul is_malicious + Choice top-10).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from state.state import SplitHint, State, SOURCE_REGISTRY, truncate_txt

# ---------------------------------------------------------------------------
# UNSW-NB15 — mapeo verificado contra el header real (2026-10-16).
# 45 columnas: id + proto/service/state (→ x_txt) + 39 numéricas + attack_cat + label. UTF-8 con BOM
# (los archivos oficiales empiezan con ﻿ → encoding="utf-8-sig").
# ---------------------------------------------------------------------------

# Columna real → índice canónico dentro de x_num (orden CONGELADO = orden del
# header del CSV real, verificado 2026-10-16; son 39 numéricas).
UNSW_FEATURE_COLUMNS: tuple[str, ...] = (
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
)

# attack_cat nativo → familia canónica (SCHEMA §3.2 — política NATIVA,
# sin colapso; familias aproximadas solo para experimentos cross-dataset).
UNSW_FAMILY_MAP: dict[str, str] = {
    "Normal": "normal",
    "DoS": "dos",
    "Reconnaissance": "recon",
    "Exploits": "exploitation",
    "Shellcode": "shellcode",
    # ⚠️ quirk UNSW: el CSV oficial training/testing-set usa "Backdoor"
    # (SINGULAR — censo 2026-10-16: 1,746 train + 583 test = 2,329); la
    # documentación del dataset dice "Backdoors". Aceptamos ambas grafías.
    "Backdoor": "backdoor",
    "Backdoors": "backdoor",
    "Analysis": "analysis",
    "Fuzzers": "fuzzing",
    "Generic": "generic",
    "Worms": "worms",
}

# Rubrik de severidad 0-3 (SCHEMA §4; limitación declarada en spec §8.1).
UNSW_SEVERITY_MAP: dict[str, int] = {
    "normal": 0,
    "recon": 1, "fuzzing": 1, "generic": 1,
    "dos": 2, "analysis": 2,
    "exploitation": 3, "shellcode": 3, "backdoor": 3, "worms": 3,
}

# 39 numéricas. Corrección 2026-10-16: el conteo original decía 43 — sumaba
# id + 3 textuales + 39 numéricas (= columnas no-label); este assert lo pilló
# antes de materializar cualquier parquet.
assert len(UNSW_FEATURE_COLUMNS) == SOURCE_REGISTRY["unsw_nb15_v1"]["d_num"] == 39


def compile_unsw_csv(
    path: Path, split_hint: SplitHint
) -> list[State]:
    """Compila uno de los CSV oficiales de UNSW-NB15 a objetos State."""
    n_expected = len(UNSW_FEATURE_COLUMNS)
    states: list[State] = []
    errors: list[str] = []

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [
            c for c in (*UNSW_FEATURE_COLUMNS, "proto", "service", "state",
                        "attack_cat", "label")
            if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(
                f"{path.name}: faltan columnas esperadas del schema: {missing}. "
                "El header del dataset cambió — actualizar data/SCHEMA.md antes de continuar."
            )

        for i, row in enumerate(reader):
            try:
                x_num = tuple(float(row[c]) for c in UNSW_FEATURE_COLUMNS)
                attack_cat = (row.get("attack_cat") or "").strip()
                family = UNSW_FAMILY_MAP.get(attack_cat)
                if family is None:
                    # attack_cat vacío en filas benignas de algunas versiones
                    family = "normal" if attack_cat == "" else "unknown"
                states.append(
                    State(
                        telemetry_source="unsw_nb15_v1",
                        x_num=x_num,
                        x_txt={
                            "proto": truncate_txt(row.get("proto")),
                            "service": truncate_txt(row.get("service")),
                            "state": truncate_txt(row.get("state")),
                        },
                        split_hint=split_hint,
                        label_family=family,
                        label_severity=UNSW_SEVERITY_MAP.get(family),
                        label_full=attack_cat or "Normal",
                    )
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"fila {reader.line_num}: {exc}")
                if len(errors) > 100:
                    raise ValueError(
                        f"{path.name}: >100 filas con errores de parseo; "
                        f"primero: {errors[:3]}"
                    ) from exc

    if errors:
        # Reporte honesto: el dataset tiene NaNs/strings raros conocidos
        # (p.ej. valores no numéricos en columnas numéricas de la versión completa)
        print(f"[warn] {path.name}: {len(errors)} filas descartadas; ej: {errors[:3]}")
    return states


# ---------------------------------------------------------------------------
# CIC-IDS2017 (MachineLearningCVE) — mapeo verificado contra header real
# (2026-10-16, mirror c01dsnap/CIC-IDS2017). 79 columnas: 78 features + Label.
# Nombres con espacios → strip. Columna 'Fwd Header Length' DUPLICADA en el
# CSV original (pandas renombra la 2ª como 'Fwd Header Length.1' → descartada).
# Sin IPs/puertos origen ni timestamp (solo Destination Port) — por eso el
# parquet enhanced del autor es la fuente PRIMARIA (SCHEMA S1); esta versión
# se usa para comparabilidad con literatura.
# ---------------------------------------------------------------------------

# Nombres EXACTOS tras strip() del header real (orden congelado):
CIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
)
# (nota: 'Fwd Header Length' duplicada del CSV se descarta — pandas '.1')

CIC_FAMILY_MAP: dict[str, str] = {
    # Graffía EXACTA verificada por censo (2026-10-16, SCHEMA §3.1): 15 etiquetas.
    # _norm_cic_label() normaliza caja, guiones y el U+FFFD mojibake antes del
    # lookup, así una sola clave legible cubre las variantes del dataset.
    "BENIGN": "normal",
    "DDoS": "dos_ddos",
    "DoS Hulk": "dos_ddos",
    "DoS GoldenEye": "dos_ddos",
    "DoS slowloris": "dos_ddos",   # 's' minúscula en el CSV real
    "DoS Slowhttptest": "dos_ddos",
    "PortScan": "recon",
    "FTP-Patator": "bruteforce",
    "SSH-Patator": "bruteforce",
    "Web Attack – Brute Force": "webattack",  # en el CSV real: U+FFFD donde va el guion (mojibake); _norm lo iguala
    "Web Attack – XSS": "webattack",
    "Web Attack – Sql Injection": "webattack",
    "Bot": "botnet",
    "Infiltration": "infiltration",
    "Heartbleed": "exploitation",
}

CIC_SEVERITY_MAP: dict[str, int] = {
    "normal": 0,
    "recon": 1,
    "dos_ddos": 2, "bruteforce": 2, "webattack": 2,
    "botnet": 3, "infiltration": 3, "exploitation": 3,
}

# Config choice_ddos_vector (spec F0 §3.1) — graffía exacta del censo.
# ⚠️ El label 'DDoS' (LOIT, 128,027 flujos) vive en el archivo del VIERNES,
# NO el miércoles: tal cual F0, SYN_Flood quedaría con 0 muestras si el
# subset se limita a miércoles. Decisión abierta (SCHEMA §3.3): extender el
# subset a Wednesday+Friday-DDoS. No bloquea F1.
CIC_DDoS_VECTOR_MAP: dict[str, str] = {
    "BENIGN": "Normal",
    "DDoS": "SYN_Flood",          # LOIT — solo en Friday-...-DDos
    "DoS Hulk": "HTTP_DoS",
    "DoS GoldenEye": "HTTP_DoS",
    "DoS slowloris": "Slowloris",
    "DoS Slowhttptest": "Slowloris",
}

# d_num CIC = 77 features (78 del CSV menos la duplicada 'Fwd Header Length.1')


def _norm_cic_label(s: str) -> str:
    """Normaliza etiquetas `Label` para el lookup: espacios, guiones, caja.

    Verificado por hexdump (2026-10-16): el CSV real del jueves trae las
    etiquetas Web Attack con U+FFFD (carácter de reemplazo — mojibake del
    dataset oficial; bytes ef bf bd; el archivo es UTF-8 válido). Otras
    versiones usan en-dash U+2013, U+2014 o guion. Los tres se mapean a '-'
    antes del lookup; el resto de inconsistencias ('DoS GoldenEye' vs
    variantes de caja) lo cubre casefold.
    """
    return (
        s.strip().casefold()
        .replace("�", "-")
        .replace("–", "-")
        .replace("—", "-")
    )


CIC_FAMILY_MAP_NORM: dict[str, str] = {
    _norm_cic_label(k): v for k, v in CIC_FAMILY_MAP.items()
}
CIC_DDoS_VECTOR_MAP_NORM: dict[str, str] = {
    _norm_cic_label(k): v for k, v in CIC_DDoS_VECTOR_MAP.items()
}

assert len(CIC_FEATURE_COLUMNS) == SOURCE_REGISTRY["cicids2017_v1"]["d_num"] == 77


def _cic_encoding(path: Path) -> str:
    """Sonda de encoding (utf-8 primero; latin-1 de fallback nunca falla)."""
    import pandas as pd

    for enc in ("utf-8", "latin-1"):
        try:
            pd.read_csv(path, nrows=1, skipinitialspace=True, encoding=enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def iter_cicids_states(
    path: Path,
    split_hint: SplitHint,
    family_map: dict[str, str] | None = None,
    chunksize: int = 100_000,
):
    """Compila un CSV MachineLearningCVE de CIC-IDS2017 → GENERADOR de State.

    Lazy por chunks: los archivos llegan a 215 MB (~440K flujos) y el dataset
    completo ~2.8M — materializar todo en RAM sería un error de diseño. Quien
    consuma debe escribir shards a disco incrementalmente (script de F1).
    Columnas extra del CSV se ignoran (selección por nombre); columnas del
    schema faltantes revientan (fail-fast, sin heurísticas).
    """
    import numpy as np
    import pandas as pd

    fm = (
        CIC_FAMILY_MAP_NORM
        if family_map is None
        else {_norm_cic_label(k): v for k, v in family_map.items()}
    )
    dup = "Fwd Header Length.1"  # bug original del CSV: columna duplicada
    cols = list(CIC_FEATURE_COLUMNS)

    reader = pd.read_csv(
        path,
        skipinitialspace=True,
        encoding=_cic_encoding(path),
        chunksize=chunksize,
    )

    n_rows = 0
    n_bad = 0  # filas con NaN/inf en features (conocido: Flow Bytes/s con duración 0)
    unknown_labels: set[str] = set()
    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        if dup in chunk.columns:
            chunk = chunk.drop(columns=[dup])

        missing = [c for c in (*cols, "Label") if c not in chunk.columns]
        if missing:
            raise ValueError(
                f"{path.name}: faltan columnas del schema CIC: {missing}. "
                "El header cambió — actualizar data/SCHEMA.md antes de continuar."
            )

        arr = (
            chunk[cols].apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype="float64")
        )
        n_bad += int((~np.isfinite(arr)).any(axis=1).sum())
        labels = chunk["Label"].astype(str).tolist()

        for i, label in enumerate(labels):
            family = fm.get(_norm_cic_label(label))
            if family is None:
                unknown_labels.add(label.strip())
                family = "unknown"  # drift visible, no colapso silencioso
            yield State(
                telemetry_source="cicids2017_v1",
                x_num=tuple(arr[i]),  # np.float64 ⊂ float
                x_txt={"protocol": None},  # MLCVE no trae Protocol; solo el parquet enhanced (SCHEMA S1)
                split_hint=split_hint,
                label_family=family,
                label_severity=CIC_SEVERITY_MAP.get(family),
                label_full=label.strip() or None,
            )
            n_rows += 1

    if unknown_labels:
        print(f"[warn] {path.name}: etiquetas sin mapeo → 'unknown': {sorted(unknown_labels)}")
    if n_bad:
        print(
            f"[warn] {path.name}: {n_bad}/{n_rows} filas con NaN/inf en features "
            "(esperado: Flow Bytes/s y Flow Packets/s con duración 0; la "
            "imputación se ajusta SOLO en train — SCHEMA §1)"
        )




# ---------------------------------------------------------------------------
# EMBER2024 (Win32) — compilador jsonl → State (heads S6, SCHEMA §3.4)
#
# Cada línea jsonl trae metadata (label, family, …) + las features PE crudas;
# thrember PEFeatureExtractor las vectoriza a float32[2568]. Heads S6:
#   * Noul  `is_malicious`   ← label_binary (0/1, balance 50/50 exacto)
#   * Choice `choice_family` ← top-10 familias AV del train + "unknown"
#     (Unknown absorbe benignos, malicious sin familia y la cola de ~1,480
#     familias); la familia cruda queda en label_full para auditoría.
#   * Sin head Score: no hay ground truth de severidad en EMBER (S6).
# ---------------------------------------------------------------------------

EMBER_D_NUM = SOURCE_REGISTRY["ember2024_v1"]["d_num"]  # 2568
EMBER_TOP_FAMILIES = SOURCE_REGISTRY["ember2024_v1"]["native_families"][:-1]
assert len(EMBER_TOP_FAMILIES) == 10, (
    "S6: Choice choice_family congela top-10 familias + 'unknown'; el registro "
    f"trae {len(EMBER_TOP_FAMILIES)} (SCHEMA §3.4)"
)

_EMBER_EXTRACTOR = None  # singleton perezoso — ver _ember_extractor()


def _ember_extractor():
    """Instancia PEFeatureExtractor una vez por PROCESO (los workers del pool
    de materialize.py son procesos separados: cada uno importa thrember una
    vez). Lazy para no arrastrar signify/thrember en imports de solo-red."""
    global _EMBER_EXTRACTOR
    if _EMBER_EXTRACTOR is None:
        from thrember import PEFeatureExtractor

        ext = PEFeatureExtractor()
        if ext.dim != EMBER_D_NUM:
            raise AssertionError(
                f"thrember dim={ext.dim} != congelado {EMBER_D_NUM} — revisar SCHEMA §6"
            )
        _EMBER_EXTRACTOR = ext
    return _EMBER_EXTRACTOR


def compile_ember_line(line: str, subset: SplitHint, week_idx: int) -> State:
    """Una línea jsonl de EMBER2024 → State (reglas S6, SCHEMA §3.4).

    benign                        → label_family="unknown", label_full="benign"
    malicious + familia top-10    → (familia, familia)          [Choice]
    malicious + familia de cola   → ("unknown", familia_cruda)  [Unknown absorbe]
    malicious sin familia (null)  → ("unknown", "malicious")
    label_severity SIEMPRE None (EMBER no trae GT de severidad — S6).
    """
    rec = json.loads(line)
    label = rec.get("label")
    if label not in (0, 1):
        raise ValueError(f"EMBER: label fuera de {{0,1}}: {label!r}")
    family = rec.get("family")
    if label == 0:
        label_family, label_full = "unknown", "benign"
    elif isinstance(family, str) and family:
        if family in EMBER_TOP_FAMILIES:
            label_family, label_full = family, family
        else:
            # Cola de ~1,480 familias: Unknown la absorbe y la cruda queda en
            # label_full (auditoría + posible re-mapeo del top-10 en v0.2).
            label_family, label_full = "unknown", family
    else:
        label_family, label_full = "unknown", "malicious"

    x_num = _ember_extractor().process_raw_features(rec)
    st = State(
        telemetry_source="ember2024_v1",
        x_num=x_num,  # np.float32[2568] — ndarray satisface Sequence[float]
        x_txt={},     # rama textual desactivada (SCHEMA §2)
        split_hint=subset,  # train/val/test según el carve semanal S7
        label_family=label_family,
        label_severity=None,
        label_full=label_full,
        label_binary=label,  # objetivo Noul is_malicious
    )
    st.validate()
    return st
