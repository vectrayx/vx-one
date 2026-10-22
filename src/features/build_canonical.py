"""Dataset Compiler → `State` (F1).

Hard rule (data/SCHEMA.md §1): column→canonical mapping is a dictionary
EXPLICIT and the compiler FAILS if an expected column is missing — without
silent heuristics. If a dataset changes its header, we want the error
here, not a model trained with misaligned columns.

Today implemented: UNSW-NB15 (official split, list), CIC-IDS2017
MachineLearningCVE (chunk generator — header verified 2026-10-16) and
EMBER2024 (jsonl lines → State with thrember PEFeatureExtractor, 2568 dims
frozen — SCHEMA §3.4; heads S6: Noul is_malicious + Choice top-10).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from state.state import SplitHint, State, SOURCE_REGISTRY, truncate_txt

# ---------------------------------------------------------------------------
# UNSW-NB15 — mapping verified against real header (2026-10-16).
# 45 columns: id + proto/service/state (→ x_txt) + 39 numeric + attack_cat + label. UTF-8 with BOM
# (official files start with ﻿ → encoding="utf-8-sig").
# ---------------------------------------------------------------------------

# Actual column → canonical index within x_num (FROZEN order = order of
# real CSV header, verified 2026-10-16; there are 39 numbers).
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

# attack_cat native → canonical family (SCHEMA §3.2 — NATIVE policy,
# no collapse; approximate families only for cross-dataset experiments).
UNSW_FAMILY_MAP: dict[str, str] = {
    "Normal": "normal",
    "DoS": "dos",
    "Reconnaissance": "recon",
    "Exploits": "exploitation",
    "Shellcode": "shellcode",
    # ⚠️ quirk UNSW: official training/testing-set CSV uses "Backdoor"
    # (SINGULAR — censo 2026-10-16: 1,746 train + 583 test = 2,329); la
    # dataset documentation says "Backdoors". We accept both spellings.
    "Backdoor": "backdoor",
    "Backdoors": "backdoor",
    "Analysis": "analysis",
    "Fuzzers": "fuzzing",
    "Generic": "generic",
    "Worms": "worms",
}

# Rubrik severity 0-3 (SCHEMA §4; limitation stated in spec §8.1).
UNSW_SEVERITY_MAP: dict[str, int] = {
    "normal": 0,
    "recon": 1, "fuzzing": 1, "generic": 1,
    "dos": 2, "analysis": 2,
    "exploitation": 3, "shellcode": 3, "backdoor": 3, "worms": 3,
}

# 39 numerical. Correction 2026-10-16: original count said 43 — added up
# id + 3 textual + 39 numeric (= no-label columns); this assert got it
# before materializing any parquet.
assert len(UNSW_FEATURE_COLUMNS) == SOURCE_REGISTRY["unsw_nb15_v1"]["d_num"] == 39


def compile_unsw_csv(
    path: Path, split_hint: SplitHint
) -> list[State]:
    """Compiles one of the official UNSW-NB15 CSVs to State objects."""
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
                    # empty attack_cat on benign rows of some versions
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
        # Honest report: dataset has known rare NaNs/strings
        # (e.g. non-numeric values ​​in numeric columns in the full version)
        print(f"[warn] {path.name}: {len(errors)} filas descartadas; ej: {errors[:3]}")
    return states


# ---------------------------------------------------------------------------
# CIC-IDS2017 (MachineLearningCVE) — mapeo verificado contra header real
# (2026-10-16, mirror c01dsnap/CIC-IDS2017). 79 columnas: 78 features + Label.
# Names with spaces → strip. DUPLICATE 'Fwd Header Length' column in the
# Original CSV (pandas renames the 2nd one as 'Fwd Header Length.1' → discarded).
# No source IPs/ports or timestamp (Destination Port only) — that's why
# author's parquet enhanced is the PRIMARY source (SCHEMA S1); this version
# It is used for comparability with literature.
# ---------------------------------------------------------------------------

# EXACT names after strip() of the real header (frozen order):
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
# (note: duplicate 'Fwd Header Length' from CSV is discarded — pandas '.1')

CIC_FAMILY_MAP: dict[str, str] = {
    # EXACT graphy verified by census (2026-10-16, SCHEMA §3.1): 15 labels.
    # _norm_cic_label() normalizes box, dashes and the U+FFFD mojibake before the
    # lookup, so a single readable key covers the dataset variants.
    "BENIGN": "normal",
    "DDoS": "dos_ddos",
    "DoS Hulk": "dos_ddos",
    "DoS GoldenEye": "dos_ddos",
    "DoS slowloris": "dos_ddos",   # Lowercase 's' in the actual CSV
    "DoS Slowhttptest": "dos_ddos",
    "PortScan": "recon",
    "FTP-Patator": "bruteforce",
    "SSH-Patator": "bruteforce",
    "Web Attack – Brute Force": "webattack",  # in the actual CSV: U+FFFD where the script goes (mojibake); _norm equals it
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

# Config choice_ddos_vector (spec F0 §3.1) — exact census spelling.
# ⚠️ The 'DDoS' label (LOIT, 128,027 streams) lives in the FRIDAY archive,
# NO on Wednesday: as F0, SYN_Flood would be left with 0 samples if the
# subset is limited to Wednesday. Open decision (SCHEMA §3.3): extend the
# subset a Wednesday+Friday-DDoS. No bloquea F1.
CIC_DDoS_VECTOR_MAP: dict[str, str] = {
    "BENIGN": "Normal",
    "DDoS": "SYN_Flood",          # LOIT — only on Friday-...-DDos
    "DoS Hulk": "HTTP_DoS",
    "DoS GoldenEye": "HTTP_DoS",
    "DoS slowloris": "Slowloris",
    "DoS Slowhttptest": "Slowloris",
}

# d_num CIC = 77 features (78 from the CSV minus the duplicate 'Fwd Header Length.1')


def _norm_cic_label(s: str) -> str:
    """Normalizes `Label` tags for the lookup: spaces, dashes, box.

Verified by hexdump (2026-10-16): Thursday's actual CSV brings the
Web Attack tags with U+FFFD (replacement character — mojibake del
official dataset; bytes ef bf bd; the file is valid UTF-8). Others
versions use en-dash U+2013, U+2014 or dash. All three map to '-'
before the lookup; the rest of the inconsistencies ('DoS GoldenEye' vs
case variants) covers it casefold.
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
    """Probe encoding (utf-8 first; fallback latin-1 never fails)."""
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
    """Compile a CIC-IDS2017 MachineLearningCVE CSV → State GENERATOR.

Lazy by chunks: the files reach 215 MB (~440K streams) and the dataset
full ~2.8M — materializing everything in RAM would be a design error. Who
consume should write shards to disk incrementally (F1 script).
Extra CSV columns are ignored (selection by name); columns of
missing schema burst (fail-fast, no heuristics).
    """
    import numpy as np
    import pandas as pd

    fm = (
        CIC_FAMILY_MAP_NORM
        if family_map is None
        else {_norm_cic_label(k): v for k, v in family_map.items()}
    )
    dup = "Fwd Header Length.1"  # Original CSV bug: duplicate column
    cols = list(CIC_FEATURE_COLUMNS)

    reader = pd.read_csv(
        path,
        skipinitialspace=True,
        encoding=_cic_encoding(path),
        chunksize=chunksize,
    )

    n_rows = 0
    n_bad = 0  # rows with NaN/inf in features (known: Flow Bytes/s with duration 0)
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
                x_txt={"protocol": None},  # MLCVE does not come with Protocol; only enhanced parquet (SCHEMA S1)
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
# Each jsonl line brings metadata (label, family, ...) + the raw PE features;
# thrember PEFeatureExtractor vectorizes them to float32[2568]. Heads S6:
#   * Noul  `is_malicious`   ← label_binary (0/1, balance 50/50 exacto)
#   * Choice `choice_family` ← top-10 familias AV del train + "unknown"
# (Unknown absorbs benign, malicious without family and tail of ~1,480
# families); the raw family remains in label_full for auditing.
# * No head Score: there is no ground truth of severity in EMBER (S6).
# ---------------------------------------------------------------------------

EMBER_D_NUM = SOURCE_REGISTRY["ember2024_v1"]["d_num"]  # 2568
EMBER_TOP_FAMILIES = SOURCE_REGISTRY["ember2024_v1"]["native_families"][:-1]
assert len(EMBER_TOP_FAMILIES) == 10, (
    "S6: Choice choice_family congela top-10 familias + 'unknown'; el registro "
    f"trae {len(EMBER_TOP_FAMILIES)} (SCHEMA §3.4)"
)

_EMBER_EXTRACTOR = None  # lazy singleton — see _ember_extractor()


def _ember_extractor():
    """Instance PEFeatureExtractor once per PROCESS (the pool workers
of materialize.py are separate processes: each one imports thrember a
time). Lazy to not drag signify/thrember into network-only imports."""
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
    """A jsonl line from EMBER2024 → State (rules S6, SCHEMA §3.4).

benign → label_family="unknown", label_full="benign"
malicious + family top-10 → (family, family) [Choice]
malicious + tail family → ("unknown", raw_family) [Unknown absorbs]
malicious without family (null) → ("unknown", "malicious")
label_severity ALWAYS None (EMBER does not bring severity GT — S6).
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
            # Queue of ~1,480 families: Unknown absorbs it and the crude remains in
            # label_full (audit + possible re-mapping of top-10 in v0.2).
            label_family, label_full = "unknown", family
    else:
        label_family, label_full = "unknown", "malicious"

    x_num = _ember_extractor().process_raw_features(rec)
    st = State(
        telemetry_source="ember2024_v1",
        x_num=x_num,  # np.float32[2568] — ndarray satisface Sequence[float]
        x_txt={},     # rama textual desactivada (SCHEMA §2)
        split_hint=subset,  # train/val/test according to weekly carve S7
        label_family=label_family,
        label_severity=None,
        label_full=label_full,
        label_binary=label,  # objetivo Noul is_malicious
    )
    st.validate()
    return st
