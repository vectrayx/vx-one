"""Materializa los datasets crudos a parquet canónico (F1).

Consume los compiladores de `build_canonical.py` y produce:

  data/processed/unsw_nb15/unsw_nb15.parquet
      Split oficial: training-set → train+val (estratificado 85/15 por familia,
      SEED fija), testing-set → test. Columna `split`.

  data/processed/cicids2017/cicids2017.parquet
      2,830,743 flujos de los 8 CSVs MachineLearningCVE, en orden de día y de
      fila original. Dos columnas de split (SCHEMA §5, ambas se reportan en E1):
        * split_strat — estratificado 70/15/15 por familia (SEED fija)
        * split_time  — corte posicional 70/15/15 sobre el orden
          Lun→Mar→Mié→Jue→Vie (el día es el proxy temporal: MLCVE no trae
          timestamps), preservando el orden de captura intra-día.

  data/processed/ember2024/part_{ii:02d}_{subset}.parquet + challenge.parquet
      64 semanas Win32 → un part por semana con split TEMPORAL congelado S7
      (train semanas 0-45, val 46-51, test 52-63) y challenge evasivo (6,315)
      como parquet único. x_num float32[2568] (mitad de disco que float64);
      cada worker escribe su part y borra el jsonl de esa semana (S7:
      re-descargable); los challenge jsonl NUNCA se borran.

Reglas duras:
  * Protección de test: si el destino existe, aborta sin --force.
  * x_num mantiene NaN/inf tal cual — la imputación se ajusta SOLO en train
    y se aplica al resto (SCHEMA §1, sin leakage). Este script no entrena.
  * Cada State pasa por validate() antes de escribirse (fail-fast del contrato).

Uso (desde la raíz del repo, venv con pandas/pyarrow/scikit-learn/thrember):
  .venv/bin/python -I src/features/materialize.py --dataset all    # unsw + cic
  .venv/bin/python -I src/features/materialize.py --dataset ember  # aparte: borra jsonl semanales (S7)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ al path (python -I)

from features.build_canonical import (  # noqa: E402
    CIC_FAMILY_MAP_NORM,
    _cic_encoding,
    _norm_cic_label,
    compile_ember_line,
    compile_unsw_csv,
    iter_cicids_states,
)
from state.state import SOURCE_REGISTRY, State  # noqa: E402

PAPER_ROOT = Path(__file__).resolve().parents[2]
DATA = PAPER_ROOT / "data"
SEED = 42  # F0 D7: seed fija documentada; cambiarla invalida los splits previos

# Dims congeladas del registro — única fuente de verdad (SCHEMA §6). Sin
# literales en este archivo: el assert de build_canonical cruza la tupla contra
# el registro y State.validate() cruza cada fila; aquí solo referenciamos.
UNSW_D_NUM = SOURCE_REGISTRY["unsw_nb15_v1"]["d_num"]   # 39
CIC_D_NUM = SOURCE_REGISTRY["cicids2017_v1"]["d_num"]   # 77
EMBER_D_NUM = SOURCE_REGISTRY["ember2024_v1"]["d_num"]  # 2568

# Orden temporal de los CSVs MLCVE (idx = posición en la semana; SCHEMA §5).
CIC_DAY_ORDER: dict[str, int] = {
    "Monday-WorkingHours.pcap_ISCX.csv": 0,
    "Tuesday-WorkingHours.pcap_ISCX.csv": 1,
    "Wednesday-workingHours.pcap_ISCX.csv": 2,
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": 3,
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": 4,
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": 5,
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": 6,
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": 7,
}

UNSW_SCHEMA = pa.schema([
    ("x_num", pa.list_(pa.float64(), UNSW_D_NUM)),
    ("x_txt_proto", pa.string()), ("x_txt_service", pa.string()),
    ("x_txt_state", pa.string()),
    ("split", pa.string()),
    ("label_family", pa.string()), ("label_full", pa.string()),
    ("label_severity", pa.int8()),
])

CIC_SCHEMA = pa.schema([
    ("x_num", pa.list_(pa.float64(), CIC_D_NUM)),
    ("x_txt_protocol", pa.string()),  # all-null en MLCVE (solo el enhanced trae Protocol)
    ("day", pa.string()), ("day_idx", pa.int8()),
    ("split_strat", pa.string()), ("split_time", pa.string()),
    ("label_family", pa.string()), ("label_full", pa.string()),
    ("label_severity", pa.int8()),
])

# EMBER2024: x_num float32 — 2568 dims × 3.85M filas ≈ 20 GB comprimidos;
# float64 los duplicaría. Split temporal por semana (S7), NO estratificado:
# estratificar mezclaría el futuro en train.
EMBER_SCHEMA = pa.schema([
    ("x_num", pa.list_(pa.float32(), EMBER_D_NUM)),
    ("label_binary", pa.int8()),    # objetivo Noul is_malicious (S6)
    ("label_family", pa.string()),  # top-10 o "unknown" (S6)
    ("label_full", pa.string()),    # familia cruda / "benign" / "malicious" (auditoría)
    ("subset", pa.string()),        # train/val/test — carve semanal S7 (SCHEMA §5)
    ("week_idx", pa.int16()),       # 0-51 train(0-45)/val(46-51), 52-63 test, 64 challenge
])
EMBER_EXPECTED_LINES = 60_000      # TODAS las semanas traen exactamente 60,000 líneas (censo 2026-10-16)
EMBER_CHALLENGE_EXPECTED = 6_315   # challenge evasivo, 100% malicious (censo 2026-10-16)
EMBER_BUFFER = 25_000              # filas por row-group (~256 MB de features en RAM por worker)
EMBER_WORKERS = 12                 # 12 × ~400 MB ≈ 5 GB < 7 GB disponibles (24 workers reventaba la RAM de 15 GB)


def _guard(out: Path, force: bool) -> None:
    if out.exists() and not force:
        raise SystemExit(
            f"[abort] {out} ya existe — el test se toca UNA vez; usa --force para regenerar"
        )


def _x_num_array(states: list[State], d_num: int) -> pa.Array:
    mat = np.asarray([s.x_num for s in states], dtype="float64")
    return pa.FixedSizeListArray.from_arrays(pa.array(mat.reshape(-1)), d_num)


def _stratified_two_step(y: np.ndarray, first_test: float, seed: int) -> np.ndarray:
    """Split 3-vías estratificado en dos pasos (sklearn no hace 3-vías directo).

    Devuelve array de 'train'/'val'/'test' alineado con y. Si la estratificación
    es imposible (clases con <2 miembros en un subconjunto), cae sin estratificar
    con warning — honesto y visible, nunca silencioso.
    """
    from sklearn.model_selection import train_test_split

    n = len(y)
    out = np.empty(n, dtype="object")
    idx = np.arange(n)
    try:
        a, rest = train_test_split(idx, test_size=first_test, stratify=y, random_state=seed)
    except ValueError as exc:
        print(f"[warn] estratificación imposible ({exc}); split NO estratificado")
        a, rest = train_test_split(idx, test_size=first_test, random_state=seed)
    try:
        b, c = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    except ValueError as exc:
        print(f"[warn] estratificación 2ª etapa imposible ({exc}); val/test NO estratificados")
        b, c = train_test_split(rest, test_size=0.5, random_state=seed)
    out[a] = "train"
    out[b] = "val"
    out[c] = "test"
    return out


# ---------------------------------------------------------------------------
# UNSW-NB15
# ---------------------------------------------------------------------------
def materialize_unsw(force: bool) -> None:
    out_dir = DATA / "processed" / "unsw_nb15"
    out = out_dir / "unsw_nb15.parquet"
    _guard(out, force)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] compilando training-set oficial (175,341)…")
    tr_states = compile_unsw_csv(
        DATA / "raw" / "unsw_nb15" / "UNSW_NB15_training-set.csv",
        split_hint="unassigned",
    )
    print("[2/3] compilando testing-set oficial (82,332)…")
    te_states = compile_unsw_csv(
        DATA / "raw" / "unsw_nb15" / "UNSW_NB15_testing-set.csv",
        split_hint="test",
    )

    # Conteo oficial del dataset (SCHEMA §5): si el compilador descartó filas
    # (parse errors) o el archivo cambió, abortamos — nunca un parquet corto.
    for name, sts, expected in (
        ("training-set", tr_states, 175_341),
        ("testing-set", te_states, 82_332),
    ):
        if len(sts) != expected:
            raise SystemExit(
                f"[abort] {name}: {len(sts):,} filas ≠ oficiales {expected:,} — "
                "el compilador descartó filas (ver warns arriba); parquet NO escrito"
            )

    for s in (*tr_states, *te_states):
        s.validate()

    # val estratificado 85/15 DENTRO del training-set (SCHEMA §5).
    fams = np.asarray([s.label_family for s in tr_states])
    splits = _stratified_two_step(fams, first_test=0.15, seed=SEED)
    # first_test=0.15 deja val+test juntos: renombramos 'test'→'val' — el test
    # real es el testing-set oficial, nunca se toca aquí.
    splits = np.where(splits == "test", "val", splits)

    print(f"[3/3] escribiendo {out} ({len(tr_states) + len(te_states)} filas)…")
    with pq.ParquetWriter(out, UNSW_SCHEMA, compression="zstd") as w:
        tbl = pa.Table.from_arrays([
            _x_num_array(tr_states, UNSW_D_NUM),
            pa.array([s.x_txt.get("proto") for s in tr_states], type=pa.string()),
            pa.array([s.x_txt.get("service") for s in tr_states], type=pa.string()),
            pa.array([s.x_txt.get("state") for s in tr_states], type=pa.string()),
            pa.array(list(splits), type=pa.string()),
            pa.array([s.label_family for s in tr_states], type=pa.string()),
            pa.array([s.label_full for s in tr_states], type=pa.string()),
            pa.array([s.label_severity for s in tr_states], type=pa.int8()),
        ], schema=UNSW_SCHEMA)
        w.write_table(tbl)

        te_split = np.array(["test"] * len(te_states), dtype="object")
        tbl = pa.Table.from_arrays([
            _x_num_array(te_states, UNSW_D_NUM),
            pa.array([s.x_txt.get("proto") for s in te_states], type=pa.string()),
            pa.array([s.x_txt.get("service") for s in te_states], type=pa.string()),
            pa.array([s.x_txt.get("state") for s in te_states], type=pa.string()),
            pa.array(list(te_split), type=pa.string()),
            pa.array([s.label_family for s in te_states], type=pa.string()),
            pa.array([s.label_full for s in te_states], type=pa.string()),
            pa.array([s.label_severity for s in te_states], type=pa.int8()),
        ], schema=UNSW_SCHEMA)
        w.write_table(tbl)

    df = pd.DataFrame({
        "family": [s.label_family for s in (*tr_states, *te_states)],
        "split": [*splits, *te_split],
    })
    print("\n[report] UNSW familias × split:")
    print(pd.crosstab(df.family, df.split))
    print(f"\n[ok] {out}")


# ---------------------------------------------------------------------------
# CIC-IDS2017 (MLCVE) — dos pasadas: labels para splits, luego stream de features
# ---------------------------------------------------------------------------
def _cic_files() -> list[Path]:
    d = DATA / "raw" / "cicids2017" / "canonical_mlcve"
    files = sorted(d.glob("*.csv"), key=lambda p: CIC_DAY_ORDER[p.name])
    if len(files) != len(CIC_DAY_ORDER):
        raise SystemExit(
            f"[abort] esperaba {len(CIC_DAY_ORDER)} CSVs en {d}, hay {len(files)}: "
            f"{[p.name for p in files]}"
        )
    return files


def _cic_families_pass1(files: list[Path]) -> list[str]:
    """Pasada 1: familia canónica por fila (solo columna Label — rápido)."""
    fams: list[str] = []
    for p in files:
        n0 = len(fams)
        for chunk in pd.read_csv(
            p, skipinitialspace=True, usecols=[78], dtype=str,
            encoding=_cic_encoding(p), chunksize=500_000,
        ):
            col = chunk.iloc[:, 0].astype(str).tolist()
            fams.extend(CIC_FAMILY_MAP_NORM.get(_norm_cic_label(lab), "unknown") for lab in col)
        print(f"  pass1 {p.name}: +{len(fams) - n0}")
    return fams


def materialize_cic(force: bool) -> None:
    out_dir = DATA / "processed" / "cicids2017"
    out = out_dir / "cicids2017.parquet"
    _guard(out, force)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _cic_files()

    print(f"[1/3] pasada 1: labels de {len(files)} archivos para asignar splits…")
    fams = _cic_families_pass1(files)
    y = np.asarray(fams)
    n_total = len(y)
    print(f"      total {n_total:,} filas (esperado: 2,830,743)")

    print("[2/3] asignando split_strat (estratificado) y split_time (día como proxy)…")
    split_strat = _stratified_two_step(y, first_test=0.30, seed=SEED)
    split_time = np.empty(n_total, dtype="object")
    b1, b2 = int(0.70 * n_total), int(0.85 * n_total)
    split_time[:b1] = "train"
    split_time[b1:b2] = "val"
    split_time[b2:] = "test"

    print(f"[3/3] pasada 2: compilando y escribiendo {out}…")
    g = 0
    with pq.ParquetWriter(out, CIC_SCHEMA, compression="zstd") as w:
        for p in files:
            day = p.stem
            day_idx = CIC_DAY_ORDER[p.name]
            buf: list[State] = []

            def _flush(buf: list[State], g0: int) -> int:
                n = len(buf)
                for s in buf:
                    s.validate()
                tbl = pa.Table.from_arrays([
                    _x_num_array(buf, CIC_D_NUM),
                    pa.array([None] * n, type=pa.string()),
                    pa.array([day] * n, type=pa.string()),
                    pa.array([day_idx] * n, type=pa.int8()),
                    pa.array(list(split_strat[g0:g0 + n]), type=pa.string()),
                    pa.array(list(split_time[g0:g0 + n]), type=pa.string()),
                    pa.array([s.label_family for s in buf], type=pa.string()),
                    pa.array([s.label_full for s in buf], type=pa.string()),
                    pa.array([s.label_severity for s in buf], type=pa.int8()),
                ], schema=CIC_SCHEMA)
                w.write_table(tbl)
                return n

            n_file = 0
            for st in iter_cicids_states(p, split_hint="unassigned"):
                buf.append(st)
                if len(buf) >= 250_000:
                    g += _flush(buf, g)
                    n_file += len(buf)
                    buf = []
            if buf:
                g += _flush(buf, g)
                n_file += len(buf)
            print(f"  {day}: {n_file:,} filas (global {g:,})")

    if g != n_total:
        raise SystemExit(
            f"[abort] filas escritas {g:,} != esperadas {n_total:,} — pass1/pass2 divergen; borrar {out}"
        )
    cs = Counter(split_strat)
    print(
        f"[check] split_strat: train {cs['train']:,} / val {cs['val']:,} / "
        f"test {cs['test']:,} (split_time posicional: {b1:,}/{b2 - b1:,}/{n_total - b2:,})"
    )

    df = pd.DataFrame({
        "family": fams,
        "split_strat": split_strat,
        "split_time": split_time,
    })
    print("\n[report] CIC familias × split_strat:")
    print(pd.crosstab(df.family, df.split_strat))
    print("\n[report] CIC familias × split_time:")
    print(pd.crosstab(df.family, df.split_time))
    print(f"\n[ok] {out} — {g:,} filas")


# ---------------------------------------------------------------------------
# EMBER2024 — pool de semanas; cada worker compila, escribe y limpia (S7)
# ---------------------------------------------------------------------------
def _ember_weeks(raw_dir: Path) -> list[tuple[list[Path], str, int]]:
    """(rutas, subset, week_idx) por semana — carve temporal S7 (SCHEMA §5):
    train = semanas 0-45, val = 46-51 (últimas 6 del año), test = 52-63.
    El challenge (evasivo) va aparte como parquet único (materialize_ember)."""
    train = sorted(raw_dir.glob("*_Win32_train.jsonl"))
    test = sorted(raw_dir.glob("*_Win32_test.jsonl"))
    # S7 borra los jsonl al materializar: 0/0 es el estado NORMAL del resume
    # (los parts mandan). Un conteo parcial sí es anomalía — la trampa real
    # (part ausente + jsonl ausente) la pilla materialize_ember por semana.
    if (len(train), len(test)) not in ((0, 0), (52, 12)):
        raise SystemExit(
            f"[abort] EMBER: esperaba 52 train + 12 test jsonl en {raw_dir} "
            f"(o 0/0 si todo está materializado), hay {len(train)}/{len(test)}"
        )
    weeks = [([p], "train" if i < 46 else "val", i) for i, p in enumerate(train)]
    weeks += [([p], "test", 52 + i) for i, p in enumerate(test)]
    return weeks


def _ember_week_task(args: tuple) -> tuple[str, str, int, int, dict]:
    """Proceso hijo del pool: compila sus archivos → parquet → replace → unlink.

    La escritura vive AQUÍ a propósito: devolver 60K filas × 2568 float32 por
    IPC son ~615 MB por semana (× workers = reventaba la RAM de 15 GB). Cada
    worker lleva un ring-buffer de EMBER_BUFFER filas y escribe row-groups al
    mismo ParquetWriter (~400 MB pico por proceso). Falla ANTES de os.replace
    si el conteo no cuadra — nunca un part corto; el jsonl NO se borra (S7).
    """
    paths, subset, week_idx, out_str, delete_src, expected = args
    out = Path(out_str)
    tmp = out.with_name(out.name + ".tmp")

    buf = np.empty((EMBER_BUFFER, EMBER_D_NUM), dtype=np.float32)
    meta: list[tuple[int, str, str]] = []
    fam_counts: Counter = Counter()
    n_read = n_written = 0
    writer = pq.ParquetWriter(tmp, EMBER_SCHEMA, compression="zstd")

    def _flush() -> None:
        nonlocal n_written
        if not meta:
            return
        n = len(meta)
        # pa.array copia por defecto: el ring-buffer se reutiliza al volver
        writer.write_table(pa.Table.from_arrays([
            pa.FixedSizeListArray.from_arrays(
                pa.array(buf[:n].reshape(-1)), EMBER_D_NUM),
            pa.array([m[0] for m in meta], type=pa.int8()),
            pa.array([m[1] for m in meta], type=pa.string()),
            pa.array([m[2] for m in meta], type=pa.string()),
            pa.array([subset] * n, type=pa.string()),
            pa.array([week_idx] * n, type=pa.int16()),
        ], schema=EMBER_SCHEMA))
        n_written += n
        meta.clear()

    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                st = compile_ember_line(line, subset, week_idx)
                buf[len(meta)] = st.x_num
                meta.append((st.label_binary, st.label_family, st.label_full))
                fam_counts[st.label_family] += 1
                n_read += 1
                if len(meta) == EMBER_BUFFER:
                    _flush()
    _flush()
    if n_read != expected or n_written != n_read:
        raise ValueError(
            f"{out.name}: leídas {n_read:,} / escritas {n_written:,} ≠ {expected:,} "
            f"esperadas — NO se promueve el part ni se borra el jsonl (S7); "
            f"inspeccionar {tmp}"
        )
    writer.close()
    os.replace(tmp, out)  # atómico: ningún part se ve a medio escribir
    if delete_src:
        for path in paths:
            path.unlink()  # S7: jsonl re-descargable; challenge NUNCA (delete_src=False)
    return (out.name, subset, week_idx, n_written, dict(fam_counts))


def materialize_ember(force: bool) -> None:
    """Materializa EMBER2024 Win32: 64 semanas → parts + challenge.parquet.

    Resume (S7): el part existente manda — si el jsonl sobrevivió (crash entre
    replace y unlink) se completa el borrado; sin part y sin jsonl es abort
    (la fuente es re-descargable con el bootstrap). Verificación global de
    conteos por subset leyendo solo (subset, label_family) de los parquets.
    """
    raw_dir = DATA / "raw" / "ember2024"
    out_dir = DATA / "processed" / "ember2024"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple] = []
    skipped = 0
    for paths, subset, week_idx in _ember_weeks(raw_dir):
        out = out_dir / f"part_{week_idx:02d}_{subset}.parquet"
        have_src = all(p.exists() for p in paths)
        if not have_src and not out.exists():
            raise SystemExit(
                f"[abort] {out.name}: sin part y sin jsonl — re-descargar con "
                "scripts/f1_bootstrap.sh (S7: los jsonl se borran tras materializar)"
            )
        if have_src and out.exists() and not force:
            for p in paths:
                p.unlink()  # crash entre replace y unlink: completar el borrado S7
            print(f"  resume {out.name}: part completo → jsonl redundante eliminado")
        if not have_src or (out.exists() and not force):
            skipped += 1
            continue
        jobs.append((paths, subset, week_idx, str(out), True, EMBER_EXPECTED_LINES))

    chal_paths = sorted(raw_dir.glob("*_challenge_malicious.jsonl"))
    chal_out = out_dir / "challenge.parquet"
    if not chal_paths and not chal_out.exists():
        raise SystemExit(
            "[abort] EMBER: no hay challenge jsonl ni challenge.parquet — "
            "re-descargar (los challenge NUNCA se borran: son el test de estrés E2)"
        )
    if chal_paths and not (chal_out.exists() and not force):
        jobs.append((chal_paths, "test", 64, str(chal_out), False, EMBER_CHALLENGE_EXPECTED))
    else:
        skipped += 1

    print(
        f"[1/2] {len(jobs)} unidades por procesar ({skipped} ya materializadas), "
        f"{EMBER_WORKERS} workers × buffer {EMBER_BUFFER:,} filas…"
    )
    if jobs:
        with Pool(EMBER_WORKERS) as pool:
            for name, subset, week_idx, n, fams in pool.imap_unordered(_ember_week_task, jobs):
                top = ", ".join(
                    f"{k}={v:,}" for k, v in sorted(fams.items(), key=lambda kv: -kv[1])[:3]
                )
                print(f"  ✓ {name} [{subset} w{week_idx}] {n:,} filas — {top}")

    print("[2/2] verificación global (solo columnas subset+label_family)…")
    totals: Counter = Counter()
    fam_x_subset: Counter = Counter()
    files = sorted(out_dir.glob("part_*.parquet"))
    if chal_out.exists():
        files.append(chal_out)
    for f in files:
        agg = (
            pq.read_table(f, columns=["subset", "label_family"])
            .group_by(["subset", "label_family"])
            .aggregate([([], "count_all")])
        )
        # El nombre de la columna de conteo varía entre versiones de pyarrow
        # ("subset_count" / "count_all" / …); tomamos la ÚLTIMA por posición.
        count_col = agg.column(agg.num_columns - 1)
        for s, fam, c in zip(
            agg.column("subset").to_pylist(),
            agg.column("label_family").to_pylist(),
            count_col.to_pylist(),
        ):
            totals[s] += c
            fam_x_subset[(s, fam)] += c

    if len(files) != 65:
        raise SystemExit(
            f"[abort] EMBER: {len(files)} parquets en {out_dir}, esperaba 65 "
            "(64 parts + challenge)"
        )
    esperado = {"train": 2_760_000, "val": 360_000, "test": 726_315}  # test = 720,000 + 6,315 challenge
    ok = True
    for s, exp in esperado.items():
        got = totals.get(s, 0)
        good = got == exp
        ok = ok and good
        print(f"  {'✓' if good else '✗'} {s}: {got:,} (esperado {exp:,})")
    if not ok:
        raise SystemExit(f"[abort] conteos globales EMBER no cuadran — revisar {out_dir}")

    piv = (
        pd.DataFrame(
            [{"subset": s, "family": fam, "n": c} for (s, fam), c in sorted(fam_x_subset.items())]
        )
        .pivot_table(index="family", columns="subset", values="n", fill_value=0, aggfunc="sum")
        .astype(int)
    )
    piv["total"] = piv.sum(axis=1)
    print("\n[report] EMBER familias × subset (top 15 por total):")
    print(piv.sort_values("total", ascending=False).head(15).to_string())
    unk = sum(c for (s, fam), c in fam_x_subset.items() if fam == "unknown")
    tot = sum(totals.values())
    print(
        f"\n[check] P(Unknown) global = {unk / tot:.1%} ({unk:,}/{tot:,}) — se REPORTA "
        "como honestidad del Choice (S6), no como fallo: el detector es Noul "
        "is_malicious; Choice aporta atribución cuando la familia está en el top-10."
    )
    print(f"\n[ok] {len(files)} parquets en {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Materializa crudos → parquet canónico (F1)")
    ap.add_argument(
        "--dataset", choices=("unsw", "cic", "ember", "all"), required=True,
        help="'all' = unsw+cic; 'ember' va APARTE (borra los jsonl semanales — S7)",
    )
    ap.add_argument("--force", action="store_true",
                    help="sobreescribir el parquet destino (invalida test ya tocado)")
    args = ap.parse_args()
    if args.dataset in ("unsw", "all"):
        materialize_unsw(args.force)
    if args.dataset in ("cic", "all"):
        materialize_cic(args.force)
    if args.dataset == "ember":
        materialize_ember(args.force)


if __name__ == "__main__":
    main()
