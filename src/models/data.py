"""Datasets y loaders del motor VX-One (spec F2 §0.1, §6.2, D12).

- UNSW / CIC: tensores completos en RAM (UNSW 149K×40×4B ≈ 24 MB;
  CIC train 1.98M×78×4B ≈ 620 MB).
- EMBER: streaming por shards (46 train parquets de 60K filas; ~616 MB
  decodificados c/u): por epoch se permutan los shards (seed 42+epoch),
  shuffle intra-shard, batches con drop_last; prefetch de 1 shard en thread.
  Shuffle de dos niveles (shard + intra) — aproximación documentada (§6.2).

Candidate set por corrida (D12): familias nativas con ≥1 fila en TRAIN bajo
el esquema de la corrida, en el ORDEN CONGELADO del registro. Pesos ω
(§3.5) derivados de los conteos de train DE LA CORRIDA (LOAO los recalcula).

φ (feature stage) se ajusta SOLO con train:
- UNSW/CIC: train completo en RAM (float64).
- EMBER: submuestra uniforme de 60,000 filas de train (1,304 por shard,
  rng seed 42, una pasada por los 46 shards) — la SE de la mediana con
  n=60K es despreciable y evita 28 GB en RAM; decisión documentada.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from src.state.state import SOURCE_REGISTRY

from .feature_stage import FeatureStage
from .losses import class_weights
from .network import HeadSpec
from .vocab import TextVocab

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

REGISTRY = SOURCE_REGISTRY
ALIAS = {"unsw": "unsw_nb15_v1", "cic": "cicids2017_v1", "ember": "ember2024_v1"}


def _registry(source: str) -> dict:
    """Alias corto (unsw/cic/ember) o clave de registro → entrada del registro."""
    return REGISTRY[ALIAS.get(source, source)]

DDOS_DAYS = {
    "Wednesday-workingHours.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",
}
DDOS_FULL = {"BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"}
DDOS_CANDIDATES = ("normal", "tcp_flood", "http_dos", "slowloris")  # orden congelado F2 D8
_DDOS_MAP = {"BENIGN": "normal", "DDoS": "tcp_flood", "DoS Hulk": "http_dos",
             "DoS GoldenEye": "http_dos", "DoS slowloris": "slowloris",
             "DoS Slowhttptest": "slowloris"}

EMBER_STATS_ROWS_PER_SHARD = 1_304  # × 46 shards = 60,024 filas de stats


@dataclass
class Labels:
    y_bin: torch.Tensor        # int64 [N] 1[family != normal] (o label_binary en EMBER)
    y_fam: torch.Tensor        # int64 [N] índice en candidates
    y_sev: torch.Tensor | None # int64 [N] 0..3 (None en EMBER)
    ddos_mask: torch.Tensor | None  # bool [N] filas del subconjunto (solo CIC strat)
    y_ddos: torch.Tensor | None     # int64 [N] índice en DDOS_CANDIDATES


@dataclass
class RunData:
    """Todo lo que una corrida necesita, materializado y verificado."""
    source: str
    scale: str
    split_scheme: str          # "oficial" (UNSW) | "strat" | "time" (CIC) | "temporal" (EMBER)
    candidates: list[str]
    ddos_candidates: list[str] | None
    omega: torch.Tensor        # float32 [K]
    omega_ddos: torch.Tensor | None
    vocab: dict[str, TextVocab] | None
    feature_stage: FeatureStage
    heads: HeadSpec
    train: torch.Tensor        # u_r float32 [Ntr, d+1]
    train_txt: dict[str, torch.Tensor] | None
    train_labels: Labels
    val: torch.Tensor
    val_txt: dict[str, torch.Tensor] | None
    val_labels: Labels
    test: torch.Tensor | None  # None en EMBER (streaming, no se materializa)
    test_txt: dict[str, torch.Tensor] | None
    test_labels: Labels | None
    val_fam_names: np.ndarray | None = None   # familia nativa por fila (E2; -1 = fuera de candidatos)
    test_fam_names: np.ndarray | None = None
    ember_train_shards: list[Path] | None = None
    ember_val_shards: list[Path] | None = None
    ember_test_shards: list[Path] | None = None
    meta: dict = field(default_factory=dict)


# ── utilidades de lectura ──────────────────────────────────────────────
def _read_x_num(path: Path) -> np.ndarray:
    tbl = pq.read_table(path, columns=["x_num"])
    arr = tbl.column("x_num").combine_chunks()
    flat = arr.flatten().to_numpy(zero_copy_only=False)
    d = arr.type.list_size
    return np.asarray(flat, dtype=np.float64).reshape(-1, d)


def _frozen_candidates(source: str, fam_counts: dict[str, int]) -> list[str]:
    """D12: familias nativas con ≥1 fila en train, en orden congelado."""
    native = _registry(source)["native_families"]
    cands = [f for f in native if fam_counts.get(f, 0) > 0]
    faltan = sorted(set(native) - set(cands))
    if faltan:
        print(f"[d12] {source}: sin filas en train → fuera del candidate set: {faltan}")
    return cands


def _idx_map(candidates: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(candidates)}


# ── UNSW / CIC: datasets en RAM ────────────────────────────────────────
def _load_network_split(path: Path, source: str, split_col: str, split: str,
                        candidates: list[str], need_ddos: bool,
                        exclude_family: str | None = None
                        ) -> tuple[np.ndarray, dict[str, list[str | None]], Labels]:
    txt_keys = ("proto", "service", "state") if source == "unsw" else ("protocol",)
    cols = ["x_num", split_col, "label_family", "label_severity"]
    cols += [f"x_txt_{k}" for k in txt_keys]
    if need_ddos:
        cols += ["day", "label_full"]  # el subset usa etiquetas crudas (label_full)
    tbl = pq.read_table(path, columns=cols)
    mask = np.asarray(tbl.column(split_col).to_pylist()) == split
    x = _read_x_num(path)[mask]
    fam = np.asarray(tbl.column("label_family").to_pylist())[mask]
    sev = np.asarray(tbl.column("label_severity").to_pylist())[mask]
    txt = {k: list(np.asarray(tbl.column(f"x_txt_{k}").to_pylist())[mask])
           for k in txt_keys}

    # LOAO: la familia excluida sale de train Y val (§6.7); test se evalúa completo
    if exclude_family is not None and split != "test":
        keep = fam != exclude_family
        x, fam, sev = x[keep], fam[keep], sev[keep]
        txt = {k: [v for v, kp in zip(vals, keep) if kp] for k, vals in txt.items()}

    imap = _idx_map(candidates)
    # filas con familia fuera del candidate set (D12: excluida LOAO en test,
    # o recon/botnet en val/test de split_time) → sentinela -1; los nombres
    # se conservan para E2 (evaluación de la familia excluida, §6.7)
    y_fam = torch.tensor([imap.get(f, -1) for f in fam], dtype=torch.int64)
    y_bin = torch.tensor([f != "normal" for f in fam], dtype=torch.int64)
    y_sev = torch.tensor(sev, dtype=torch.int64)

    if need_ddos:
        # subset congelado (SCHEMA §3.3): días Wednesday+Friday-DDoS; solo
        # mapean a las 4 clases las filas con etiqueta nativa en DDOS_FULL
        # (642,862 de las 918,437; el resto del día no entrena este head)
        day = np.asarray(tbl.column("day").to_pylist())[mask]
        full = np.asarray(tbl.column("label_full").to_pylist())[mask]
        if exclude_family is not None and split != "test":
            day, full = day[keep], full[keep]  # misma poda LOAO que x/fam
        in_subset = np.asarray([d in DDOS_DAYS and f in DDOS_FULL
                                for d, f in zip(day, full)])
        ddos_mask = torch.tensor(in_subset)
        y_ddos = torch.zeros(len(fam), dtype=torch.int64)  # alineado a la fila
        y_ddos[ddos_mask] = torch.tensor(
            [_idx_map(DDOS_CANDIDATES)[_DDOS_MAP[f]] for f in np.asarray(full)[in_subset]],
            dtype=torch.int64)
    else:
        ddos_mask = y_ddos = None
    return x, txt, Labels(y_bin, y_fam, y_sev, ddos_mask, y_ddos), fam


def build_network_run(source: str, scale: str, split_scheme: str,
                      exclude_family: str | None = None) -> RunData:
    path = DATA / ("unsw_nb15/unsw_nb15.parquet" if source == "unsw"
                   else "cicids2017/cicids2017.parquet")
    split_col = "split" if source == "unsw" else f"split_{split_scheme}"
    need_ddos = source == "cic" and split_scheme == "strat"

    # 1) conteos de train de la corrida → candidates (D12) + ω (§3.5)
    count_cols = [split_col, "label_family"] + (["day"] if need_ddos else [])
    tbl = pq.read_table(path, columns=count_cols)
    tr = np.asarray(tbl.column(split_col).to_pylist()) == "train"
    fam_all = np.asarray(tbl.column("label_family").to_pylist())
    if exclude_family is not None:
        tr = tr & (fam_all != exclude_family)
    counts = {f: int((fam_all[tr] == f).sum()) for f in _registry(source)["native_families"]}
    candidates = _frozen_candidates(source, counts)
    omega = torch.tensor(class_weights([counts[c] for c in candidates]))

    # 2) splits (LOAO: la familia sale de train Y val; test completo §6.7)
    x_tr, txt_tr, lab_tr, _ = _load_network_split(path, source, split_col, "train", candidates, need_ddos, exclude_family)
    ddos_counts = None
    if need_ddos:
        sub = lab_tr.ddos_mask
        ddos_counts = [int((lab_tr.y_ddos[sub] == i).sum())
                       for i in range(len(DDOS_CANDIDATES))]
        if any(c == 0 for c in ddos_counts):
            # fold LOAO dos_ddos: las clases de ataque del vector ddos viven
            # en la familia excluida (el subset conserva solo BENIGN) → sin
            # head ddos en esta corrida (E2 evalúa noul/choice en la familia)
            need_ddos = False
            ddos_counts = None
            lab_tr = Labels(lab_tr.y_bin, lab_tr.y_fam, lab_tr.y_sev, None, None)
    x_va, txt_va, lab_va, va_fams = _load_network_split(path, source, split_col, "val", candidates, need_ddos, exclude_family)
    x_te, txt_te, lab_te, te_fams = _load_network_split(path, source, split_col, "test", candidates, need_ddos, None)

    # 3) vocab (train de la corrida, §4.2) + φ (train)
    vocab = None
    if source == "unsw":
        vocab = {k: TextVocab.fit(k, txt_tr[k]) for k in ("proto", "service", "state")}
    stage = FeatureStage.fit(x_tr.astype(np.float64))

    u_tr, _ = stage.transform(x_tr)
    u_va, _ = stage.transform(x_va)
    u_te, _ = stage.transform(x_te)
    train_txt = {k: torch.tensor(vocab[k].encode(txt_tr[k]), dtype=torch.int64) for k in vocab} if vocab else None
    val_txt = {k: torch.tensor(vocab[k].encode(txt_va[k]), dtype=torch.int64) for k in vocab} if vocab else None
    test_txt = {k: torch.tensor(vocab[k].encode(txt_te[k]), dtype=torch.int64) for k in vocab} if vocab else None

    ddos_candidates = list(DDOS_CANDIDATES) if need_ddos else None
    omega_ddos = None
    if need_ddos:
        assert all(c > 0 for c in ddos_counts), f"ddos class sin filas: {ddos_counts}"
        omega_ddos = torch.tensor(class_weights(ddos_counts))

    heads = HeadSpec(n_candidates=len(candidates), ddos_candidates=4 if need_ddos else 0)
    return RunData(
        source=source, scale=scale, split_scheme=split_scheme,
        candidates=candidates, ddos_candidates=ddos_candidates,
        omega=omega, omega_ddos=omega_ddos, vocab=vocab, feature_stage=stage,
        heads=heads,
        train=torch.tensor(u_tr), train_txt=train_txt, train_labels=lab_tr,
        val=torch.tensor(u_va), val_txt=val_txt, val_labels=lab_va,
        test=torch.tensor(u_te), test_txt=test_txt, test_labels=lab_te,
        val_fam_names=va_fams, test_fam_names=te_fams,
        meta={"train_counts": {c: counts[c] for c in candidates},
              "exclusion": {"exclude": exclude_family} if exclude_family else {}},
    )


# ── EMBER: streaming por shards ────────────────────────────────────────
def ember_shards(subset: str) -> list[Path]:
    d = DATA / "ember2024"
    if subset == "train":
        return [d / f"part_{i:02d}_train.parquet" for i in range(46)]
    if subset == "val":
        return [d / f"part_{i:02d}_val.parquet" for i in range(46, 52)]
    return [d / f"part_{i:02d}_test.parquet" for i in range(52, 64)]


def _ember_labels(candidates: list[str]):
    imap = _idx_map(candidates)

    def decode(tbl) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        fam = tbl.column("label_family").to_pylist()
        # A5-EMBER: 'unknown' fuera del candidate set → sentinela -1 (la
        # máscara de _batch_losses la saca del CE de choice)
        y_fam = torch.tensor([imap.get(f, -1) for f in fam], dtype=torch.int64)
        y_bin = torch.tensor(tbl.column("label_binary").to_pylist(), dtype=torch.int64)
        return y_fam, y_bin
    return decode


def build_ember_run(scale: str, exclude_family: str | None = None) -> RunData:
    tr_shards, va_shards, te_shards = ember_shards("train"), ember_shards("val"), ember_shards("test")

    # 1) conteos de train (una pasada ligera por shard, solo label_family)
    counts = {f: 0 for f in _registry("ember")["native_families"]}
    for p in tr_shards:
        for f in pq.read_table(p, columns=["label_family"]).column("label_family").to_pylist():
            counts[f] += 1
    if exclude_family is not None:
        counts[exclude_family] = 0
    candidates = _frozen_candidates("ember2024_v1", counts)
    omega = torch.tensor(class_weights([counts[c] for c in candidates]))

    # 2) φ con submuestra uniforme de train (decisión documentada arriba)
    rng = np.random.default_rng(42)
    sample_chunks: list[np.ndarray] = []
    for p in tr_shards:
        x = _read_x_num(p).astype(np.float32)
        take = rng.choice(x.shape[0], size=EMBER_STATS_ROWS_PER_SHARD, replace=False)
        sample_chunks.append(x[take].astype(np.float64))
        del x
    stage = FeatureStage.fit(np.concatenate(sample_chunks))
    del sample_chunks

    return RunData(
        source="ember2024_v1", scale=scale, split_scheme="temporal",
        candidates=candidates, ddos_candidates=None,
        omega=omega, omega_ddos=None, vocab=None, feature_stage=stage,
        heads=HeadSpec(n_candidates=len(candidates), has_score=False),
        train=torch.empty(0), train_txt=None,
        train_labels=Labels(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64), None, None, None),
        val=torch.empty(0), val_txt=None,
        val_labels=Labels(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64), None, None, None),
        test=None, test_txt=None, test_labels=None,
        ember_train_shards=tr_shards, ember_val_shards=va_shards, ember_test_shards=te_shards,
        meta={"train_counts": {c: counts[c] for c in candidates}, "exclusion": {"exclude": exclude_family} if exclude_family else {}},
    )


class EmberBatcher:
    """Itera batches de train (o evalúa val) con streaming por shards.

    Train: permutación de shards seed 42+epoch → decode → shuffle intra-shard
    → batches con drop_last. Prefetch de 1 shard en thread (§6.2).
    Eval (val/test/challenge): shards en orden fijo, batches sin drop_last,
    shuffle off.
    """

    def __init__(self, run: RunData) -> None:
        self.run = run
        self.stage = run.feature_stage
        self._decode = _ember_labels(run.candidates)

    def _load_shard(self, path: Path) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        tbl = pq.read_table(path, columns=["x_num", "label_family", "label_binary"])
        x = _read_x_num(path)  # float64 para φ
        u, _ = self.stage.transform(x)
        del x
        y_fam, y_bin = self._decode(tbl)
        return u, y_fam, y_bin

    def _iter_shard_batches(self, path: Path, batch_size: int, rng: np.random.Generator | None):
        u, y_fam, y_bin = self._load_shard(path)
        n = u.shape[0]
        order = rng.permutation(n) if rng is not None else np.arange(n)
        n_batches = n // batch_size if rng is not None else int(np.ceil(n / batch_size))
        for b in range(n_batches):
            idx = order[b * batch_size:(b + 1) * batch_size]
            yield (torch.tensor(u[idx]), y_fam[idx], y_bin[idx])

    def _stream(self, shards: list[Path], batch_size: int, rng: np.random.Generator | None):
        """Genera batches; con prefetch de 1 shard (thread) si rng (train)."""
        if rng is None:  # eval: secuencial simple
            for p in shards:
                yield from self._iter_shard_batches(p, batch_size, None)
            return

        # train: productor en thread descarga el shard i+1 mientras se
        # consumen los batches del shard i (prefetch=1, §6.2).
        import queue

        q: queue.Queue = queue.Queue(maxsize=1)
        exc: list[BaseException] = []

        def producer() -> None:
            try:
                for p in shards:
                    q.put(self._load_shard(p))
            except BaseException as e:
                exc.append(e)
            finally:
                q.put(None)

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            u, y_fam, y_bin = item
            n = u.shape[0]
            order = rng.permutation(n)
            for b in range(n // batch_size):  # drop_last
                idx = order[b * batch_size:(b + 1) * batch_size]
                yield (torch.tensor(u[idx]), y_fam[idx], y_bin[idx])
        if exc:
            raise exc[0]

    def train_batches(self, epoch: int, batch_size: int):
        rng = np.random.default_rng(42 + epoch)
        return self._stream(self.run.ember_train_shards, batch_size, rng)

    def eval_batches(self, subset: str, batch_size: int):
        # "train" = streaming en orden (sin shuffle/drop_last): para
        # materializar features en los baselines tabulares
        shards = {"train": self.run.ember_train_shards,
                  "val": self.run.ember_val_shards,
                  "test": self.run.ember_test_shards}[subset]
        return self._stream(shards, batch_size, None)


# ── fábrica ────────────────────────────────────────────────────────────
def build_run(source: str, scale: str, split_scheme: str,
              exclude_family: str | None = None) -> RunData:
    if source == "unsw":
        return build_network_run("unsw", scale, "oficial", exclude_family)
    if source == "cic":
        return build_network_run("cic", scale, split_scheme, exclude_family)
    if source == "ember":
        return build_ember_run(scale, exclude_family)
    raise ValueError(source)
