"""VX-One engine datasets and loaders (spec F2 §0.1, §6.2, D12).

- UNSW / CIC: full tensors in RAM (UNSW 149K×40×4B ≈ 24 MB;
CIC train 1.98M×78×4B ≈ 620 MB).
- EMBER: streaming by shards (46 train parquets of 60K rows; ~616 MB
decoded each): shards are swapped per epoch (seed 42+epoch),
intra-shard shuffle, batches with drop_last; prefetch of 1 shard in thread.
Two-level shuffle (shard + intra) — documented approach (§6.2).

Candidate set per run (D12): native families with ≥1 row in low TRAIN
the scheme of the run, in the FROZEN ORDER of the record. Weights ω
(§3.5) derived from the train counts OF THE RUN (LOAO recalculates them).

φ (feature stage) is set ONLY with train:
- UNSW/CIC: complete train in RAM (float64).
- EMBER: uniform subsample of 60,000 train rows (1,304 per shard,
rng seed 42, one pass through the 46 shards) — the SE of the median with
n=60K is negligible and avoids 28 GB in RAM; documented decision.
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
    """Short alias (unsw/cic/ember) or registry key → registry entry."""
    return REGISTRY[ALIAS.get(source, source)]

DDOS_DAYS = {
    "Wednesday-workingHours.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",
}
DDOS_FULL = {"BENIGN", "DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"}
DDOS_CANDIDATES = ("normal", "tcp_flood", "http_dos", "slowloris")  # frozen order F2 D8
_DDOS_MAP = {"BENIGN": "normal", "DDoS": "tcp_flood", "DoS Hulk": "http_dos",
             "DoS GoldenEye": "http_dos", "DoS slowloris": "slowloris",
             "DoS Slowhttptest": "slowloris"}

EMBER_STATS_ROWS_PER_SHARD = 1_304  # × 46 shards = 60,024 stat rows


@dataclass
class Labels:
    y_bin: torch.Tensor        # int64 [N] 1[family != normal] (or label_binary in EMBER)
    y_fam: torch.Tensor        # int64 [N] index in candidates
    y_sev: torch.Tensor | None # int64 [N] 0..3 (None in EMBER)
    ddos_mask: torch.Tensor | None  # bool [N] subset rows (CIC strat only)
    y_ddos: torch.Tensor | None     # int64 [N] index in DDOS_CANDIDATES


@dataclass
class RunData:
    """Everything a bullfight needs, materialized and verified."""
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
    test: torch.Tensor | None  # None in EMBER (streaming, does not materialize)
    test_txt: dict[str, torch.Tensor] | None
    test_labels: Labels | None
    val_fam_names: np.ndarray | None = None   # native family by row (E2; -1 = out of candidates)
    test_fam_names: np.ndarray | None = None
    ember_train_shards: list[Path] | None = None
    ember_val_shards: list[Path] | None = None
    ember_test_shards: list[Path] | None = None
    meta: dict = field(default_factory=dict)


# ── reading utilities ─────────────────────── ───────────────────────
def _read_x_num(path: Path) -> np.ndarray:
    tbl = pq.read_table(path, columns=["x_num"])
    arr = tbl.column("x_num").combine_chunks()
    flat = arr.flatten().to_numpy(zero_copy_only=False)
    d = arr.type.list_size
    return np.asarray(flat, dtype=np.float64).reshape(-1, d)


def _frozen_candidates(source: str, fam_counts: dict[str, int]) -> list[str]:
    """D12: native families with ≥1 row in train, in frozen order."""
    native = _registry(source)["native_families"]
    cands = [f for f in native if fam_counts.get(f, 0) > 0]
    faltan = sorted(set(native) - set(cands))
    if faltan:
        print(f"[d12] {source}: sin filas en train → fuera del candidate set: {faltan}")
    return cands


def _idx_map(candidates: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(candidates)}


# ── UNSW / CIC: datasets in RAM ──────────────────── ────────────────────
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

    # LOAO: the excluded family leaves train Y val (§6.7); test is evaluated complete
    if exclude_family is not None and split != "test":
        keep = fam != exclude_family
        x, fam, sev = x[keep], fam[keep], sev[keep]
        txt = {k: [v for v, kp in zip(vals, keep) if kp] for k, vals in txt.items()}

    imap = _idx_map(candidates)
    # rows with family outside the candidate set (D12: excluded LOAO in test,
    # or recon/botnet in val/test of split_time) → sentinel -1; the names
    # are retained for E2 (excluded family evaluation, §6.7)
    y_fam = torch.tensor([imap.get(f, -1) for f in fam], dtype=torch.int64)
    y_bin = torch.tensor([f != "normal" for f in fam], dtype=torch.int64)
    y_sev = torch.tensor(sev, dtype=torch.int64)

    if need_ddos:
        # frozen subset (SCHEMA §3.3): days Wednesday+Friday-DDoS; only
        # map the rows with native label in DDOS_FULL to the 4 classes
        # (642,862 of the 918,437; the rest of the day this head does not train)
        day = np.asarray(tbl.column("day").to_pylist())[mask]
        full = np.asarray(tbl.column("label_full").to_pylist())[mask]
        if exclude_family is not None and split != "test":
            day, full = day[keep], full[keep]  # same LOAO pruning as x/fam
        in_subset = np.asarray([d in DDOS_DAYS and f in DDOS_FULL
                                for d, f in zip(day, full)])
        ddos_mask = torch.tensor(in_subset)
        y_ddos = torch.zeros(len(fam), dtype=torch.int64)  # aligned to the row
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

    # 1) train counts of the run → candidates (D12) + ω (§3.5)
    count_cols = [split_col, "label_family"] + (["day"] if need_ddos else [])
    tbl = pq.read_table(path, columns=count_cols)
    tr = np.asarray(tbl.column(split_col).to_pylist()) == "train"
    fam_all = np.asarray(tbl.column("label_family").to_pylist())
    if exclude_family is not None:
        tr = tr & (fam_all != exclude_family)
    counts = {f: int((fam_all[tr] == f).sum()) for f in _registry(source)["native_families"]}
    candidates = _frozen_candidates(source, counts)
    omega = torch.tensor(class_weights([counts[c] for c in candidates]))

    # 2) splits (LOAO: the family leaves train Y val; full test §6.7)
    x_tr, txt_tr, lab_tr, _ = _load_network_split(path, source, split_col, "train", candidates, need_ddos, exclude_family)
    ddos_counts = None
    if need_ddos:
        sub = lab_tr.ddos_mask
        ddos_counts = [int((lab_tr.y_ddos[sub] == i).sum())
                       for i in range(len(DDOS_CANDIDATES))]
        if any(c == 0 for c in ddos_counts):
            # fold LOAO dos_ddos: ddos ​​vector attack classes live
            # in the excluded family (the subset keeps only BENIGN) → without
            # head ddos ​​in this run (E2 evaluates noul/choice in the family)
            need_ddos = False
            ddos_counts = None
            lab_tr = Labels(lab_tr.y_bin, lab_tr.y_fam, lab_tr.y_sev, None, None)
    x_va, txt_va, lab_va, va_fams = _load_network_split(path, source, split_col, "val", candidates, need_ddos, exclude_family)
    x_te, txt_te, lab_te, te_fams = _load_network_split(path, source, split_col, "test", candidates, need_ddos, None)

    # 3) vocab (train of the run, §4.2) + φ (train)
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


# ── EMBER: streaming by shards ──────────────────── ────────────────────
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
        # _batch_losses mask takes it out of the CE of choice)
        y_fam = torch.tensor([imap.get(f, -1) for f in fam], dtype=torch.int64)
        y_bin = torch.tensor(tbl.column("label_binary").to_pylist(), dtype=torch.int64)
        return y_fam, y_bin
    return decode


def build_ember_run(scale: str, exclude_family: str | None = None) -> RunData:
    tr_shards, va_shards, te_shards = ember_shards("train"), ember_shards("val"), ember_shards("test")

    # 1) train counts (a light pass through shard, label_family only)
    counts = {f: 0 for f in _registry("ember")["native_families"]}
    for p in tr_shards:
        for f in pq.read_table(p, columns=["label_family"]).column("label_family").to_pylist():
            counts[f] += 1
    if exclude_family is not None:
        counts[exclude_family] = 0
    candidates = _frozen_candidates("ember2024_v1", counts)
    omega = torch.tensor(class_weights([counts[c] for c in candidates]))

    # 2) φ with uniform train subsampling (decision documented above)
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
    """Iterates train batches (or evaluates val) with streaming by shards.

Train: permutation of shards seed 42+epoch → decode → shuffle intra-shard
→ batches with drop_last. Prefetch 1 shard in thread (§6.2).
Eval (val/test/challenge): shards in fixed order, batches without drop_last,
shuffle off.
    """

    def __init__(self, run: RunData) -> None:
        self.run = run
        self.stage = run.feature_stage
        self._decode = _ember_labels(run.candidates)

    def _load_shard(self, path: Path) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        tbl = pq.read_table(path, columns=["x_num", "label_family", "label_binary"])
        x = _read_x_num(path)  # float64 for φ
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
        """Generate batches; with prefetch of 1 shard (thread) if rng (train)."""
        if rng is None:  # eval: simple sequential
            for p in shards:
                yield from self._iter_shard_batches(p, batch_size, None)
            return

        # train: producer in thread downloads shard i+1 while
        # consume the batches from shard i (prefetch=1, §6.2).
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
        # "train" = streaming in order (without shuffle/drop_last): for
        # materialize features in tabular baselines
        shards = {"train": self.run.ember_train_shards,
                  "val": self.run.ember_val_shards,
                  "test": self.run.ember_test_shards}[subset]
        return self._stream(shards, batch_size, None)


# ── factory ────────────────────────────── ──────────────────────────────
def build_run(source: str, scale: str, split_scheme: str,
              exclude_family: str | None = None) -> RunData:
    if source == "unsw":
        return build_network_run("unsw", scale, "oficial", exclude_family)
    if source == "cic":
        return build_network_run("cic", scale, split_scheme, exclude_family)
    if source == "ember":
        return build_ember_run(scale, exclude_family)
    raise ValueError(source)
