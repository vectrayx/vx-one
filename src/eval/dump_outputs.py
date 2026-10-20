"""Dump de logits crudos de val/test de un artefacto VX-One (§8.2) a npz.

Puente entrenamiento→evaluación: los trabajos GPU (Modal) solo entrenan y
dumpean; TODAS las métricas E1-E3 se computan local desde estos npz.
Guarda logits crudos (la calibración viaja en el ckpt; calibrated =
f(logits, T) se recomputa en eval) + labels + nombres nativos por fila
(necesario para E2 LOAO: y_fam = -1 con test_fam_names = familia excluida).

Uso:
    python -m src.eval.dump_outputs --artifact results/artifacts/unsw_small_oficial.pt \
        --out results/outputs/unsw_small_oficial.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from ..models.data import build_run
from ..models.engine import SOURCE_ALIAS
from ..models.network import VXOneNet, HeadSpec
from ..models.vocab import TextVocab

EVAL_BS = 8192


def _load_net(artifact: str | Path, device: str) -> tuple[VXOneNet, dict, dict]:
    ckpt = torch.load(artifact, map_location=device, weights_only=False)
    source = SOURCE_ALIAS[ckpt["source"]]
    hm = ckpt["head_meta"]
    has_ddos = "ddos" in hm
    vocab = {k: TextVocab.from_dict(v) for k, v in ckpt.get("vocab", {}).items()}
    heads = HeadSpec(
        n_candidates=len(hm["choice"]["candidates"]),
        ddos_candidates=4 if has_ddos else 0,
        has_score="score" in hm)
    net = VXOneNet(source, ckpt["config"]["scale"], heads,
                   ckpt["config"]["d_num"],
                   vocab_sizes={k: v.size for k, v in vocab.items()})
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net, ckpt, {"source": source}


@torch.no_grad()
def _forward(net: VXOneNet, u: torch.Tensor, txt: dict | None,
             device: str) -> dict[str, np.ndarray]:
    outs: dict[str, list[np.ndarray]] = {}
    for i in range(0, u.shape[0], EVAL_BS):
        ub = u[i:i + EVAL_BS].to(device)
        tb = {k: v[i:i + EVAL_BS].to(device) for k, v in (txt or {}).items()}
        o = net(ub, tb if tb else None)
        for k, v in o.items():
            outs.setdefault(k, []).append(v.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def dump_ember_challenge(artifact: str | Path, out_npz: str | Path,
                         device: str = "cpu") -> dict:
    """Streaming de logits sobre el challenge set EMBER (estrés E2/E4:
    0 benign, ~99% unknown, evasivo). Mismo pipeline/feature-stage que
    dump_run (FeatureStage refit determinista en build_run), solo cambia
    el fuente: challenge.parquet como único 'shard'.
    Guarda noul + choice logits crudos + labels (y_bin/y_fam)."""
    net, ckpt, _ = _load_net(artifact, device)
    source = SOURCE_ALIAS[ckpt["source"]]
    assert source == "ember", f"challenge solo para EMBER (source={source})"
    exc = ckpt.get("manifest", {}).get("exclusion", {}).get("exclude")
    run = build_run(source, ckpt["config"]["scale"],
                    ckpt["config"]["split_scheme"], exclude_family=exc)

    from ..models.data import DATA, EmberBatcher
    challenge_path = DATA / "ember2024/challenge.parquet"
    assert challenge_path.exists(), challenge_path

    b = EmberBatcher(run)
    outs: dict[str, list[np.ndarray]] = {}
    yb_l, yf_l = [], []
    with torch.no_grad():
        for u, yf, yb in b._stream([challenge_path], EVAL_BS, None):
            o = net(u.to(device), None)
            for k, v in o.items():
                outs.setdefault(k, []).append(v.float().cpu().numpy())
            yb_l.append(yb.numpy())
            yf_l.append(yf.numpy())
    parts = {
        "challenge_noul_logits": np.concatenate(outs["noul"]),
        "challenge_choice_logits": np.concatenate(outs["choice"]),
        "challenge_y_bin": np.concatenate(yb_l),
        "challenge_y_fam": np.concatenate(yf_l),
        "challenge_label_family": np.asarray(
            pq.read_table(challenge_path, columns=["label_family"])
            .column(0).to_pylist(), dtype=object),
    }
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **parts)
    hm = ckpt["head_meta"]
    meta = {
        "artifact": str(artifact), "challenge": True,
        "scale": ckpt["config"]["scale"],
        "split_scheme": ckpt["config"]["split_scheme"],
        "exclude": exc, "temperatures": ckpt["manifest"]["temperatures"],
        "m_val": ckpt["manifest"]["m_val"],
        "candidates": hm["choice"]["candidates"],
        "omega": hm["choice"]["class_weights"].tolist(),
        "mode": hm["choice"]["mode"], "tau_abs": hm["choice"]["tau_abs"],
        "n": int(len(parts["challenge_y_bin"])),
        "n_pos": int((parts["challenge_y_bin"] == 1).sum()),
    }
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return {"out": str(out_npz), **{k: v for k, v in meta.items()
                                    if k in ("n", "n_pos")}}


def dump_run(artifact: str | Path, out_npz: str | Path,
             device: str = "cpu") -> dict:
    net, ckpt, _ = _load_net(artifact, device)
    source = SOURCE_ALIAS[ckpt["source"]]
    exc = ckpt["manifest"].get("exclusion", {}).get("exclude")
    run = build_run(source, ckpt["config"]["scale"],
                    ckpt["config"]["split_scheme"], exclude_family=exc)

    vocab_sizes_ckpt = {k: len(v["token_to_slot"]) + 2
                        for k, v in ckpt.get("vocab", {}).items()}
    if not vocab_sizes_ckpt:
        # A1 (sin rama textual): el artefacto se entrenó sin vocab → correr
        # sin txt (la red no tiene embeddings) en vez de chocar con el run
        run.train_txt = run.val_txt = run.test_txt = None
        vocab_sizes_run = {}
    else:
        vocab_sizes_run = {k: v.size for k, v in (run.vocab or {}).items()}
    assert vocab_sizes_ckpt == vocab_sizes_run, \
        f"vocab mismatch: ckpt {vocab_sizes_ckpt} vs run {vocab_sizes_run}"

    # candidate set del artefacto puede diferir del run reconstruido
    # (A5-EMBER 10-way): remap y_fam run→ckpt, filas fuera → -1
    cand_ckpt = ckpt["head_meta"]["choice"]["candidates"]
    cand_map = None
    if cand_ckpt != run.candidates:
        imap = {c: i for i, c in enumerate(cand_ckpt)}
        cand_map = np.array([imap.get(c, -1) for c in run.candidates],
                            dtype=np.int64)

    def _yf(lab_y_fam: np.ndarray) -> np.ndarray:
        yf = lab_y_fam
        if cand_map is None:
            return yf
        return np.where(yf >= 0, cand_map[np.clip(yf, 0, None)], -1)

    parts: dict[str, np.ndarray] = {}
    for prefix, tensor, txt, lab, fam_names in (
            ("va", run.val, run.val_txt, run.val_labels, run.val_fam_names),
            ("te", run.test, run.test_txt, run.test_labels, run.test_fam_names)):
        if tensor is None or tensor.numel() == 0:
            continue
        o = _forward(net, tensor, txt, device)
        parts[f"{prefix}_noul_logits"] = o["noul"]
        parts[f"{prefix}_choice_logits"] = o["choice"]
        parts[f"{prefix}_y_bin"] = lab.y_bin.numpy()
        parts[f"{prefix}_y_fam"] = _yf(lab.y_fam.numpy())
        if lab.y_sev is not None:
            parts[f"{prefix}_y_sev"] = lab.y_sev.numpy()
            parts[f"{prefix}_corn_logits"] = o["corn_logits"]
        if lab.ddos_mask is not None:
            parts[f"{prefix}_ddos_mask"] = lab.ddos_mask.numpy()
            parts[f"{prefix}_y_ddos"] = lab.y_ddos.numpy()
            parts[f"{prefix}_ddos_logits"] = o["ddos"]
        if fam_names is not None:
            parts[f"{prefix}_fam_names"] = np.asarray(fam_names, dtype=object)
    if source == "ember":
        # EMBER: val/test por streaming (no materializados en RunData)
        from ..models.data import EmberBatcher
        b = EmberBatcher(run)
        with torch.no_grad():
            for subset, prefix in (("val", "va"), ("test", "te")):
                outs: dict[str, list[np.ndarray]] = {}
                yb_l, yf_l = [], []
                for u, yf, yb in b.eval_batches(subset, EVAL_BS):
                    o = net(u.to(device), None)
                    for k, v in o.items():
                        outs.setdefault(k, []).append(v.float().cpu().numpy())
                    yb_l.append(yb.numpy())
                    yf_l.append(yf.numpy())
                flat = {k: np.concatenate(v) for k, v in outs.items()}
                parts[f"{prefix}_noul_logits"] = flat["noul"]
                parts[f"{prefix}_choice_logits"] = flat["choice"]
                parts[f"{prefix}_y_bin"] = np.concatenate(yb_l)
                parts[f"{prefix}_y_fam"] = _yf(np.concatenate(yf_l))

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **parts)
    hm = ckpt["head_meta"]
    meta = {"artifact": str(artifact), "source": source,
            "scale": ckpt["config"]["scale"],
            "split_scheme": ckpt["config"]["split_scheme"],
            "exclude": exc, "temperatures": ckpt["manifest"]["temperatures"],
            "m_val": ckpt["manifest"]["m_val"],
            "best_epoch": ckpt["manifest"]["best_epoch"],
            # contrato de inferencia completo para recomputar calibradas
            "candidates": hm["choice"]["candidates"],
            "omega": hm["choice"]["class_weights"].tolist(),
            "mode": hm["choice"]["mode"], "tau_abs": hm["choice"]["tau_abs"],
            **({"omega_ddos": hm["ddos"]["class_weights"].tolist(),
                "ddos_candidates": hm["ddos"]["candidates"]}
               if "ddos" in hm else {})}
    Path(str(out_npz).replace(".npz", ".json")).write_text(
        json.dumps(meta, indent=2, default=str))
    print(f"[dump] {out_npz}: {len(parts)} arrays "
          + " ".join(f"{k}={v.shape}" for k, v in list(parts.items())[:4]) + " ...")
    return meta


def merge_a2_outputs(base: str, out_dir: str | Path) -> dict:
    """A2 (sin multi-task): combina los dumps single-head (a2_noul /
    a2_choice / a2_score) en el npz de la corrida `<base>_abl_a2`. Cada
    contribución aporta SU cabeza entrenada (noul_logits de a2_noul,
    choice_logits de a2_choice, corn_logits de a2_score); las cabezas sin
    gradiente de cada corrida se descartan. El merge usa el mismo formato
    §8.2, así que el evaluador E1-E3 estándar lo lee sin cambios."""
    out_dir = Path(out_dir)
    parts_d: dict[str, dict] = {}
    metas: dict[str, dict] = {}
    for head in ("noul", "choice", "score"):
        p = out_dir / f"{base}_abl_a2_{head}.npz"
        if not p.exists():
            print(f"[a2] falta {p.name} — merge parcial")
            continue
        parts_d[head] = dict(np.load(p, allow_pickle=True))
        metas[head] = json.loads(Path(str(p).replace(".npz", ".json")).read_text())
    if "noul" not in parts_d or "choice" not in parts_d:
        raise FileNotFoundError(f"[a2] se requieren noul y choice: {sorted(parts_d)}")

    merged: dict[str, np.ndarray] = {}
    # labels + fam_names de la corrida noul (idénticos entre corridas)
    for k, v in parts_d["noul"].items():
        if any(k.endswith(s) for s in ("y_bin", "y_fam", "y_sev", "fam_names",
                                       "ddos_mask", "y_ddos")):
            merged[k] = v
    arr_key = {"noul": "noul_logits", "choice": "choice_logits",
               "score": "corn_logits"}
    for head, arr in arr_key.items():
        if head not in parts_d:
            continue
        for pre in ("va", "te"):
            k = f"{pre}_{arr}"
            if k in parts_d[head]:
                merged[k] = parts_d[head][k]

    meta = dict(metas["noul"])
    meta["artifact"] = str(out_dir / f"{base}_abl_a2.npz")
    meta["a2"] = True
    meta["m_val"] = {h: metas[h]["m_val"] for h in metas}
    meta["best_epoch"] = {h: metas[h]["best_epoch"] for h in metas}
    temps: dict[str, float] = {}
    for head in ("noul", "choice", "score"):
        if head in metas and head in metas[head]["temperatures"]:
            temps[head] = metas[head]["temperatures"][head]
    meta["temperatures"] = temps
    # meta de la cabeza Choice (mode/tau_abs/ω/candidates) desde a2_choice
    for k in ("candidates", "omega", "mode", "tau_abs",
              "omega_ddos", "ddos_candidates"):
        if "choice" in metas and k in metas["choice"]:
            meta[k] = metas["choice"][k]

    out_npz = out_dir / f"{base}_abl_a2.npz"
    np.savez_compressed(out_npz, **merged)
    Path(str(out_npz).replace(".npz", ".json")).write_text(
        json.dumps(meta, indent=2, default=str))
    print(f"[a2] {base}_abl_a2.npz: {len(merged)} arrays, "
          f"heads={sorted(parts_d)}, T={temps}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dump_run(args.artifact, args.out, args.device)


if __name__ == "__main__":
    main()
