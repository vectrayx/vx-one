"""VX-One training loop (spec F2 §6.3-§6.5, D11).

Frozen hyperparameters:
- AdamW β=(0.9,0.999), eps=1e-8; weight decay 1e-2 ONLY in weight matrices
Linear 2D (excludes LN, biases, embeddings, corn_b and s_q).
- Schedule: linear warmup 5% + cosine up to 0; standard clip 1.0.
- Early stopping: patience 4 over M_val = average head metrics
active (AUROC noul, macro-F1 choice without abstention, 1−MAE(ŝ,y)/3 score,
macro-F1 ddos ​​in subset).
- AMP: bf16 on L4, fp16 on T4, off on CPU.
- Upon completion: the best epoch is restored, each head is calibrated in val
(§5, NLL) and the self-contained artifact is written (§8.2).
"""

from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .calibration import (fit_choice_temperature, fit_corn_temperature,
                          fit_noul_temperature)
from .data import RunData
from .engine import SOURCE_KEY, TAU_ABS, save_artifact
from .losses import (KendallMultiTask, choice_loss, corn_expected, corn_loss,
                     corn_pmf, noul_loss)
from .network import VXOneNet

# Frozen table §6.3: {scale: {source: (batch_size, epochs)}}; lr by scale.
TRAIN_TABLE: dict[str, dict[str, tuple[int, int]]] = {
    "nano":  {"unsw": (256, 40),  "cic": (1024, 25), "ember": (2048, 12)},
    "small": {"unsw": (512, 30),  "cic": (2048, 20), "ember": (4096, 10)},
    "base":  {"unsw": (1024, 25), "cic": (4096, 15), "ember": (8192, 8)},
}
LR_TABLE = {"nano": 3e-3, "small": 1e-3, "base": 3e-4}
WARMUP_FRAC = 0.05
CLIP_NORM = 1.0
PATIENCE = 4
WEIGHT_DECAY = 1e-2
SEED = 42
EVAL_BS = 8192


def pick_amp(device: str, amp: str = "auto") -> str:
    """'off' | 'bf16' | 'fp16' — auto: bf16 si CC≥8 (L4), fp16 si T4, off CPU."""
    if amp != "auto":
        return amp
    if device.startswith("cuda"):
        cc = torch.cuda.get_device_capability(device)
        return "bf16" if cc[0] >= 8 else "fp16"
    return "off"


def _param_groups(net: VXOneNet) -> list[dict]:
    """wd only in 2D Linear weight matrices (D11): LN/biases/embeddings out
(corn_b and s_q are 1D/0D and are excluded by the dims rule)."""
    decay = [p for n, p in net.named_parameters() if p.dim() == 2 and not n.startswith("emb.")]
    no_decay = [p for n, p in net.named_parameters() if not (p.dim() == 2 and not n.startswith("emb."))]
    return [{"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0}]


def _make_optimizer(net: VXOneNet, lr: float,
                    total_steps: int) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    opt = torch.optim.AdamW(_param_groups(net), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    warmup = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class _BatchLabels:
    """Light view of batch labels (slices or shards)."""

    def __init__(self, y_fam: torch.Tensor, y_bin: torch.Tensor,
                 y_sev: torch.Tensor | None = None,
                 ddos_mask: torch.Tensor | None = None,
                 y_ddos: torch.Tensor | None = None) -> None:
        self.y_fam, self.y_bin, self.y_sev = y_fam, y_bin, y_sev
        self.ddos_mask, self.y_ddos = ddos_mask, y_ddos


def _batch_losses(net: VXOneNet, run: RunData, u, txt, lab: _BatchLabels):
    """Loss per head + activity masks (§3.6). The ω-tensors live
on CPU (run is serializable); They are moved to the batch device here."""
    out = net(u, txt)
    dev = u.device
    fm = lab.y_fam >= 0  # sentinel -1 (A5-EMBER): outside the CE of choice
    losses = {
        "noul": noul_loss(out["noul"], lab.y_bin),
        "choice": choice_loss(out["choice"][fm], lab.y_fam[fm], run.omega.to(dev))
        if bool(fm.any()) else torch.zeros((), device=dev),
    }
    active = {"noul": True, "choice": bool(fm.any())}
    if net.has_score:
        losses["score"] = corn_loss(out["corn_logits"], lab.y_sev)
        active["score"] = True
    if net.has_ddos:
        m = lab.ddos_mask
        if bool(m.any()):
            losses["ddos"] = choice_loss(out["ddos"][m], lab.y_ddos[m],
                                         run.omega_ddos.to(dev))
            active["ddos"] = True
        else:
            active["ddos"] = False
    head = run.meta.get("a2_head")
    if head:  # A2: single active head (loss + M_val)
        losses = {head: losses[head]}
        active = {q: q == head for q in active}
    return losses, active


@torch.no_grad()
def _forward_numpy(net: VXOneNet, u: torch.Tensor, txt, lab, device: str) -> dict[str, np.ndarray]:
    """Forward in batches → numpy arrays for metrics/calibration (UNSW/CIC)."""
    outs: dict[str, list[np.ndarray]] = {}
    for i in range(0, u.shape[0], EVAL_BS):
        ub = u[i:i + EVAL_BS].to(device)
        tb = {k: v[i:i + EVAL_BS].to(device) for k, v in (txt or {}).items()}
        o = net(ub, tb if tb else None)
        for k, v in o.items():
            outs.setdefault(k, []).append(v.float().cpu().numpy())
    flat = {k: np.concatenate(v) for k, v in outs.items()}
    res: dict[str, np.ndarray] = {
        "y_bin": lab.y_bin.numpy(),
        "y_fam": lab.y_fam.numpy(),
        "noul_logits": flat["noul"],
        "choice_logits": flat["choice"],
    }
    if "corn_logits" in flat:
        res["corn_logits"] = flat["corn_logits"]
        res["y_sev"] = lab.y_sev.numpy()
    if "ddos" in flat:
        m = lab.ddos_mask.numpy()
        res.update(ddos_logits=flat["ddos"][m], y_ddos=lab.y_ddos.numpy()[m])
    return res


@torch.no_grad()
def _eval_ember(net: VXOneNet, run: RunData, batcher, subset: str, device: str) -> dict[str, np.ndarray]:
    """Forward in streaming (val/test EMBER) → numpy arrays."""
    acc: dict[str, list[np.ndarray]] = {}
    y_bin_l, y_fam_l = [], []
    for u_v, y_fam_v, y_bin_v in batcher.eval_batches(subset, EVAL_BS):
        o = net(u_v.to(device), None)
        for k, v in o.items():
            acc.setdefault(k, []).append(v.float().cpu().numpy())
        y_bin_l.append(y_bin_v.numpy())
        y_fam_l.append(y_fam_v.numpy())
    flat = {k: np.concatenate(v) for k, v in acc.items()}
    return {"noul_logits": flat["noul"], "choice_logits": flat["choice"],
            "y_bin": np.concatenate(y_bin_l), "y_fam": np.concatenate(y_fam_l)}


def _m_val(o: dict[str, np.ndarray], names: tuple[str, ...]) -> tuple[dict[str, float], float]:
    """M_val = average of active head metrics (§6.4). The rows with
y_fam = -1 (outside the candidate set, D12) do not participate in Choice."""
    p_mal = 1.0 / (1.0 + np.exp(-o["noul_logits"]))  # AUROC invariante a T
    fam_mask = o["y_fam"] >= 0
    fam_pred = np.argmax(o["choice_logits"], axis=1)
    m: dict[str, float] = {
        "noul": (roc_auc_score(o["y_bin"], p_mal)
                 if len(np.unique(o["y_bin"])) > 1 else 0.5),
        "choice": f1_score(o["y_fam"][fam_mask], fam_pred[fam_mask],
                           average="macro", zero_division=0),
    }
    if "corn_logits" in o:
        pi = corn_pmf(torch.tensor(o["corn_logits"]))[1].numpy()
        sev_hat = (pi * np.arange(4)).sum(axis=1)
        m["score"] = 1.0 - float(np.abs(sev_hat - o["y_sev"]).mean()) / 3.0
    if "y_ddos" in o:
        m["ddos"] = f1_score(o["y_ddos"], np.argmax(o["ddos_logits"], axis=1),
                             average="macro", zero_division=0)
    active = {k: m[k] for k in names if k in m}
    return active, float(np.mean(list(active.values())))


def train_run(run: RunData, out_path: str | Path, device: str = "cpu",
              amp: str = "auto", seed: int = SEED, epochs: int | None = None,
              batch_size: int | None = None, log=print,
              ckpt_path: str | Path | None = None,
              ckpt_every_steps: int = 2000, resume: bool = False,
              ablation: str | None = None) -> dict:
    """Train a run and write the artifact §8.2. Returns the summary.

Ablations E6 (§7): `ablation` in {a1, a2_*, a3, a4, a5, a6} — a single
change by run on the reference Small: a1 without textual branch (vocab a
None), a2 without multi-task (a2_noul / a2_choice / a2_score: one model per
head, the merge of the three dumps is the run a2), a3 without calibration
(T forced to 1), a4 ω≡1 (posterior with the
train tilt), a5 unknown off (τ_abs→1 in abstention; EMBER: the
candidate set is already passed without unknown), a6 dropout 0.

Recovery checkpointing (OOM/kill): every `ckpt_every_steps` steps and
At the close of each epoch, `ckpt_path` is written ATOMICALLY with
net+opt+sched+scaler+early-stop counters+rng. `resume=True` resume
from that checkpoint RESTARTING the current epoch (the permutation is
deterministic seed+epoch) — maximum loss <1 epoch; the runs that
They finish well, they delete the checkpoint. Exact determinism (fresh rng,
fixed seed) applies to the paper's single-shot reports; the summary is
a crash-recovery mechanism."""
    torch.manual_seed(seed)
    src = {"unsw": "unsw", "cic": "cic", "ember2024_v1": "ember"}[run.source]
    bs_def, ep_def = TRAIN_TABLE[run.scale][src]
    bs = batch_size or bs_def
    n_epochs = epochs or ep_def
    lr = LR_TABLE[run.scale]
    amp_mode = pick_amp(device, amp)

    # ── ablations E6 ──────────────────────────────────────────────────
    if ablation == "a1":   # without textual branch: the net does not see embeddings
        run.vocab = None
        for part in ("train_txt", "val_txt", "test_txt"):
            setattr(run, part, None)
    elif ablation == "a4":  # ω≡1: CE sin tilt, de-tilt identidad
        run.omega = torch.ones_like(run.omega)
        if run.omega_ddos is not None:
            run.omega_ddos = torch.ones_like(run.omega_ddos)
    elif ablation == "a5" and run.source == "ember2024_v1":
        # 10-way without unknown: the class leaves the candidate set; their ranks
        # (benign + without family + tail) are left out of the CE of choice via
        # imap sentinel -1 and _batch_losses mask
        from .losses import class_weights

        unk = run.candidates.index("unknown")
        counts = run.meta.get("train_counts", {})
        run.candidates = [c for c in run.candidates if c != "unknown"]
        run.omega = torch.tensor(class_weights([counts[c] for c in run.candidates]))
        # network is built from run.heads (dataclass frozen): replace it
        import dataclasses
        run.heads = dataclasses.replace(run.heads,
                                        n_candidates=len(run.candidates))
        for part in ("train_labels", "val_labels", "test_labels"):
            lab = getattr(run, part, None)
            if lab is None or lab.y_fam is None:
                continue
            yf = lab.y_fam
            lab.y_fam = torch.where(yf == unk, torch.full_like(yf, -1),
                                    torch.where(yf > unk, yf - 1, yf))
    elif ablation == "a6":  # dropout 0
        import src.models.network as _net_mod
        _net_mod.DROPOUT = 0.0
    elif ablation in ("a2_noul", "a2_choice", "a2_score"):
        # A2 without multi-task (§7): ONE model per head with the same
        # total budget — heads contribute <1% of params, width
        # no adjustment required. Only the chosen head receives loss and M_val;
        # the rest are left without a gradient and the combined dump excludes them.
        run.meta["a2_head"] = ablation.removeprefix("a2_")

    log(f"[train] {run.source}/{run.scale}/{run.split_scheme} bs={bs} epochs={n_epochs} "
        f"lr={lr} amp={amp_mode} K={run.heads.n_candidates} device={device}"
        + (f" ablation={ablation}" if ablation else ""))

    net = VXOneNet(src, run.scale, run.heads,
                   d_num=len(run.feature_stage.impute_median),
                   vocab_sizes={k: v.size for k, v in (run.vocab or {}).items()})
    net.to(device)
    kendall = KendallMultiTask(run.heads.names).to(device)

    ember = run.source == "ember2024_v1"
    if ember:
        from .data import EmberBatcher
        batcher = EmberBatcher(run)
        rows = [pq.ParquetFile(str(p)).metadata.num_rows for p in run.ember_train_shards]
        n_train = sum(rows)
        steps_per_epoch = sum(r // bs for r in rows)
    else:
        n_train = run.train.shape[0]
        steps_per_epoch = math.ceil(n_train / bs)

    total_steps = n_epochs * steps_per_epoch
    opt, sched = _make_optimizer(net, lr, total_steps)
    scaler = torch.amp.GradScaler(device, enabled=amp_mode == "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_mode)
    params = [p for g in opt.param_groups for p in g["params"]]

    # ── recovery checkpointing (atomic tmp→replace) ───────────
    def _save_ckpt(epoch_done: int, best_m: float, best_epoch: int,
                   bad: int, best_state) -> None:
        """epoch_done = number of COMPLETE epochs (summarizes and replays the current one)."""
        if not ckpt_path:
            return
        tmp = f"{ckpt_path}.tmp"
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net": {k: v.cpu() for k, v in net.state_dict().items()},
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "scaler": scaler.state_dict() if amp_mode == "fp16" else None,
            "amp": amp_mode, "epoch_done": epoch_done, "step": step,
            "best_m": best_m, "best_epoch": best_epoch, "bad": bad,
            "best_state": ({k: v.cpu() for k, v in best_state.items()}
                           if best_state else None),
            "torch_rng": torch.get_rng_state(),
        }, tmp)
        os.replace(tmp, ckpt_path)

    start_epoch, step = 0, 0
    best_m, best_epoch, bad, best_state = -1.0, -1, 0, None
    if resume and ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        for st in opt.state.values():  # AdamW: states to device of the params
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        # fresh sched: the current epoch is REPLAYED in full (permutation
        # deterministic seed+epoch), so last_epoch realigns with
        # e*steps_per_epoch and cosine ends in total_steps as single-shot.
        _, sched = _make_optimizer(net, lr, total_steps)
        if amp_mode == "fp16" and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"])
        best_m, best_epoch, bad = ck["best_m"], ck["best_epoch"], ck["bad"]
        best_state = ck.get("best_state")
        start_epoch = ck["epoch_done"]
        step = start_epoch * steps_per_epoch
        log(f"[train] RESUME desde {ckpt_path}: epochs completados "
            f"{start_epoch}/{n_epochs} M_val={best_m:.4f} (best ep {best_epoch})")

    def run_epoch(epoch: int) -> None:
        nonlocal step
        net.train()
        if ember:
            batches = ((u.to(device), None,
                        _BatchLabels(yf.to(device), yb.to(device)))
                       for u, yf, yb in batcher.train_batches(epoch, bs))
        else:
            order = np.random.default_rng(seed + epoch).permutation(n_train)
            txt_tr = run.train_txt or {}
            label_tr = run.train_labels

            def _gen():
                for s in range(0, n_train, bs):
                    idx = torch.tensor(order[s:s + bs])
                    u_b = run.train[idx].to(device)
                    txt_b = {k: v[idx].to(device) for k, v in txt_tr.items()} or None
                    yield u_b, txt_b, _BatchLabels(
                        label_tr.y_fam[idx].to(device),
                        label_tr.y_bin[idx].to(device),
                        label_tr.y_sev[idx].to(device)
                        if label_tr.y_sev is not None else None,
                        label_tr.ddos_mask[idx].to(device)
                        if label_tr.ddos_mask is not None else None,
                        label_tr.y_ddos[idx].to(device)
                        if label_tr.y_ddos is not None else None)
            batches = _gen()
        for u_b, txt_b, lab_b in batches:
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype is not None and device.startswith("cuda")):
                losses, active = _batch_losses(net, run, u_b, txt_b, lab_b)
                total, _lambdas = kendall(losses, active)
            opt.zero_grad(set_to_none=True)
            if amp_mode == "fp16":
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
                scaler.step(opt)
                scaler.update()
            else:
                total.backward()
                torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
                opt.step()
            sched.step()
            step += 1
            # checkpoint per step (every N): the current epoch is replayed at
            # resume, so epoch_done = epoch (NOT epoch+1)
            if ckpt_path and ckpt_every_steps and step % ckpt_every_steps == 0:
                _save_ckpt(epoch, best_m, best_epoch, bad, best_state)

    def val_outputs() -> dict[str, np.ndarray]:
        net.eval()
        if ember:
            return _eval_ember(net, run, batcher, "val", device)
        txt_va = {k: v.to(device) for k, v in (run.val_txt or {}).items()}
        return _forward_numpy(net, run.val.to(device), txt_va or None,
                              run.val_labels, device)

    t0 = time.time()
    for epoch in range(start_epoch, n_epochs):
        run_epoch(epoch)
        names = run.heads.names
        if run.meta.get("a2_head"):  # A2: early-stop solo sobre su cabeza
            names = (run.meta["a2_head"],)
        active_m, mv = _m_val(val_outputs(), names)
        marker = ""
        if mv > best_m:
            best_m, best_epoch, bad = mv, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
            marker = " ←"
        else:
            bad += 1
        log(f"[train] epoch {epoch + 1}/{n_epochs} M_val={mv:.4f} "
            + " ".join(f"{k}={v:.3f}" for k, v in active_m.items())
            + f" bad={bad}{marker}")
        if ckpt_path:  # end of epoch: epoch_done = epoch+1 (cheap, safe)
            _save_ckpt(epoch + 1, best_m, best_epoch, bad, best_state)
        if bad >= PATIENCE:
            log(f"[train] early stop (patience {PATIENCE}) en epoch {epoch + 1}")
            break

    # ── restore better epoch + calibration in val (§5) ──────────────
    net.load_state_dict(best_state)
    o = val_outputs()

    if ablation == "a3":  # without calibration: T=1 on all heads
        temperatures = {"noul": 1.0}
        if "corn_logits" in o:
            temperatures["score"] = 1.0
        if "y_ddos" in o:
            temperatures["ddos"] = 1.0
        temperatures["choice"] = 1.0
    else:
        temperatures = {"noul": fit_noul_temperature(
            torch.tensor(o["noul_logits"]), torch.tensor(o["y_bin"]))}
        if "corn_logits" in o:
            temperatures["score"] = fit_corn_temperature(
                torch.tensor(o["corn_logits"]), torch.tensor(o["y_sev"]))
        if "y_ddos" in o:
            temperatures["ddos"] = fit_choice_temperature(
                torch.tensor(o["ddos_logits"]), torch.tensor(o["y_ddos"]), run.omega_ddos)
        # Choice: only rows within the candidate set (y_fam ≥ 0, D12)
        fam_mask = o["y_fam"] >= 0
        temperatures["choice"] = fit_choice_temperature(
            torch.tensor(o["choice_logits"][fam_mask]),
            torch.tensor(o["y_fam"][fam_mask]), run.omega)

    run_id = f"{src}_{run.scale}_{run.split_scheme}"
    if ablation:
        run_id += f"_abl_{ablation}"
    # A5 (unknown off): τ_abs→1 ⇒ nadie abtiene (UNSW/CIC) o modo abstain
    # pure after removing class (EMBER); the candidate set without unknown already
    # left train_run in run.candidates.
    tau = 1.0 if ablation == "a5" else TAU_ABS
    save_artifact(out_path, net, run_meta={
        "source_key": SOURCE_KEY[src],
        "feature_stage": run.feature_stage,
        "vocab": run.vocab,
        "candidates": run.candidates,
        "omega": run.omega,
        "omega_ddos": run.omega_ddos,
        "ddos_candidates": run.ddos_candidates,
        "mode": "abstain" if ablation == "a5" else ("trained_unknown" if ember else "abstain"),
        "tau_abs": tau,
        "config": {"scale": run.scale, "split_scheme": run.split_scheme,
                   "batch_size": bs, "lr": lr, "amp": amp_mode, "seed": seed},
        "manifest": {
            "run_id": run_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seed": seed, "best_epoch": best_epoch, "m_val": best_m,
            "n_train": n_train, "candidates": run.candidates,
            "train_counts": run.meta.get("train_counts", {}),
            "exclusion": run.meta.get("exclusion", {}),
            "temperatures": temperatures,
            "wall_time_s": round(time.time() - t0, 1),
        },
    }, temperatures=temperatures, d_num=len(run.feature_stage.impute_median))
    log(f"[train] artefacto → {out_path} (best_epoch={best_epoch} M_val={best_m:.4f})")
    if ckpt_path:  # full run: recovery ckpt no longer needed
        Path(ckpt_path).unlink(missing_ok=True)
    return {"run_id": run_id, "best_epoch": best_epoch, "m_val": best_m,
            "temperatures": temperatures, "wall_time_s": time.time() - t0}
