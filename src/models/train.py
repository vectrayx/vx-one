"""Loop de entrenamiento VX-One (spec F2 §6.3-§6.5, D11).

Hiperparámetros congelados:
- AdamW β=(0.9,0.999), eps=1e-8; weight decay 1e-2 SOLO en matrices de peso
  Linear 2D (excluye LN, biases, embeddings, corn_b y s_q).
- Schedule: warmup lineal 5% + coseno hasta 0; clip de norma 1.0.
- Early stopping: patience 4 sobre M_val = media de métricas de cabezas
  activas (AUROC noul, macro-F1 choice sin abstención, 1−MAE(ŝ,y)/3 score,
  macro-F1 ddos en subconjunto).
- AMP: bf16 en L4, fp16 en T4, off en CPU.
- Al terminar: se restaura el mejor epoch, se calibra cada cabeza en val
  (§5, NLL) y se escribe el artefacto self-contained (§8.2).
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .calibration import (fit_choice_temperature, fit_corn_temperature,
                          fit_noul_temperature)
from .data import RunData
from .engine import SOURCE_KEY, TAU_ABS, save_artifact
from .losses import (KendallMultiTask, choice_loss, corn_expected, corn_loss,
                     corn_pmf, noul_loss)
from .network import VXOneNet

# Tabla congelada §6.3: {escala: {fuente: (batch_size, epochs)}}; lr por escala.
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
    """wd solo en matrices de peso Linear 2D (D11): LN/biases/embeddings fuera
    (corn_b y s_q son 1D/0D y quedan excluidos por la regla de dims)."""
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
    """Vista ligera de labels de un batch (slices o shards)."""

    def __init__(self, y_fam: torch.Tensor, y_bin: torch.Tensor,
                 y_sev: torch.Tensor | None = None,
                 ddos_mask: torch.Tensor | None = None,
                 y_ddos: torch.Tensor | None = None) -> None:
        self.y_fam, self.y_bin, self.y_sev = y_fam, y_bin, y_sev
        self.ddos_mask, self.y_ddos = ddos_mask, y_ddos


def _batch_losses(net: VXOneNet, run: RunData, u, txt, lab: _BatchLabels):
    """Loss por cabeza + máscaras de actividad (§3.6). Los tensores ω viven
    en CPU (run es serializable); se mueven al device del batch aquí."""
    out = net(u, txt)
    dev = u.device
    if "flat" in out:  # A7: CE de familia (mismo tilt omega) + CORN; sin gate
        fm = lab.y_fam >= 0
        losses = {}
        if bool(fm.any()):
            losses["choice"] = choice_loss(out["flat"][fm], lab.y_fam[fm],
                                           run.meta["a7_omega_train"].to(dev))
        if net.has_score:
            losses["score"] = corn_loss(out["corn_logits"], lab.y_sev)
        if net.has_ddos:
            m = lab.ddos_mask
            if bool(m.any()):
                losses["ddos"] = choice_loss(out["ddos"][m], lab.y_ddos[m],
                                             run.omega_ddos.to(dev))
        active = {k: True for k in losses}
        return losses, active
    fm = lab.y_fam >= 0  # sentinela -1 (A5-EMBER): fuera del CE de choice
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
    if head:  # A2: una sola cabeza activa (loss + M_val)
        losses = {head: losses[head]}
        active = {q: q == head for q in active}
    return losses, active


@torch.no_grad()
def _forward_numpy(net: VXOneNet, u: torch.Tensor, txt, lab, device: str) -> dict[str, np.ndarray]:
    """Forward en batches → arrays numpy para métricas/calibración (UNSW/CIC)."""
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
    """Forward en streaming (val/test EMBER) → arrays numpy."""
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
    """M_val = media de métricas de cabezas activas (§6.4). Las filas con
    y_fam = -1 (fuera del candidate set, D12) no participan en Choice."""
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
              ablation: str | None = None,
              oe: dict | None = None) -> dict:
    """Entrena una corrida y escribe el artefacto §8.2. Devuelve el resumen.

    oe (opcional, E9b): {"x": tensor [Noe, d+1] ya φ-transformado,
     "txt": dict[str, tensor]|None, "weight": float} — outlier exposure
     (Hendrycks et al. 2019) sobre la choice head: las filas del pool
     reciben target uniforme sobre los candidatos conocidos, con peso fijo
     fuera del weighting de Kendall. Solo UNSW/CIC (no EMBER).

    Ablations E6 (§7): `ablation` en {a1, a2_*, a3, a4, a5, a6, a7} — un solo
    cambio por corrida sobre la referencia Small: a1 sin rama textual (vocab a
    None), a2 sin multi-task (a2_noul / a2_choice / a2_score: un modelo por
    cabeza, el merge de los tres dumps es la corrida a2), a3 sin calibración
    (T forzada a 1), a4 ω≡1 (posterior con el
    tilt del train), a5 unknown off (τ_abs→1 en abstención; EMBER: el
    candidate set ya se pasa sin unknown), a6 dropout 0.

    Checkpointing de recuperación (OOM/kill): cada `ckpt_every_steps` pasos y
    al cierre de cada epoch se escribe ATÓMICAMENTE `ckpt_path` con
    net+opt+sched+scaler+contadores de early-stop+rng. `resume=True` retoma
    desde ese checkpoint REINICIANDO el epoch en curso (la permutación es
    determinista seed+epoch) — pérdida máxima <1 epoch; las corridas que
    terminan bien borran el checkpoint. El determinismo exacto (rng fresco,
    seed fijo) se aplica a los reportes single-shot del paper; el resume es
    un mecanismo de crash-recovery."""
    torch.manual_seed(seed)
    src = {"unsw": "unsw", "cic": "cic", "ember2024_v1": "ember"}[run.source]
    bs_def, ep_def = TRAIN_TABLE[run.scale][src]
    bs = batch_size or bs_def
    n_epochs = epochs or ep_def
    lr = LR_TABLE[run.scale]
    amp_mode = pick_amp(device, amp)

    # ── ablations E6 ──────────────────────────────────────────────────
    if ablation == "a1":   # sin rama textual: el net no ve embeddings
        run.vocab = None
        for part in ("train_txt", "val_txt", "test_txt"):
            setattr(run, part, None)
    elif ablation == "a4":  # ω≡1: CE sin tilt, de-tilt identidad
        run.omega = torch.ones_like(run.omega)
        if run.omega_ddos is not None:
            run.omega_ddos = torch.ones_like(run.omega_ddos)
    elif ablation == "a5" and run.source == "ember2024_v1":
        # 10-way sin unknown: la clase sale del candidate set; sus filas
        # (benignos + sin familia + cola) quedan fuera del CE de choice vía
        # el sentinela -1 del imap y la máscara de _batch_losses
        from .losses import class_weights

        unk = run.candidates.index("unknown")
        counts = run.meta.get("train_counts", {})
        run.candidates = [c for c in run.candidates if c != "unknown"]
        run.omega = torch.tensor(class_weights([counts[c] for c in run.candidates]))
        # la red se construye desde run.heads (dataclass frozen): reemplazarlo
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
    elif ablation == "a7":  # A7: una sola cabeza plana familia x severidad
        import dataclasses
        run.heads = dataclasses.replace(run.heads, flat=True)
        # Mismo tilt por familia que la referencia, replicado por severidad y
        # con de-tilt analitico dentro de la red; omega del artefacto = 1 para
        # que vx_calibrated no lo aplique dos veces.
        run.meta["a7_omega_train"] = run.omega.clone()
        run.omega = torch.ones_like(run.omega)
    elif ablation in ("a2_noul", "a2_choice", "a2_score"):
        # A2 sin multi-task (§7): UN modelo por cabeza con el mismo
        # presupuesto total — las cabezas aportan <1% de params, la anchura
        # no requiere ajuste. Solo la cabeza elegida recibe pérdida y M_val;
        # las demás quedan sin gradiente y el dump combinado las excluye.
        run.meta["a2_head"] = ablation.removeprefix("a2_")

    log(f"[train] {run.source}/{run.scale}/{run.split_scheme} bs={bs} epochs={n_epochs} "
        f"lr={lr} amp={amp_mode} K={run.heads.n_candidates} device={device}"
        + (f" ablation={ablation}" if ablation else ""))

    net = VXOneNet(src, run.scale, run.heads,
                   d_num=len(run.feature_stage.impute_median),
                   vocab_sizes={k: v.size for k, v in (run.vocab or {}).items()})
    net.to(device)
    if run.heads.flat:
        with torch.no_grad():
            net.flat_logw.copy_(torch.log(run.meta["a7_omega_train"]))
    flat_names = ("choice", "score", "ddos") if run.heads.ddos_candidates \
        else ("choice", "score")
    kendall = KendallMultiTask(flat_names if run.heads.flat
                               else run.heads.names).to(device)

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

    # ── checkpointing de recuperación (atomic tmp→replace) ───────────
    def _save_ckpt(epoch_done: int, best_m: float, best_epoch: int,
                   bad: int, best_state) -> None:
        """epoch_done = nº de epochs COMPLETOS (resume re-playea el actual)."""
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
        for st in opt.state.values():  # AdamW: estados a device de los params
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        # sched fresco: el epoch en curso se REPLAYEA completo (permutación
        # determinista seed+epoch), así last_epoch vuelve a alinear con
        # e*steps_per_epoch y el coseno termina en total_steps como single-shot.
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
            oe_x = oe["x"] if oe else None
            oe_txt_t = oe.get("txt") if oe else None
            oe_perm = (np.random.default_rng(seed + 1000 + epoch)
                       .permutation(oe_x.shape[0]) if oe_x is not None else None)
            oe_cursor = [0]

            def _next_oe(k: int):
                """k índices del pool OE, cíclicos con orden por-epoch."""
                if oe_x is None:
                    return None
                cur = oe_cursor[0]
                take = oe_perm[(cur + np.arange(k)) % len(oe_perm)]
                oe_cursor[0] = (cur + k) % len(oe_perm)
                return take

            def _gen():
                for s in range(0, n_train, bs):
                    idx = torch.tensor(order[s:s + bs])
                    u_b = run.train[idx].to(device)
                    txt_b = {k: v[idx].to(device) for k, v in txt_tr.items()} or None
                    oe_b = None
                    if oe_x is not None:
                        take = _next_oe(len(idx))
                        u_oe = oe_x[take].to(device)
                        txt_oe = ({k: v[take].to(device)
                                   for k, v in oe_txt_t.items()}
                                  if oe_txt_t else None)
                        oe_b = (u_oe, txt_oe)
                    yield u_b, txt_b, _BatchLabels(
                        label_tr.y_fam[idx].to(device),
                        label_tr.y_bin[idx].to(device),
                        label_tr.y_sev[idx].to(device)
                        if label_tr.y_sev is not None else None,
                        label_tr.ddos_mask[idx].to(device)
                        if label_tr.ddos_mask is not None else None,
                        label_tr.y_ddos[idx].to(device)
                        if label_tr.y_ddos is not None else None), oe_b
            batches = _gen()
        for u_b, txt_b, lab_b, oe_b in batches:
            with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                                enabled=amp_dtype is not None and device.startswith("cuda")):
                losses, active = _batch_losses(net, run, u_b, txt_b, lab_b)
                total, _lambdas = kendall(losses, active)
                if oe_b is not None and "choice" in losses:
                    u_oe, txt_oe = oe_b
                    out_oe = net(u_oe, txt_oe)
                    logp_oe = nn.functional.log_softmax(out_oe["choice"], dim=1)
                    # KL(uniform ‖ p): constante −log(1/K) + −mean_k log p_k
                    oe_loss = -logp_oe.mean(dim=1).mean()
                    total = total + oe["weight"] * oe_loss
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
            # checkpoint por paso (cada N): el epoch en curso se replayea al
            # retomar, así que epoch_done = epoch (NO epoch+1)
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
        names = (("noul", "choice", "score")
                 + (("ddos",) if run.heads.ddos_candidates else ())
                 if run.heads.flat else run.heads.names)
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
        if ckpt_path:  # fin de epoch: epoch_done = epoch+1 (barato, seguro)
            _save_ckpt(epoch + 1, best_m, best_epoch, bad, best_state)
        if bad >= PATIENCE:
            log(f"[train] early stop (patience {PATIENCE}) en epoch {epoch + 1}")
            break

    # ── restaurar mejor epoch + calibración en val (§5) ──────────────
    net.load_state_dict(best_state)
    o = val_outputs()

    if ablation == "a3":  # sin calibración: T=1 en todas las cabezas
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
        # Choice: solo filas dentro del candidate set (y_fam ≥ 0, D12)
        fam_mask = o["y_fam"] >= 0
        temperatures["choice"] = fit_choice_temperature(
            torch.tensor(o["choice_logits"][fam_mask]),
            torch.tensor(o["y_fam"][fam_mask]), run.omega)

    run_id = f"{src}_{run.scale}_{run.split_scheme}"
    if ablation:
        run_id += f"_abl_{ablation}"
    # A5 (unknown off): τ_abs→1 ⇒ nadie abtiene (UNSW/CIC) o modo abstain
    # puro tras quitar la clase (EMBER); el candidate set sin unknown ya lo
    # dejó train_run en run.candidates.
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
    if ckpt_path:  # corrida completa: el ckpt de recuperación ya no hace falta
        Path(ckpt_path).unlink(missing_ok=True)
    return {"run_id": run_id, "best_epoch": best_epoch, "m_val": best_m,
            "temperatures": temperatures, "wall_time_s": time.time() - t0}
