#!/usr/bin/env python
"""E9 — per-window retraining (continue-train) bajo drift temporal CIC.

Cierra el pendiente que E8 deja declarado: la recalibración post-hoc no
puede reparar la discriminación (techo oracle ECE 0.445 con day-AUROC 0.56),
así que la pregunta medible es si el retraining con la ventana fresca SÍ
la recupera en el día de PortScan.

Protocolo congelado:
  * Checkpoint: results/artifacts/cic_small_time.pt (la corrida oficial
    by-day, seed 42; state_dict + temperatures + feature_stage contained).
  * Continue-train (streaming-honesto): train = split_time=='train' ∪
    filas del día de test 1 (Friday-DDos, labels retrasadas al cierre del
    día). FeatureStage / candidates / omega CONGELADOS del artifact (el
    redeploy continuo no re-deriva candidate set ni φ ni vocab; las filas
    nuevas solo alimentan el loss).
  * Optimización: AdamW (betas 0.9/0.999, eps 1e-8; wd 1e-2 solo en pesos
    2-D como D11), lr = lr_scale × lr_original constante, grad-clip 5.0.
    s_q de Kendall no viven en el artifact (era módulo externo) → parten de 0.
  * Calibración post-retrain: temperaturas NLL sobre la val del pipeline
    (como-run). El threshold FPR@95 se re-deriva del cuantil 0.95 de los
    positivos de la MISMA ventana de calibración (criterio repo).
  * Eval: día 1 (DDos, sanity), día 2 (PortScan). Métricas: AUROC,
    ECE15, Brier, FPR@95, macro-F1, ECE-ch + correct calib choice.

Salida: results/e9_retrain/{e9_retrain.json, cic_small_time_friday.npz}
Uso (CPU local, ~10–40 min según hardware):
    python scripts/e9_window_retrain.py
Opciones: --epochs 1 --lr-scale 0.15 --bs (por defecto el del artifact) --calib val|valday
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

# Convención repo: pool del sistema (fragmentación acotada bajo ulimit -v)
pa.set_memory_pool(pa.system_memory_pool())

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from src.eval.dump_outputs import _load_net, EVAL_BS  # noqa: E402
from src.models.feature_stage import FeatureStage    # noqa: E402
from src.models.losses import choice_loss, corn_loss, noul_loss  # noqa: E402
import e7_vendor_api as e7                            # noqa: E402
from e8_window_recal import temp_scale_binary, temp_scale_multi, sigmoid, soft_pred, ece_bin, ece_multi  # noqa: E402

PARQUET = REPO / "data/processed/cicids2017/cicids2017.parquet"
ARTIFACT = REPO / "results/artifacts/cic_small_time.pt"
OUT = REPO / "results/e9_retrain"
DDOS_DAY = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX"
PORTSCAN_DAY = "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr-scale", type=float, default=0.15)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--tag", default="",
                    help="sufijo de salida (sensitivity: ep2/ep4); '' = corrida principal")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)

    print("[1/6] artifact + parquet…", flush=True)
    net, ckpt, _ = _load_net(ARTIFACT, "cpu")
    stage = FeatureStage.from_dict(ckpt["feature_stage"])
    cands = ckpt["head_meta"]["choice"]["candidates"]
    imap = {c: i for i, c in enumerate(cands)}
    omega = torch.tensor(ckpt["head_meta"]["choice"]["class_weights"], dtype=torch.float32)
    cfg = ckpt["config"]
    bs = args.bs or int(cfg.get("batch_size") or 256)
    lr0 = float(cfg["lr"]) * args.lr_scale

    pt = pq.ParquetFile(PARQUET)
    def col(name, dtype=object):
        v = pt.read(columns=[name]).column(name).to_pylist()
        return np.asarray(v, dtype=dtype)
    xnum_arr = pt.read(columns=["x_num"]).column("x_num").combine_chunks()
    d_num = int(xnum_arr.type.list_size)
    assert d_num == stage.impute_median.shape[0], f"d artifact {stage.impute_median.shape[0]} ≠ parquet {d_num}"
    flat_x = np.asarray(
        xnum_arr.flatten().to_numpy(zero_copy_only=False), dtype=np.float32
    ).reshape(-1, d_num)
    del xnum_arr
    days = col("day"); fams = col("label_family"); sev = col("label_severity").astype(np.int64)
    split = col("split_time")
    total = len(days)
    print(f"  parquet {total:,} filas; d_num={d_num}", flush=True)

    tr_u = (split == "train") | (days == DDOS_DAY)
    va_m = split == "val"
    te_dd = (split == "test") & (days == DDOS_DAY)
    te_ps = (split == "test") & (days == PORTSCAN_DAY)
    print(f"  train∪day1 = {int(tr_u.sum()):,} | val = {int(va_m.sum()):,} | "
          f"te1 = {int(te_dd.sum()):,} | te2 = {int(te_ps.sum()):,}", flush=True)

    print("[2/6] transform stage congelado…", flush=True)

    def transform_chunked(X: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        """φ en bloques: evita 4×d float64 simultáneos (peak ≈ flat_x + out)."""
        out = np.empty((X.shape[0], X.shape[1] + 1), dtype=np.float32)
        for i0 in range(0, X.shape[0], chunk):
            out[i0:i0 + chunk] = stage.transform(X[i0:i0 + chunk])[0]
        return out

    u_tr = transform_chunked(flat_x[tr_u])
    u_va = transform_chunked(flat_x[va_m])
    u_dd = transform_chunked(flat_x[te_dd])
    u_ps = transform_chunked(flat_x[te_ps])
    del flat_x

    def lab(mask):
        fam = fams[mask]
        return torch.tensor([(f != "normal") for f in fam], dtype=torch.int64), \
               torch.tensor([imap.get(f, -1) for f in fam], dtype=torch.int64), \
               torch.tensor(sev[mask], dtype=torch.int64)
    yb_tr, yf_tr, ys_tr = lab(tr_u)
    yb_va, yf_va, _ = lab(va_m)
    yb_dd, yf_dd, ys_dd = lab(te_dd)
    yb_ps, yf_ps, ys_ps = lab(te_ps)
    n = u_tr.shape[0]

    print(f"[3/6] continue-train: {args.epochs} ep | lr={lr0:.3e} | bs={bs}", flush=True)
    net.train()
    decay = [p for nm_, p in net.named_parameters() if p.dim() == 2 and not nm_.startswith("emb.")]
    nodesc = [p for nm_, p in net.named_parameters() if not (p.dim() == 2 and not nm_.startswith("emb."))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2},
                             {"params": nodesc, "weight_decay": 0.0}],
                            lr=lr0, betas=(0.9, 0.999), eps=1e-8)
    t0 = time.time()
    for ep in range(args.epochs):
        order = torch.randperm(n, generator=torch.Generator().manual_seed(42)).numpy()
        tot_loss, nb = 0.0, 0
        for bi, i0 in enumerate(range(0, n, bs)):
            idx = order[i0:i0 + bs]
            ub = torch.from_numpy(u_tr[idx]).float()
            yb = yb_tr[torch.from_numpy(idx)]; yf = yf_tr[torch.from_numpy(idx)]; ys = ys_tr[torch.from_numpy(idx)]
            fm = yf >= 0
            out = net(ub, None)
            loss = noul_loss(out["noul"], yb)
            if bool(fm.any()):
                loss = loss + choice_loss(out["choice"][fm], yf[fm], omega)
            loss = loss + corn_loss(out["corn_logits"], ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss); nb += 1
            if bi % 2000 == 0:
                print(f"  ep{ep} step {bi}/{n//bs} loss={tot_loss/(bi+1):.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
        print(f"  ep{ep} fin: loss_medio={tot_loss/max(1,nb):.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)

    print("[4/6] calibración (temperaturas NLL)…", flush=True)
    net.eval()
    with torch.no_grad():
        def fwd(U):
            acc = {"noul": [], "choice": [], "corn": []}
            for i0 in range(0, U.shape[0], EVAL_BS):
                o = net(torch.from_numpy(U[i0:i0 + EVAL_BS]).float(), None)
                for k in ("noul", "choice"):
                    acc[k].append(o[k].float().numpy())
                acc["corn"].append(o["corn_logits"].float().numpy())
            return {k: np.concatenate(v) for k, v in acc.items()}
        vo = fwd(u_va)
        o1 = fwd(u_dd)
        o2 = fwd(u_ps)
        np.savez_compressed(OUT / f"cic_small_time_friday{args.tag}.npz",
                            te1_noul=o1["noul"], te1_choice=o1["choice"], te1_corn=o1["corn"],
                            te1_y_bin=yb_dd.numpy(), te1_y_fam=yf_dd.numpy(),
                            te2_noul=o2["noul"], te2_choice=o2["choice"], te2_corn=o2["corn"],
                            te2_y_bin=yb_ps.numpy(), te2_y_fam=yf_ps.numpy())

    # Dos ventanas de calibración (ambas medibles de la MISMA corrida):
    #  * asrun_val : val del pipeline (labels anteriores al día 1)
    #  * calib_valday : val ∪ día-1 retrasado (semántica walk de E8, comparable)
    wins = {
        "asrun_val": (vo["noul"], vo["choice"], yb_va.numpy(), yf_va.numpy()),
        "calib_valday": (np.concatenate([vo["noul"], o1["noul"]]),
                         np.concatenate([vo["choice"], o1["choice"]]),
                         np.concatenate([yb_va.numpy(), yb_dd.numpy()]),
                         np.concatenate([yf_va.numpy(), yf_dd.numpy()])),
    }
    temps = {k: (temp_scale_binary(ch, y), temp_scale_multi(c, f))
             for k, (ch, c, y, f) in wins.items()}

    print("[5/6] métricas día-1 / día-2…", flush=True)
    res = {"protocol": {"epochs": args.epochs, "lr_scale": args.lr_scale,
                        "lr": lr0, "bs": bs,
                        "temperatures": {k: {"T_noul": round(tn, 4), "T_choice": round(tc, 4)}
                                         for k, (tn, tc) in temps.items()},
                        "artifact": str(ARTIFACT), "seed": 42},
           "per_day": {}}
    for calib, (Tn, Tc) in temps.items():
        ch, c, cy, cf = wins[calib]
        for name, o, yb, yf, ys in (("day1_ddos", o1, yb_dd, yf_dd, ys_dd),
                                    ("day2_portscan", o2, yb_ps, yf_ps, ys_ps)):
            gp = sigmoid(o["noul"].astype(np.float64) / Tn).astype(np.float32)
            thr = float(np.quantile(sigmoid(ch.astype(np.float64) / Tn)[cy == 1], 0.95))
            fpr = float((gp[yb.numpy() == 0] >= thr).mean())
            pc = soft_pred(o["choice"], Tc)
            # CORN decode con T=1 (la calibración viaja en τ del gate/choice; el
            # score solo se usa para MAE sev, comparable con la corrida estática)
            ok = sigmoid(o["corn"].astype(np.float64))
            lvl = np.zeros((len(yb), 4))
            lvl[:, 0] = 1 - ok[:, 0]; lvl[:, 1] = ok[:, 0] - ok[:, 1]
            lvl[:, 2] = ok[:, 1] - ok[:, 2]; lvl[:, 3] = ok[:, 2]
            sev_est = (lvl * np.arange(4)).sum(1)
            res["per_day"][f"{name}[{calib}]"] = {
                "n": len(yb), "auroc": round(e7.auc_from(yb.numpy(), gp), 4),
                "ece15": round(ece_bin(gp, yb.numpy()), 4),
                "brier": round(float(((gp - yb.numpy()) ** 2).mean()), 4),
                "fpr@95": round(fpr, 4),
                "macro_f1": round(e7.macro_f1(yf.numpy(), pc.argmax(1), len(cands)), 4),
                "ece15_choice": round(ece_multi(pc, yf.numpy()), 4),
                "sev_mae": round(float(np.abs(sev_est - ys.numpy()).mean()), 4),
                "thr@0.95cal": round(thr, 4),
            }

    print("[6/6] tabla…", flush=True)
    lines = ["| día[calib] | n | AUROC | ECE15 | Brier | FPR@95 | macroF1 | ECE-ch | sevMAE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["per_day"].items():
        lines.append(f"| {name} | {r['n']} | {r['auroc']} | {r['ece15']} | {r['brier']} | "
                     f"{r['fpr@95']} | {r['macro_f1']} | {r['ece15_choice']} | {r['sev_mae']} |")
    md = "\n".join(lines)
    (OUT / f"e9_retrain{args.tag}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"e9_retrain{args.tag}.md").write_text(md + "\n")
    print(md, flush=True)
    print("RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
