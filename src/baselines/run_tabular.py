"""Runner de baselines tabulares (XGBoost, RF, MLP) sobre los MISMOS splits
y el MISMO preprocesado que VX-One (build_run), con calibración §5.3.

Uso:
    python -I -m src.baselines.run_tabular --source unsw --out results/baselines
    python -I -m src.baselines.run_tabular --source cic --scheme strat --out results/baselines
    python -I -m src.baselines.run_tabular --source unsw --exclude-family worms ...

Salidas por corrida (results/baselines/<run_id>/):
    xgboost.npz / rf.npz / mlp.npz   — probs crudas de test + labels
    xgboost.json / rf.json / mlp.json — temperaturas val, tiempos, M_val
Probs crudas + T ⇒ calibrated = f(raw, T) en eval (E1-E3 reportan ambos).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models.data import build_run

from .tabular import (_mlp_forward, _probs_to_logits, baseline_features,
                      calibrate_heads, fit_rf, fit_xgboost, predict_tree,
                      train_mlp)


def _compact(y: np.ndarray, n_full: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Compacta clases de train a contiguas cuando el split deja huecos
    (scheme time: clases que solo aparecen en val/test). XGBoost/RF exigen
    targets [0..K-1]. Devuelve (y_compacto, ids originales); None si ya es
    contiguo (caso común)."""
    present = np.unique(y)
    if np.array_equal(present, np.arange(int(y.max()) + 1)):
        return None
    remap = np.zeros(n_full, dtype=np.int64)
    remap[present] = np.arange(len(present))
    return remap[y], present


def _scatter(p: np.ndarray, cols: np.ndarray, n_full: int) -> np.ndarray:
    """Probs compactas → matriz [N, n_full] en las columnas originales."""
    if len(cols) == n_full:
        return p
    out = np.zeros((p.shape[0], n_full), dtype=p.dtype)
    out[:, cols] = p
    return out


def _remap_val(y_va: np.ndarray, cols: np.ndarray, n_full: int) -> np.ndarray:
    """y de val/test a espacio compacto; clases ausentes de train → -1
    (excluidas de f1/CE igual que las sin familia)."""
    if len(cols) == n_full:
        return y_va
    remap = np.full(n_full, -1, dtype=np.int64)
    remap[cols] = np.arange(len(cols))
    y = np.asarray(y_va)
    return np.where(y >= 0, remap[np.clip(y, 0, None)], -1)


def _collect_split(run, split: str, split_labels) -> dict[str, np.ndarray]:
    """Features + labels de un split en formato baselines."""
    x = {"train": run.train, "val": run.val, "test": run.test}[split].numpy()
    txt = {"train": run.train_txt, "val": run.val_txt, "test": run.test_txt}[split]
    # one-hot necesita las listas crudas, no los ids: re-encode desde vocab
    # (los tensores *_txt ya son ids; los one-hoteamos directo)
    blocks = [x.astype(np.float32)]
    if txt and run.vocab:
        for k, vocab in run.vocab.items():
            ids = txt[k].numpy()
            onehot = np.zeros((len(ids), vocab.size), dtype=np.float32)
            onehot[np.arange(len(ids)), ids] = 1.0
            blocks.append(onehot)
    lab = split_labels
    out = {"x": np.concatenate(blocks, axis=1),
           "y_bin": lab.y_bin.numpy(),
           "y_fam": lab.y_fam.numpy(),
           "fam_mask": lab.y_fam.numpy() >= 0,
           "y_sev": lab.y_sev.numpy() if lab.y_sev is not None else None}
    if lab.ddos_mask is not None:
        m = lab.ddos_mask.numpy()
        out["ddos_mask"] = m
        out["y_ddos"] = lab.y_ddos.numpy()[m]
    return out


def _collect_ember(run, bs: int = 32768) -> tuple[dict, dict, dict]:
    """EMBER: materializa u+labels desde los shards (RunData no materializa
    splits; el batcher yield-ea (u, y_fam, y_bin) ya transformados por φ —
    mismos inputs que el motor, sin one-hot textual)."""
    from ..models.data import EmberBatcher
    b = EmberBatcher(run)
    out = {}
    for subset in ("train", "val", "test"):
        us, yfs, ybs = [], [], []
        for u, yf, yb in b.eval_batches(subset, bs):
            us.append(u.numpy())
            yfs.append(yf.numpy())
            ybs.append(yb.numpy())
        d = {"x": np.concatenate(us).astype(np.float32),
             "y_bin": np.concatenate(ybs).astype(np.int64),
             "y_fam": np.concatenate(yfs).astype(np.int64)}
        d["fam_mask"] = d["y_fam"] >= 0
        d["y_sev"] = None
        out[subset] = d
        print(f"[baselines] ember/{subset}: n={len(d['x'])} "
              f"fam_mask={int(d['fam_mask'].sum())}")
    return out["train"], out["val"], out["test"]


def _probs_tree(models: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    cols = models.get("_cols", {})

    def sc(p: np.ndarray, head: str) -> np.ndarray:
        spec = cols.get(head)
        return p if spec is None else _scatter(p, spec[0], spec[1])

    out = {"p_noul": predict_tree(models["noul"], x, binary=True),
           "p_choice": sc(predict_tree(models["choice"], x, binary=False),
                          "choice")}
    if "score" in models:
        out["p_score"] = sc(predict_tree(models["score"], x, binary=False),
                            "score")
    if "ddos" in models:
        m = models["_ddos_mask"]
        p = np.zeros((len(x), len(models["ddos_classes"])), dtype=np.float64)
        p[m] = predict_tree(models["ddos"], x[m], binary=False)
        out["p_ddos"] = p
    return out


def _probs_mlp(net, net_ddos, x: np.ndarray, ddos_mask, n_ddos_classes: int,
               device: str, ddos_cols: np.ndarray | None = None
               ) -> dict[str, np.ndarray]:
    o = _mlp_forward(net, x, y_bin=np.zeros(len(x)), y_fam=np.zeros(len(x)),
                     y_sev=None, device=device)
    out = {"p_noul": 1.0 / (1.0 + np.exp(-np.clip(o["noul"], -30, 30))),
           "p_choice": np.exp(o["choice"] - o["choice"].max(1, keepdims=True))}
    out["p_choice"] /= out["p_choice"].sum(1, keepdims=True)
    if "score" in o:
        s = o["score"]
        p = np.exp(s - s.max(1, keepdims=True))
        out["p_score"] = p / p.sum(1, keepdims=True)
    if net_ddos is not None:
        od = _mlp_forward(net_ddos, x[ddos_mask], y_bin=np.zeros(int(ddos_mask.sum())),
                          y_fam=np.zeros(int(ddos_mask.sum())), y_sev=None,
                          device=device)
        pd = np.exp(od["choice"] - od["choice"].max(1, keepdims=True))
        pd = pd / pd.sum(1, keepdims=True)
        if ddos_cols is not None and len(ddos_cols) != n_ddos_classes:
            full = np.zeros((pd.shape[0], n_ddos_classes), dtype=pd.dtype)
            full[:, ddos_cols] = pd
            pd = full
        p = np.zeros((len(x), n_ddos_classes), dtype=np.float64)
        p[ddos_mask] = pd
        out["p_ddos"] = p
    return out


def run_baselines(source: str, scheme: str, out_dir: Path,
                  exclude_family: str | None = None,
                  device: str = "cpu", mlp_epochs: int = 30,
                  models: tuple[str, ...] = ("xgboost", "rf", "mlp"),
                  seed: int = 42) -> None:
    out_dir = Path(out_dir)
    run = build_run(source, "small", scheme, exclude_family=exclude_family)
    fam_part = f"_loao_{exclude_family}" if exclude_family else ""
    run_id = f"{source}_{scheme}{fam_part}"
    out = out_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    # EMBER no materializa splits (streaming por shards; el marcador es
    # torch.empty(0), NO None): colección aparte vía EmberBatcher
    if run.ember_train_shards is not None:
        tr, va, te = _collect_ember(run)
    else:
        tr = _collect_split(run, "train", run.train_labels)
        va = _collect_split(run, "val", run.val_labels)
        te = _collect_split(run, "test", run.test_labels)
    print(f"[baselines] {run_id}: K={len(run.candidates)} "
          f"n_tr={len(tr['x'])} n_va={len(va['x'])} n_te={len(te['x'])}")
    omega = run.omega.numpy()
    omega_ddos = run.omega_ddos.numpy() if run.omega_ddos is not None else None

    # calibración necesita las PROBS de val por modelo + labels comunes
    def val_dict(p: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        d = dict(p)
        d.update(y_bin=va["y_bin"], y_fam=va["y_fam"], fam_mask=va["fam_mask"],
                 y_sev=va["y_sev"])
        if "p_ddos" in d:
            # p_ddos viaja full-length; recortar a las filas de la máscara
            # ANTES de calibrar (gather tolera desalineados en silencio)
            d["p_ddos"] = d["p_ddos"][va["ddos_mask"]]
            d["y_ddos"] = va["y_ddos"]
        return d

    def save(name: str, cal, probs: dict[str, np.ndarray], extra: dict) -> None:
        np.savez_compressed(out / f"{name}.npz",
                            p_noul=probs["p_noul"].astype(np.float32),
                            p_choice=probs["p_choice"].astype(np.float32),
                            y_bin=te["y_bin"], y_fam=te["y_fam"],
                            fam_names=np.array(run.candidates, dtype=object),
                            **({"y_sev": te["y_sev"],
                                "p_score": probs["p_score"].astype(np.float32)}
                               if "p_score" in probs else {}),
                            **({"p_ddos": probs["p_ddos"].astype(np.float32),
                                "ddos_mask": te["ddos_mask"],
                                "y_ddos": te["y_ddos"],
                                "ddos_names": np.array(run.ddos_candidates, dtype=object)}
                               if "p_ddos" in probs else {}))
        (out / f"{name}.json").write_text(json.dumps({
            "run_id": run_id, "model": name, "candidates": run.candidates,
            "temperatures": {"noul": cal.noul_T, "choice": cal.choice_T,
                             "score": cal.score_T, "ddos": cal.ddos_T},
            **extra}, indent=2, default=str))
        print(f"[baselines] {run_id}/{name}: T_noul={cal.noul_T:.3f} "
              f"T_choice={cal.choice_T:.3f}"
              + (f" T_score={cal.score_T:.3f}" if cal.score_T else "")
              + (f" T_ddos={cal.ddos_T:.3f}" if cal.ddos_T else ""))

    has_score = run.heads.has_score
    ddos = run.ddos_candidates is not None

    # compactación de clases UNA vez (scheme time: huecos en train) +
    # filas de choice = solo con familia (EMBER tiene y_fam=-1 en train)
    n_full = len(run.candidates)
    fm_tr = tr["fam_mask"]
    y_ch_tr = tr["y_fam"][fm_tr]
    comp_c = _compact(y_ch_tr, n_full)
    cols_c = comp_c[1] if comp_c else np.arange(n_full)
    y_ch_fit = comp_c[0] if comp_c else y_ch_tr
    y_mlp_fit = _remap_val(tr["y_fam"], cols_c, n_full)  # −1 preservado
    if has_score:
        comp_s = _compact(tr["y_sev"], 4)
        y_s_fit, cols_s = comp_s if comp_s else (tr["y_sev"], np.arange(4))
    else:
        y_s_fit, cols_s = None, None
    if ddos:
        comp_d = _compact(tr["y_ddos"], len(run.ddos_candidates))
        y_d_fit, cols_d = comp_d if comp_d else (tr["y_ddos"],
                                                 np.arange(len(run.ddos_candidates)))
    # clases de val ausentes en train → fuera de la calibración
    fam_keep = np.isin(va["y_fam"], cols_c) if len(cols_c) != n_full else None
    sev_keep = (np.isin(va["y_sev"], cols_s)
                if has_score and len(cols_s) != 4 else None)

    if "xgboost" in models or "rf" in models:
        w_ch = omega[y_ch_tr]  # ω por fila sobre las filas con familia
        w_ch = w_ch * (len(w_ch) / w_ch.sum())
        w_sc = None  # score sin pesos (simétrico con CORN, D3)
        ch_rows = fm_tr if not bool(fm_tr.all()) else None
        specs = {"noul": (tr["y_bin"], None, None, 2, None),
                 "choice": (y_ch_fit, w_ch, cols_c, n_full, ch_rows)}
        if has_score:
            specs["score"] = (y_s_fit, w_sc, cols_s, 4, None)
        for name in [m for m in models if m in ("xgboost", "rf")]:
            t0 = time.time()
            fits, probs = {}, {}
            for head, (y, w, _c, _n, rows) in specs.items():
                task = "binary" if head == "noul" else "multiclass"
                x_fit = tr["x"] if rows is None else tr["x"][rows]
                fitter = fit_xgboost if name == "xgboost" else fit_rf
                fits[head] = fitter(x_fit, y, w, task, seed=seed)
            fits["_cols"] = {h: (c, _n) for h, (_y, _w, c, _n, _r) in specs.items()
                             if c is not None}
            if ddos:
                m = tr["ddos_mask"]
                w_d = omega_ddos[tr["y_ddos"]]
                w_d = w_d * (len(w_d) / w_d.sum())
                fitter = fit_xgboost if name == "xgboost" else fit_rf
                fits["ddos"] = fitter(tr["x"][m], y_d_fit, w_d, "multiclass",
                                      seed=seed)
                fits["ddos_classes"] = list(run.ddos_candidates)
                fits["_ddos_mask"] = va["ddos_mask"]
                if len(cols_d) != len(run.ddos_candidates):
                    fits["_cols"]["ddos"] = (cols_d, len(run.ddos_candidates))
            fit_s = time.time() - t0
            for split, arr in (("val", va), ("test", te)):
                if ddos:
                    fits["_ddos_mask"] = arr["ddos_mask"]
                p = _probs_tree(fits, arr["x"])
                if split == "val":
                    probs.update({f"val_{k}": v for k, v in p.items()})
                else:
                    probs.update(_probs_tree(fits, arr["x"]))
            cal = calibrate_heads(
                val_dict({k.removeprefix("val_"): v for k, v in probs.items()
                          if k.startswith("val_")}), run.omega,
                torch.tensor(omega_ddos) if omega_ddos is not None else None,
                fam_keep=fam_keep, sev_keep=sev_keep)
            test_probs = {k: v for k, v in probs.items() if not k.startswith("val_")}
            save(name, cal, test_probs, {"fit_s": round(fit_s, 1)})

    if "mlp" in models:
        t0 = time.time()
        om_c = omega[cols_c]  # ω acotada a las clases presentes
        yf_va_c = _remap_val(va["y_fam"], cols_c, n_full)
        net = train_mlp(tr["x"], tr["y_bin"], y_mlp_fit, om_c,
                        tr["y_sev"], va["x"], va["y_bin"], yf_va_c,
                        va["y_sev"], device=device, epochs=mlp_epochs,
                        seed=seed, log=print)
        net_ddos = None
        if ddos:
            m = tr["ddos_mask"]
            om_d = omega_ddos[cols_d]
            y_d_va = _remap_val(va["y_ddos"], cols_d, len(run.ddos_candidates))
            net_ddos = train_mlp(tr["x"][m], tr["y_bin"][m], y_d_fit,
                                 om_d, None, va["x"][va["ddos_mask"]],
                                 va["y_bin"][va["ddos_mask"]],
                                 y_d_va, None,
                                 device=device, epochs=mlp_epochs, width=128,
                                 seed=seed, log=print)
        fit_s = time.time() - t0
        va_ddos = va["ddos_mask"] if ddos else None
        te_ddos = te["ddos_mask"] if ddos else None
        n_ddos_cls = len(run.ddos_candidates) if ddos else 0
        dcols = cols_d if ddos else None
        probs = _probs_mlp(net, net_ddos, va["x"], va_ddos, n_ddos_cls, device,
                           ddos_cols=dcols)
        if len(cols_c) != n_full:
            probs["p_choice"] = _scatter(probs["p_choice"], cols_c, n_full)
        cal = calibrate_heads(val_dict(probs), run.omega,
                              torch.tensor(omega_ddos) if omega_ddos is not None else None,
                              fam_keep=fam_keep, sev_keep=sev_keep)
        test_probs = _probs_mlp(net, net_ddos, te["x"], te_ddos, n_ddos_cls, device,
                                ddos_cols=dcols)
        if len(cols_c) != n_full:
            test_probs["p_choice"] = _scatter(test_probs["p_choice"], cols_c, n_full)
        save("mlp", cal, test_probs, {"fit_s": round(fit_s, 1)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["unsw", "cic", "ember"])
    ap.add_argument("--scheme", default=None, help="oficial|strat|time")
    ap.add_argument("--exclude-family", default=None)
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--models", default="xgboost,rf,mlp")
    args = ap.parse_args()

    scheme = args.scheme or ("oficial" if args.source == "unsw"
                             else "strat" if args.source == "cic" else "temporal")
    run_baselines(args.source, scheme, Path(args.out),
                  exclude_family=args.exclude_family, device=args.device,
                  models=tuple(args.models.split(",")))


if __name__ == "__main__":
    main()
