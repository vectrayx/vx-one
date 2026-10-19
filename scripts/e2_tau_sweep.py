#!/usr/bin/env python3
"""E2-extensión: sweep de τ_abs (abstención del choice) sobre logits LOAO ya
computados — sin reentrenar. Produce:
  results/e2_rc/tau_sweep.json   (curvas por fold + agregados)
  results/e2_rc/tau_sweep.md     (tabla resumen)
  results/e2_rc/fig_unsw_tau.pdf
  results/e2_rc/fig_cic_tau.pdf
Figuras: tasa de atribución falsa (excluded→clase conocida) y abstención
sobre la familia excluida, vs τ; y risk–coverage sobre las familias vistas.

Semántica idéntica al evaluador del paper (src/eval/metrics.py):
  p_choice = softmax((z_choice - log ω) / T_choice); abstiene si max p < τ.
Uso:
  python3 e2_tau_sweep.py --outputs <dir con los npz/json> --out <dir salida>
"""
from __future__ import annotations
import argparse, glob, json, math, os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAUS = [round(t, 2) for t in np.arange(0.05, 0.995, 0.05)] + [0.99]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_pair(npz_path: str):
    j = json.load(open(npz_path[:-4] + ".json"))
    T = j["temperatures"]
    omega = np.asarray(j["omega"], dtype=np.float64)
    return np.load(npz_path, allow_pickle=True), j, T, omega


def calibrated_choice(npz, T, omega, prefix="te"):
    zc = npz[f"{prefix}_choice_logits"].astype(np.float64) - np.log(omega)
    return softmax(zc / T["choice"])


def risk_coverage(conf: np.ndarray, correct: np.ndarray, taus) -> dict:
    """risk-coverage estilo Geifman/El-Yaniv: orden por conf descendente.
    devolvemos por-τ: cobertura y riesgo-selectivo (sobre atribuidos)."""
    order = np.argsort(-conf)
    conf_s, corr_s = conf[order], correct[order]
    n = len(conf)
    out = []
    for tau in taus:
        sel = conf_s >= tau
        cov = float(sel.mean())
        rsel = float(1.0 - corr_s[sel].mean()) if sel.any() else 0.0
        out.append({"tau": tau, "coverage": cov, "risk_selected": rsel})
    # AURC (curva exacta por-punto, interp para todos los umbrales)
    cum_err = np.concatenate([[0], np.cumsum(1 - corr_s)])
    cover_axis = np.arange(1, n + 1)
    risk_curve =cum_err[1:] / cover_axis
    aurc = float(risk_curve.mean())
    return {"per_tau": out, "aurc": aurc}


def sweep_one(name: str, npz_path: str) -> dict:
    npz, j, T, omega = load_pair(npz_path)
    cand = list(j["candidates"])          # espacio visible al modelo
    excluded = j["exclude"]               # familia sacada de train
    fam_names = np.array([str(x) for x in npz["te_fam_names"]])
    y_fam = npz["te_y_fam"].astype(int)
    p = calibrated_choice(npz, T, omega, "te")
    conf = p.max(1)
    pred = p.argmax(1)

    # índices de la familia excluida en el test
    is_excl = np.array([n == excluded for n in fam_names])
    # clases "ataque conocida" (ni normal ni None)
    attack_known = [c for c in cand if c != "normal"]
    cand_arr = np.array(cand)

    res = {"fold": name, "exclude": excluded,
           "tau_abs_paper": j.get("tau_abs", 0.5),
           "temperatures": T, "candidates": cand, "n_test": int(len(conf))}

    # ── τ sweep sobre la familia excluida ────────────────────────────────
    excl_curve = []
    if is_excl.any():
        conf_e, pred_e = conf[is_excl], pred[is_excl]
        fam_e = y_fam[is_excl]
        for tau in TAUS:
            abst = conf_e < tau
            attr_known = ~abst & np.array([cand_arr[i] != "normal" for i in pred_e])
            attribution_normal = ~abst & (cand_arr[pred_e] == "normal")
            excl_curve.append({
                "tau": tau,
                "n_excl": int(is_excl.sum()),
                "abstain_rate": float(abst.mean()),
                "false_attack_rate": float(attr_known.mean()),
                "normal_absorption": float(attribution_normal.mean()),
                # cobertura útil: event-level honesty = se abstiene O lo
                # manda a normal DADO que la familia es la excluida
            })
        res["excluded_curve"] = excl_curve
        res["excluded_conf_stats"] = {
            "conf_p10": float(np.quantile(conf_e, 0.10)),
            "conf_p50": float(np.quantile(conf_e, 0.50)),
            "conf_p90": float(np.quantile(conf_e, 0.90)) if is_excl.any() else None,
        }
        # margen alternativo (max − 2º): ¿ayuda a rescatar CIC?
        if is_excl.any():
            ps = np.sort(p[is_excl], axis=1)
            margin_e = ps[:, -1] - ps[:, -2]
            res["excluded_margin_stats"] = {
                "margin_p10": float(np.quantile(margin_e, 0.10)),
                "margin_p50": float(np.quantile(margin_e, 0.50)),
                "margin_p90": float(np.quantile(margin_e, 0.90)),
            }

    # ── risk–coverage sobre familias vISTAS (las del vocab) ─────────────
    known_mask = np.array([n in attack_known for n in fam_names])
    if known_mask.any():
        gt_idx = np.array([cand.index(fam_names[i]) for i in np.where(known_mask)[0]])
        correct_k = pred[known_mask] == gt_idx
        conf_k = conf[known_mask]
        res["known_rc"] = risk_coverage(conf_k, correct_k, TAUS)
        # riesgo a conf≥0.5 (el default del paper) por comparación
        sel = conf_k >= 0.50
        res["known_risk_at_0.5"] = float(1 - correct_k[sel].mean()) if sel.any() else None
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="/home/jsantillana/projects/paper/results/outputs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "e2_rc")
    os.makedirs(out, exist_ok=True)

    pairs = sorted(glob.glob(os.path.join(args.outputs, "*loao*.npz")))
    pairs = [p for p in pairs if "_abl" not in p and "choice" not in p
             and "noul" not in p and "score" not in p]
    print(f"fold pairs: {len(pairs)}")

    summary = {"sweep": {}, "unsw_mean": {}, "cic_mean": {}, "generated_by":
               "scripts/e2_tau_sweep.py", "semantics": "de-tilt+T_choice, "
               "abstain iff max p_choice < tau (Chow λ_r=1−τ)"}
    for npz_path in pairs:
        base = os.path.basename(npz_path)[:-4]
        fold = sweep_one(base, npz_path)
        summary["sweep"][base] = fold
        label = "unsw" if base.startswith("unsw") else "cic"
        summary[f"{label}_mean"].setdefault("folds", []).append(base)

    # agrego por dataset: tasas medias de falsa atribución vs τ
    for label in ("unsw", "cic"):
        folds = [summary["sweep"][f] for f in summary[f"{label}_mean"]["folds"]]
        cur= []
        for tau in TAUS:
            row = {"tau": tau}
            for key, agg in (("abstain_rate", np.nanmean),
                             ("false_attack_rate", np.nanmean),
                             ("normal_absorption", np.nanmean)):
                vals = []
                for f in folds:
                    ec = f.get("excluded_curve")
                    if not ec:
                        vals.append(np.nan); continue
                    d = next(x for x in ec if abs(x["tau"] - tau) < 1e-9)
                    vals.append(d[key])
                row[key] = float(agg(vals))
            cur.append(row)
        summary[f"{label}_mean"]["curve"] = cur
        onef_mean = float(np.mean([f["known_rc"]["aurc"] for f in folds
                                   if "known_rc" in f]))
        summary[f"{label}_mean"]["aurc_known"] = onef_mean

    json.dump(summary, open(os.path.join(out, "tau_sweep.json"), "w"), indent=2)

    # ── figura 2 paneles: UNSW vs CIC ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, label in zip(axes, ("unsw", "cic")):
        cur = summary[f"{label}_mean"]["curve"]
        taus = [r["tau"] for r in cur]
        ax.plot(taus, [r["false_attack_rate"] * 100 for r in cur], "o-", ms=3,
                label="false attack attribution", color="tab:red")
        ax.plot(taus, [r["abstain_rate"] * 100 for r in cur], "s-", ms=3,
                label="abstention", color="tab:blue")
        ax.plot(taus, [r["normal_absorption"] * 100 for r in cur], "^-", ms=3,
                label="absorbed to normal", color="tab:green")
        ax.axvline(0.5, color="gray", ls=":", lw=1, label=r"$\tau_{abs}=0.5$")
        ax.set_title(label.upper())
        ax.set_xlabel(r"$\tau_{abs}$")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[0].set_ylabel("% of excluded-family test rows")
    fig.savefig(os.path.join(out, "fig_tau_excluded.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig_tau_excluded.png"), dpi=200, bbox_inches="tight")

    # ── markdown ────────────────────────────────────────────────────────
    with open(os.path.join(out, "tau_sweep.md"), "w") as f:
        f.write("# E2-ext: sweep τ_abs (de-tilt + T_choice) — logits LOAO\n\n")
        for label in ("unsw", "cic"):
            f.write(f"## {label.upper()} (media sobre {len(summary[label+'_mean']['folds'])} folds)\n\n")
            f.write("| τ | abstención % | falsa-atrib. ataque % | absorbido→normal % |\n|---|---|---|---|\n")
            for r in summary[f"{label}_mean"]["curve"]:
                f.write(f"| {r['tau']:.2f} | {r['abstain_rate']*100:.2f} | "
                        f"{r['false_attack_rate']*100:.2f} | "
                        f"{r['normal_absorption']*100:.2f} |\n")
            f.write(f"\nAURC (familias vistas): {summary[f'{label}_mean'].get('aurc_known', 'n/a'):.4f}\n\n")
        # detalle por fold clave: conf en familia excluida
        f.write("## Estadístico de confianza y margen en la familia excluida\n\n")
        f.write("| fold | excl | conf p50 | conf p90 | margin p50 | margin p90 |\n|---|---|---|---|---|---|\n")
        for k, v in summary["sweep"].items():
            st = v.get("excluded_conf_stats")
            if st:
                m = v.get("excluded_margin_stats", {})
                f.write(f"| {k} | {v['exclude']} | {st['conf_p50']:.4f} | {st['conf_p90']:.4f} "
                        f"| {m.get('margin_p50', float('nan')):.4f} | {m.get('margin_p90', float('nan')):.4f} |\n")
    print("OK ->", out)


if __name__ == "__main__":
    main()
