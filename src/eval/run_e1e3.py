"""Evaluador E1/E3 de baselines + tabla comparativa contra VX-One.

Los npz de baselines guardan probs CRUDAS y las temperaturas van en el json
(mismo protocolo §5.3). Para Choice/ddos se aplica el MISMO de-tilt analítico
z − log ω que VX-One (mismos pesos de entrenamiento vía sample_weight ⇒ mismo
tilt ⇒ mismo de-tilt, Prop. 2 simétrica). La ω se toma del meta json de la
corrida VX correspondiente (mismos candidatos/conteos D12).

Uso:
    python -m src.eval.run_e1e3 --vx results/outputs --bl results/baselines \
        --out results/e1e3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import (_calibration_measures, _mc_conf_correct, _mc_nll_brier,
                      _softmax, _binary_metrics, noul_block, score_block,
                      choice_block, eval_vx_run)


def _baseline_calibrated(npz: dict, temps: dict, omega: np.ndarray,
                         omega_ddos: np.ndarray | None) -> dict:
    """probs crudas → crudas + calibradas (logits = log p para árboles).
    Kitsune solo aporta p_noul (Platt) y no p_choice."""
    out = {}
    eps = 1e-7
    p_n = np.clip(npz["p_noul"].astype(np.float64), eps, 1 - eps)
    z_n = np.log(p_n / (1 - p_n))
    out["p_noul_raw"], out["p_noul"] = p_n, 1.0 / (1.0 + np.exp(-z_n / temps["noul"]))
    if "p_choice" in npz:
        p_c = np.clip(npz["p_choice"].astype(np.float64), eps, 1.0)
        out["p_choice_raw"] = p_c / p_c.sum(1, keepdims=True)
        z_c = np.log(p_c) - np.log(omega)
        out["p_choice"] = _softmax(z_c / temps["choice"])
    if "p_score" in npz and temps.get("score"):
        p_s = np.clip(npz["p_score"].astype(np.float64), eps, 1.0)
        out["pi_raw"] = p_s / p_s.sum(1, keepdims=True)
        out["pi"] = _softmax(np.log(p_s) / temps["score"])
    if "p_ddos" in npz and temps.get("ddos") and omega_ddos is not None:
        p_d = np.clip(npz["p_ddos"].astype(np.float64), eps, 1.0)
        out["p_ddos_raw"] = p_d / p_d.sum(1, keepdims=True)
        out["p_ddos"] = _softmax((np.log(p_d) - np.log(omega_ddos)) / temps["ddos"])
    return out


def eval_baseline_run(npz_path: str | Path, vx_meta_path: str | Path) -> dict:
    npz = dict(np.load(npz_path, allow_pickle=True))
    meta = json.loads(Path(str(npz_path).replace(".npz", ".json")).read_text())
    vx_meta = json.loads(Path(vx_meta_path).read_text())
    omega = np.asarray(vx_meta["omega"], dtype=np.float64)
    omega_ddos = (np.asarray(vx_meta["omega_ddos"], dtype=np.float64)
                  if "omega_ddos" in vx_meta else None)
    cal = _baseline_calibrated(npz, meta["temperatures"], omega, omega_ddos)
    res: dict = {"model": meta["model"], "run_id": meta["run_id"]}
    # los baselines solo guardan test
    y_bin, y_fam = npz["y_bin"], npz["y_fam"]
    fam_mask = y_fam >= 0
    res["te"] = {
        "noul": noul_block(cal["p_noul_raw"], cal["p_noul"], y_bin),
    }
    if "p_choice" in cal:
        res["te"]["choice"] = choice_block(cal["p_choice_raw"][fam_mask],
                                           cal["p_choice"][fam_mask],
                                           y_fam[fam_mask],
                                           vx_meta.get("tau_abs", 0.5),
                                           candidates=meta.get("candidates"))
    if "pi" in cal and "y_sev" in npz:
        res["te"]["score"] = score_block(cal["pi"], npz["y_sev"])
    if "p_ddos" in cal:
        from sklearn.metrics import f1_score
        m = npz["ddos_mask"].astype(bool)
        res["te"]["ddos"] = {"macro_f1": float(f1_score(
            npz["y_ddos"], cal["p_ddos"][m].argmax(1),
            average="macro", zero_division=0))}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", default="results/outputs")
    ap.add_argument("--bl", default="results/baselines")
    ap.add_argument("--out", default="results/e1e3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {"vx": {}, "baselines": {}}
    for npz in sorted(Path(args.vx).glob("*.npz")):
        if npz.name.startswith("smoke"):
            continue
        results["vx"][npz.stem] = eval_vx_run(npz)
    vx_dir = Path(args.vx)
    for bl_dir in sorted(Path(args.bl).iterdir()):
        if not bl_dir.is_dir():
            continue
        # la ω/candidatos vienen de la corrida VX small correspondiente:
        # "unsw_oficial" → "unsw_small_oficial"; "unsw_loao_worms" →
        # "unsw_small_loao_worms" (los folds LOAO son Small)
        bl_name = bl_dir.name
        vx_name = bl_name.replace("_loao_", "_small_loao_") \
            if "_loao_" in bl_name else bl_name.replace("_", "_small_", 1)
        vx_meta = vx_dir / f"{vx_name}.json"
        if not vx_meta.exists():
            print(f"[e1e3] {bl_dir.name}: sin meta VX ({vx_meta}), skip")
            continue
        for npz in sorted(bl_dir.glob("*.npz")):
            key = f"{bl_dir.name}/{npz.stem}"
            results["baselines"][key] = eval_baseline_run(npz, vx_meta)

    (out / "e1e3.json").write_text(json.dumps(results, indent=1, default=str))
    _markdown(results, out / "e1e3.md")
    print(f"[e1e3] {len(results['vx'])} VX + {len(results['baselines'])} "
          f"baselines → {out/'e1e3.json'}")


def _fmt(v: float | None) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def _markdown(results: dict, path: Path) -> None:
    rows = []
    for run, r in results["vx"].items():
        te = r.get("te", r) if isinstance(r, dict) else r
        te = r.get("te") or r.get("va") or {}
        if not te:
            continue
        n, c = te.get("noul"), te.get("choice")
        rows.append((f"VX-One {run}", _fmt(n["e1"]["auroc"]) if n else "—",
                     _fmt(n["e1"]["fpr@95tpr"]) if n else "—",
                     _fmt(c["e1"]["macro_f1"]) if c else "—",
                     _fmt(n["e3"]["cal"]["ece"]) if n else "—",
                     _fmt(c["e3"]["cal"]["ece"]) if c else "—"))
    for key, r in results["baselines"].items():
        te = r.get("te", {})
        n, c = te.get("noul"), te.get("choice")
        rows.append((key, _fmt(n["e1"]["auroc"]) if n else "—",
                     _fmt(n["e1"]["fpr@95tpr"]) if n else "—",
                     _fmt(c["e1"]["macro_f1"]) if c else "—",
                     _fmt(n["e3"]["cal"]["ece"]) if n else "—",
                     _fmt(c["e3"]["cal"]["ece"]) if c else "—"))
    lines = [
        "| modelo | AUROC | FPR@95TPR | macro-F1 | ECE noul | ECE choice |",
        "|---|---|---|---|---|---|", *[f"| {a} | {b} | {c} | {d} | {e} | {f} |"
                                          for a, b, c, d, e, f in rows]]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
