#!/usr/bin/env python
"""E5b — curvas risk–coverage del gating compuesto de la cascada.

Cierra el next-step (ii) de la conclusión: risk–coverage de la propia
adjudicación de la cascada Fase1 (Suricata) → Fase2 (VX-One gate).

Semántica de adjudicación (selectiva, escalación):
  A(τ) = {F1 alertó} ∪ {p_noul ≥ τ}    (verédito = ataque)
  abstain = lo no adjudicado (escala a otro mecanismo)
  coverage(τ) = |A|/N ; risk(τ) = 1 − precision de A (FPs normal/|A|)

Tres bloques por día, cada uno con su exactitud declarada:
  1. F1-only (EXACTO por flujo): flag de eve.json casada por 5-tupla + ventana.
  2. F2-solo τ-sweep (EXACTO por fila test): probs Noul del npz strat.
  3. COMPUESTO (estimación, supuesto documentado E5: "los flujos que F1
     deja pasar son muestra aleatoria de su familia"): cov(τ) y risk(τ) por
     clase binaria con f1_c exacto y f2_c(τ) exacto en test.

Salida: results/e5_rc/{e5b_risk_coverage_<day>.json, e5b.md, e5b_curves.npz}
Uso: /tmp/opencode/venv/bin/python scripts/e5_risk_coverage.py [--days ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import src.e5.cascade as e5  # noqa: E402

# Overrides de rutas (el repo original vive en projects/paper)
e5.ENH = Path("${HOME}/projects/paper/data/raw/cicids2017/enhanced_flows/CICIDS_Flow.parquet")
e5.PROCESSED = Path("${HOME}/projects/paper/data/processed/cicids2017/cicids2017.parquet")
e5.NPZ = str(REPO / "results/outputs_seed/cic_small_strat.npz")
e5.OUT_DIR = REPO / "results/e5"   # eve_*.json.gz descargados del volumen

DAYS = ["2017-07-03", "2017-07-04", "2017-07-05", "2017-07-06", "2017-07-07"]
OUT = REPO / "results/e5_rc"
GRID = np.linspace(0.0, 1.0, 201)


def curves(p: np.ndarray, y: np.ndarray) -> list[dict]:
    """τ-sweep exacto de la adjudicación p≥τ. y=1 ataque."""
    cov, risk, rec = [], [], []
    for τ in GRID:
        m = p >= τ
        n = max(1, int(m.sum()))
        precision = float(y[m].mean()) if m.any() else 0.0
        cov.append(m.mean())
        risk.append(1.0 - precision)
        rec.append(float(y[m].sum() / max(1, y.sum())))
    return [{"tau": float(t), "cov": float(c), "risk": float(r), "recall": float(rc)}
            for t, c, r, rc in zip(GRID, cov, risk, rec)]


def composed(fam_flows: list[str], flag1: np.ndarray, p: np.ndarray,
             fam_test: list[str]) -> dict:
    """Estimación compuesta (supuesto E5) con f1_c exacto y f2_c(τ) exacto."""
    fams_f, fams_t = np.asarray(fam_flows), np.asarray(fam_test)
    attack_f, attack_t = fams_f != "normal", fams_t != "normal"
    prev_a = float(attack_f.mean())
    f1_a = float(flag1[attack_f].mean()) if attack_f.any() else 0.0
    f1_n = float(flag1[~attack_f].mean()) if (~attack_f).any() else 0.0
    out = []
    for τ in GRID:
        sel_t = p >= τ
        # f2_c(τ) exacto por clase en test
        f2_a = float(sel_t[attack_t].mean()) if attack_t.any() else 0.0
        f2_n = float(sel_t[~attack_t].mean()) if (~attack_t).any() else 0.0
        # adjudicado: atacante si f1_a ∨ gray∧f2_a; normal idem (FP si dispara)
        cov_a = f1_a + (1.0 - f1_a) * f2_a   # P(adjudicar|attack)
        cov_n = f1_n + (1.0 - f1_n) * f2_n   # P(adjudicar|normal) = FP rate
        # todo lo adjudicado se etiqueta "ataque": los normal adjudicados son error
        cov = prev_a * cov_a + (1.0 - prev_a) * cov_n
        risk = ((1.0 - prev_a) * cov_n) / cov if cov > 0 else 0.0
        recall = cov_a  # TP rate de ataques = P(adjudicar|attack)
        out.append({"tau": float(τ), "cov": cov, "risk": risk, "recall": recall,
                    "cov_attack": cov_a, "fpr_normal": cov_n})
    return out


def cov_at_risk(crv: list[dict], rmax: float) -> float:
    ok = [c["cov"] for c in crv if c["risk"] <= rmax]
    return round(max(ok), 4) if ok else 0.0


def summarize(crv: list[dict]) -> dict:
    return {"cov@risk<=0.05": cov_at_risk(crv, 0.05),
            "cov@risk<=0.10": cov_at_risk(crv, 0.10),
            "cov@risk<=0.20": cov_at_risk(crv, 0.20),
            "max_cov@risk<=0.30": cov_at_risk(crv, 0.30)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=DAYS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows, curves_npz = [], {}
    for day in args.days:
        print(f"\n=== {day} ===", flush=True)
        flows = e5.load_day(day)
        alerts = e5.load_alerts(e5.OUT_DIR / f"eve_{day}.json.gz")
        flag1, lat1 = e5.match_f1(flows, alerts)
        fam_flows = flows["fam"]
        attack_f = np.asarray([f != "normal" for f in fam_flows])
        f2 = e5.load_f2(day)
        p = np.asarray(f2["p"], dtype=np.float64)
        attack_t = np.asarray([f != "normal" for f in f2["fam"]])

        # 1. F1-only exacto
        n_A1 = int(flag1.sum())
        f1_point = {"cov": round(n_A1 / len(flag1), 6),
                    "risk": round(1.0 - float(attack_f[flag1].mean()), 6) if n_A1 else None,
                    "precision": round(float(attack_f[flag1].mean()), 4) if n_A1 else None,
                    "recall": round(float(flag1[attack_f].mean()), 4),
                    "n_flag": n_A1, "n_flows": len(flag1)}

        # 2. F2 τ-sweep exacto (test rows)
        c_f2 = curves(p, attack_t.astype(np.int64))

        # 3. Compuesto (estimación E5)
        c_cmp = composed(fam_flows, flag1, p, f2["fam"])

        res = {"day": day,
               "f1_only_exact": f1_point,
               "f2_sweep_exact": {"summaries": summarize(c_f2),
                                  "curve": c_f2},
               "composed_estimate": {"summaries": summarize(c_cmp),
                                     "curve": c_cmp,
                                     "note": "supuesto E5: flujos no resueltos por F1 = muestra aleatoria de su clase"},
               "n_flows_f1": len(flag1), "n_flows_f2": len(p),
               "alert_latency_p50": round(float(np.nanpercentile(lat1[flag1], 50)), 3) if n_A1 else None}
        (OUT / f"e5b_risk_coverage_{day}.json").write_text(json.dumps(res, indent=1))
        curves_npz[f"{day}_tau_f2"] = GRID
        curves_npz[f"{day}_cov_f2"] = np.asarray([c["cov"] for c in c_f2])
        curves_npz[f"{day}_risk_f2"] = np.asarray([c["risk"] for c in c_f2])
        curves_npz[f"{day}_cov_cmp"] = np.asarray([c["cov"] for c in c_cmp])
        curves_npz[f"{day}_risk_cmp"] = np.asarray([c["risk"] for c in c_cmp])
        all_rows.append((day, f1_point, res["f2_sweep_exact"]["summaries"],
                         res["composed_estimate"]["summaries"]))
        print(f"  F1 exact: cov={f1_point['cov']:.4f} prec={f1_point['precision']}", flush=True)
        print(f"  F2   @r≤10%: cov={res['f2_sweep_exact']['summaries']['cov@risk<=0.10']}", flush=True)
        print(f"  COMP @r≤10%: cov={res['composed_estimate']['summaries']['cov@risk<=0.10']}", flush=True)

    np.savez_compressed(OUT / "e5b_curves.npz", **curves_npz)

    lines = ["# E5b — risk–coverage del gating compuesto (F1 exacto ∪ F2 con τ)",
             "",
             "| día | F1 cov | F1 precision | F2 cov@r≤5% | F2 cov@r≤10% | comp cov@r≤5% | comp cov@r≤10% |",
             "|---|---|---|---|---|---|---|"]
    for day, f1p, s2, s3 in all_rows:
        lines.append(f"| {day} | {f1p['cov']} | {f1p['precision']} | "
                     f"{s2['cov@risk<=0.05']} | {s2['cov@risk<=0.10']} | "
                     f"{s3['cov@risk<=0.05']} | {s3['cov@risk<=0.10']} |")
    md = "\n".join(lines) + "\n"
    (OUT / "e5b.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
