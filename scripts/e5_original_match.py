#!/usr/bin/env python
"""E5b-orig — Fase 1 CON payload real: cobertura por 5-tupla del día.

Problema de fondo (documentado): los Timestamp del parquet enhanced son
re-computados por el pipeline de flujos (no coinciden 1:1 con los del pcap
original; desfase no constante, modos a +10 min / −2 h / +10 h según día).
Un matching por ventana temporal (2 s / 5 s) como en la reconstrucción NO
es aplicable a los pcaps originales.

Protocolo defensible aquí (misma semántica de "cobertura de la zona gris"):
por día calendario D (filtrando las alertas del eve por fecha REAL del pcap;
los consolidados de cicresearch traen la semana completa), conjunto de
5-tuplas alertadas (bidireccionales) A_D; un flujo enhanced del día D está
"resuelto por Fase 1" si su 5-tupla ∈ A_D. El re-uso de puertos efímeros
dentro del día acota la resolución a nivel día y sesga el "benign flagged"
HACIA ARRIBA (se declara). Salida: results/e5_original/coverage_byday.*
Uso: PYTHONPATH=. python scripts/e5_original_match.py
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.e5.cascade import load_day  # noqa: E402

EVE_DIR = pathlib.Path("results/e5_original")
OUT_DIR = pathlib.Path("results/e5_original")
DAY_BIN = {"2017-07-03": "Monday-WorkingHours",
           "2017-07-04": "Tuesday-WorkingHours",
           "2017-07-05": "Wednesday-workingHours",
           "2017-07-06": "Thursday-WorkingHours",
           "2017-07-07": "Friday-WorkingHours"}


def parse_eve_ts(t: str):
    return datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")


def load_alerts_day(path: pathlib.Path, day: str) -> tuple[int, int, dict]:
    """(n_alerts_total, n_alerts_día, buckets 5-tupla→ts de alertas del día)"""
    day_num = day[8:10].lstrip("0")
    month_num = day[5:7].lstrip("0")
    buckets: dict[tuple, list] = {}
    n_total = n_day = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if '"event_type":"alert"' not in line:
                continue
            n_total += 1
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_eve_ts(j["timestamp"])
            if f"{ts.day}" != day_num or f"{ts.month}" != month_num:
                continue
            n_day += 1
            k1 = (j["src_ip"], int(j.get("src_port") or 0), j["dest_ip"],
                  int(j.get("dest_port") or 0), str(j.get("proto", "")).lower())
            k2 = (j["dest_ip"], int(j.get("dest_port") or 0), j["src_ip"],
                  int(j.get("src_port") or 0), str(j.get("proto", "")).lower())
            buckets.setdefault(k1, []).append(ts)
            buckets.setdefault(k2, []).append(ts)
    return n_total, n_day, buckets


def main() -> None:
    out = {}
    for day, binname in DAY_BIN.items():
        eve = EVE_DIR / f"eve_original_{binname}.json.gz"
        if not eve.exists():
            print(f"[orig] {day}: sin eve, skip")
            continue
        n_total, n_day, buckets = load_alerts_day(eve, day)
        flows = load_day(day)
        fams = np.array(flows["fam"])
        keys = list(zip(flows["src"], flows["sport"], flows["dst"],
                        flows["dport"], flows["proto"]))
        flag = np.array([k in buckets for k in keys], dtype=bool)
        y = fams != "normal"
        tp = int((flag & y).sum()); fp = int((flag & ~y).sum())
        fn = int((~flag & y).sum())
        per_class = {}
        for f in sorted(set(fams)):
            m = fams == f
            per_class[f] = {"n": int(m.sum()),
                            "resolved": round(float(flag[m].mean()), 4)}
        est = {"bin": binname, "n_alerts_total": n_total, "n_alerts_day": n_day,
               "n_flows": int(len(flag)), "n_attack": int(y.sum()),
               "attack_resolved": round(float(flag[y].mean()), 4) if y.sum() else 0.0,
               "benign_flagged": round(float(flag[~y].mean()), 5) if (~y).sum() else 0.0,
               "precision": round(tp / max(tp + fp, 1), 4),
               "recall": round(tp / max(tp + fn, 1), 4),
               "per_class": per_class}
        out[day] = est
        print(f"[orig] {day}: {n_day:,}/{n_total:,} alertas del día → "
              f"attack resolved {est['attack_resolved']}, benign flagged "
              f"{est['benign_flagged']}, P={est['precision']} R={est['recall']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "coverage_byday.json").write_text(json.dumps(out, indent=1))
    md = ["# E5-orig — Fase 1 con payload real (pcaps originales)", "",
          "Cobertura por 5-tupla del día (protocolo anti desfase-ts; ver docstring).",
          "", "| día | alertas(día/total) | attack resolved | benign flagged | P | R |",
          "|---|---|---|---|---|---|"]
    for d, v in out.items():
        md.append(f"| {d} | {v['n_alerts_day']:,}/{v['n_alerts_total']:,} | "
                  f"{v['attack_resolved']} | {v['benign_flagged']} | "
                  f"{v['precision']} | {v['recall']} |")
    md += ["", "## Por familia (resolved share)"]
    for d, v in out.items():
        md += [f"### {d}"]
        md += [f"- {f}: n={c['n']:,} resolved={c['resolved']}"
               for f, c in v["per_class"].items()]
    (OUT_DIR / "coverage_byday.md").write_text("\n".join(md) + "\n")
    print(f"[orig] → {OUT_DIR}/coverage_byday.md")


if __name__ == "__main__":
    main()
