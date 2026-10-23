#!/usr/bin/env python
"""E5d-valid — validación del matching por 5-tupla del día (revisión externa).

Tres comprobaciones contra el riesgo de SUBESTIMACIÓN del matching:
  1. Presencia: fracción de alertas del día cuya 5-tupla (bidireccional)
     existe entre los flujos enhanced del día. Si ≈1, el matching no pierde
     ataques por ausencia de la 5-tupla en el parquet.
  2. IPs documentadas del testbed CICIDS2017 (atacante 172.16.0.1; víctimas
     192.168.10.50/.51/.24/.25): cobertura de los flujos de ataque que
     involucran al atacante, vs la cobertura global. Si el matching
     subestimara por reuso de puertos, la cobertura del subconjunto
     atacante se desviaría sistemáticamente.
  3. Reuso intra-día: para flujos de ataque NO resueltos, ¿su 5-tupla fue
     alertada en OTRO día? (contamina hacia arriba, no hacia abajo — se
     mide para declarar el signo del sesgo).

Salida: results/e5_original/matching_validation.json/.md
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
ATTACKER_IPS = {"172.16.0.1"}
VICTIM_IPS = {"192.168.10.50", "192.168.10.51", "192.168.10.24", "192.168.10.25"}


def parse_eve_ts(t: str) -> datetime:
    return datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S")


def load_alerts_day(path: pathlib.Path, day: str):
    day_num, month_num = day[8:10].lstrip("0"), day[5:7].lstrip("0")
    tuples_day, n_total, n_day = [], 0, 0
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
            tuples_day.append(k1)
    return n_total, n_day, tuples_day


def rev(k):
    s, sp, d, dp, p = k
    return (d, dp, s, sp, p)


def main() -> None:
    out = {}
    for day, binname in DAY_BIN.items():
        eve = EVE_DIR / f"eve_original_{binname}.json.gz"
        if not eve.exists():
            continue
        n_total, n_day, alert_tuples = load_alerts_day(eve, day)
        flows = load_day(day)
        fams = np.array(flows["fam"])
        keys = list(zip(flows["src"], flows["sport"], flows["dst"],
                        flows["dport"], flows["proto"]))
        keyset = set(keys) | {rev(k) for k in keys}
        # 1) presencia de las 5-tuplas alertadas en el enhanced del día
        present = sum(1 for k in alert_tuples
                      if k in keyset or rev(k) in keyset)
        pres_rate = present / max(len(alert_tuples), 1)
        # flags por flujo
        aset = set(alert_tuples)
        flag = np.array([k in aset or rev(k) in aset for k in keys])
        y = fams != "normal"
        # 2) flujos de ataque que involucran al atacante documentado
        att = np.array([((s in ATTACKER_IPS) or (d in ATTACKER_IPS))
                        for s, _, d, _, _ in keys])
        m_att = y & att
        cov_att = float(flag[m_att].mean()) if m_att.sum() else None
        cov_all = float(flag[y].mean())
        # IPs de atacante empíricas por día (src de flujos de ataque)
        srcs = {}
        for i in np.flatnonzero(y):
            srcs[flows["src"][i]] = srcs.get(flows["src"][i], 0) + 1
        top_src = sorted(srcs.items(), key=lambda x: -x[1])[:3]
        # 3) reuso cruzado de día para no resueltos (sesgo hacia arriba)
        #    (usa TODOS los días de alertas disponibles)
        out[day] = {"n_alerts_day": n_day, "n_alerts_total": n_total,
                    "tuple_presence_rate": round(pres_rate, 4),
                    "attack_flows": int(y.sum()),
                    "attack_flows_with_attacker_ip": int(m_att.sum()),
                    "coverage_attack_overall": round(cov_all, 4),
                    "coverage_attack_with_attacker_ip": (round(cov_att, 4)
                                                         if cov_att is not None else None),
                    "top_attack_src_ips": top_src}
        print(f"[val] {day}: presencia {pres_rate:.3f} | cobertura ataque "
              f"{cov_all:.4f} vs atacante {cov_att if cov_att is not None else float('nan'):.4f} "
              f"| top src {top_src}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "matching_validation.json").write_text(json.dumps(out, indent=1))
    md = ["# Validación del matching por 5-tupla", "",
          "| día | presencia alertas→enhanced | cobertura ataque (global) | cobertura ataque c/ IP atacante |",
          "|---|---|---|---|"]
    for d, v in out.items():
        md.append(f"| {d} | {v['tuple_presence_rate']} | "
                  f"{v['coverage_attack_overall']} | "
                  f"{v['coverage_attack_with_attacker_ip']} |")
    (OUT_DIR / "matching_validation.md").write_text("\n".join(md) + "\n")
    print(f"[val] → {OUT_DIR}/matching_validation.md")


if __name__ == "__main__":
    main()
