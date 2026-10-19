"""E5 (§4.4): harness de cascada Fase1 (Suricata) → Fase2 (VX-One).

NOTA DE DISEÑO (hallazgo empírico): el parquet enhanced NO es una copia
fila-a-fila de MLCVE — recomputa las estadísticas de flujo (misma secuencia
de captura, stats corregidas; la fila 0-1 coincide exacta, la fila 2 difiere
en 12 de 76 features, maxdiff 8e6 µs). No existe join fila-a-fila entre
ambos, así que la cascada es TWO-SAMPLE por día:

- Fase 1 (per-row REAL): flujos del día del parquet enhanced + alertas de
  eve.json casadas por 5-tupla + ventana temporal → cobertura de F1,
  composición de la zona gris (lo que F1 NO resuelve), latencia de alerta.
- Fase 2 (estimación insesgada): filas TEST del día (split_strat + day_idx
  del parquet processed) con las logits del npz → recall/FPR de F2 por
  familia sobre el mismo día.
- Combinado (estimación, supuesto documentado): P(F2 flag | familia, día)
  es independiente de "¿lo detectó F1?" dado el tipo de flujo — los flujos
  que F1 deja pasar son una muestra aleatoria de su familia (exacto para
  familias invisibles a F1 sin payload, p. ej. web: zona gris = clase
  completa; optimista para clases donde F1 falla los sigilosos). Se
  reporta como estimación con el supuesto explícito.

Métricas por día (§4.4): % resuelto por F1 por familia, composición de la
zona gris, accuracy/P/R combinado vs F1-solo vs F2-solo, latencia E2E
(F1: ts de la 1ª alerta − inicio de flujo; F2: p50 de E4, batch=1 CPU).

Uso:
    .venv/bin/python -m src.e5.cascade --day 2017-07-06
    .venv/bin/python -m src.e5.cascade --report      # tabla e5.md
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq

ENH = "data/raw/cicids2017/enhanced_flows/CICIDS_Flow.parquet"
PROCESSED = "data/processed/cicids2017/cicids2017.parquet"
NPZ = "results/outputs/cic_small_strat.npz"
OUT_DIR = pathlib.Path("results/e5")

# día calendario → day_idx de los CSVs MLCVE (SCHEMA §5; jueves = 2 CSVs,
# viernes = 3)
DAY_FILES = {"2017-07-03": [0], "2017-07-04": [1], "2017-07-05": [2],
             "2017-07-06": [3, 4], "2017-07-07": [5, 6, 7]}
# attack_label enhanced (crudo MLCVE) → familia nativa del registro
FAM_MAP = {"BENIGN": "normal", "DoS Hulk": "dos_ddos", "DDoS": "dos_ddos",
           "DoS GoldenEye": "dos_ddos", "DoS slowloris": "dos_ddos",
           "DoS Slowhttptest": "dos_ddos", "PortScan": "recon",
           "FTP-Patator": "bruteforce", "SSH-Patator": "bruteforce",
           "Bot": "botnet", "Infiltration": "infiltration",
           "Heartbleed": "exploitation",
           "Web Attack – Brute Force": "webattack",
           "Web Attack – XSS": "webattack",
           "Web Attack – Sql Injection": "webattack"}

_NON = ("flow_id", "source_ip", "source_port", "destination_ip",
        "destination_port", "Timestamp", "attack_label")
F1_T0_SLACK, F1_END_SLACK = 2.0, 5.0   # s alrededor de [inicio, inicio+dur]


def _norm_day(s: str) -> str:
    d, m, y = s.split("/")
    return f"{int(d):02d}-{int(m):02d}-{y}"


def _parse_enh_ts(t: str) -> float:
    return datetime.strptime(t if t.count(":") == 2 else t + ":00",
                             "%d/%m/%Y %H:%M:%S").timestamp()


def _parse_eve_ts(t: str) -> float:
    try:
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    except ValueError:
        return datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def load_alerts(eve_path: pathlib.Path) -> list[dict]:
    op = gzip.open if eve_path.suffix == ".gz" else open
    alerts = []
    with op(eve_path, "rt") as f:
        for line in f:
            try:
                j = json.loads(line)
            except json.JSONDecodeError:
                continue
            if j.get("event_type") != "alert":
                continue
            a = j
            alerts.append({
                "src": a["src_ip"], "sport": int(a.get("src_port") or 0),
                "dst": a["dest_ip"], "dport": int(a.get("dest_port") or 0),
                "proto": str(a.get("proto", "")).lower(),
                "ts": _parse_eve_ts(a["timestamp"]),
                "sig": a.get("signature", ""),
                "cat": a.get("category", ""),
                "sev": a.get("severity", -1)})
    return alerts


def load_day(day: str) -> dict:
    """Flujos del día (enhanced): 5-tupla, inicio, dur, familia mapeada."""
    pf = pq.ParquetFile(ENH)
    want = datetime.strptime(day, "%Y-%m-%d").strftime("%-d/%-m/%Y")
    idx, ts_s = [], []
    pos = 0
    for b in pf.iter_batches(batch_size=500_000, columns=["Timestamp"]):
        ts = np.asarray(b.column(0).to_pylist(), dtype=object)
        for k, t in enumerate(ts):
            if t.split(" ")[0].lstrip("0").replace("/0", "/") == want:
                idx.append(pos + k)
                ts_s.append(t)
        pos += len(ts)
    tbl = pq.read_table(ENH, columns=["source_ip", "source_port",
                                      "destination_ip", "destination_port",
                                      "protocol", "Flow Duration",
                                      "attack_label"]).take(idx)
    t0 = np.array([_parse_enh_ts(t) for t in ts_s])
    return {
        "idx": np.asarray(idx, dtype=np.int64), "t0": t0,
        "src": tbl.column("source_ip").to_pylist(),
        "sport": [int(p or 0) for p in tbl.column("source_port").to_pylist()],
        "dst": tbl.column("destination_ip").to_pylist(),
        "dport": [int(p or 0) for p in tbl.column("destination_port").to_pylist()],
        "proto": [str(p).lower() for p in tbl.column("protocol").to_pylist()],
        "dur": np.asarray(tbl.column("Flow Duration").to_pylist(),
                          dtype=np.float64) / 1e6,
        "fam": [FAM_MAP.get(l, "otros") for l in
                tbl.column("attack_label").to_pylist()]}


def match_f1(flows: dict, alerts: list[dict]) -> np.ndarray:
    """flag[i] = F1 alertó el flujo i; lat[i] = 1ª alerta − inicio (o nan)."""
    buckets: dict[tuple, list[float]] = {}
    for a in alerts:
        buckets.setdefault((a["src"], a["sport"], a["dst"], a["dport"],
                            a["proto"]), []).append(a["ts"])
    n = len(flows["t0"])
    flag = np.zeros(n, dtype=bool)
    lat = np.full(n, np.nan)
    for i in range(n):
        key = (flows["src"][i], flows["sport"][i], flows["dst"][i],
               flows["dport"][i], flows["proto"][i])
        ats = buckets.get(key)
        if not ats:
            continue
        end = flows["t0"][i] + min(flows["dur"][i], 3600.0)
        w = [t for t in ats
             if flows["t0"][i] - F1_T0_SLACK <= t <= end + F1_END_SLACK]
        if w:
            flag[i] = True
            lat[i] = min(w) - flows["t0"][i]
    return flag, lat


def load_f2(day: str) -> dict:
    """Filas test del día + probs Noul del npz (test_pos → fila del npz)."""
    d_idx = DAY_FILES[day]
    tbl = pq.read_table(PROCESSED, columns=["day_idx", "split_strat",
                                            "label_family"])
    dcol = np.asarray(tbl.column("day_idx").to_pylist())
    scol = np.asarray(tbl.column("split_strat").to_pylist())
    test_all = np.flatnonzero(scol == "test")   # filas test del parquet, orden parquet
    is_test_day = np.isin(scol, ["test"]) & np.isin(dcol, d_idx)
    rows = np.flatnonzero(is_test_day)
    # posición de cada fila test-del-día dentro del npz = rango en test_all
    npz_pos = np.searchsorted(test_all, rows)
    parts = np.load(NPZ, allow_pickle=True)
    meta = json.loads(pathlib.Path(NPZ.replace(".npz", ".json")).read_text())
    T = meta["temperatures"]["noul"]
    p_noul = 1.0 / (1.0 + np.exp(-parts["te_noul_logits"][npz_pos] / T))
    fam_names = meta["candidates"]   # te_fam_names es por fila, no vocab
    y_fam = parts["te_y_fam"][npz_pos]
    return {"fam": [fam_names[i] if i >= 0 else "otros" for i in y_fam],
            "flag": p_noul >= 0.5, "p": p_noul}


def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    n = tp + fp + fn + tn
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "accuracy": round((tp + tn) / n, 4) if n else 0.0}


def run_day(day: str, f2_lat_us: float) -> dict:
    alerts = load_alerts(OUT_DIR / f"eve_{day}.json.gz")
    flows = load_day(day)
    f1_flag, f1_lat = match_f1(flows, alerts)
    n = len(f1_flag)
    y = np.array([f != "normal" for f in flows["fam"]])
    print(f"[e5] {day}: {n:,} flujos ({int(y.sum()):,} ataque), "
          f"{len(alerts):,} alertas → F1 flag {int(f1_flag.sum()):,}")

    f2 = load_f2(day)
    fams = sorted(set(flows["fam"]))

    # ── cobertura F1 por familia (per-row real, día completo) ──
    per_class = {}
    for f in fams:
        m = np.array(flows["fam"]) == f
        per_class[f] = {"n": int(m.sum()),
                        "f1_resolved": round(float(f1_flag[m].mean()), 4)}

    # ── zona gris = flujos que F1 NO resolvió (día completo) ──
    gz = ~f1_flag
    gz_comp = {f: int((gz & (np.array(flows["fam"]) == f)).sum())
               for f in fams if int((gz & (np.array(flows["fam"]) == f)).sum())}

    # ── F2 sobre filas test del día: recall por familia + FPR ──
    f2_fam = np.array(f2["fam"])
    f2_flag = f2["flag"]
    y2 = f2_fam != "normal"
    f2_recall = {f: round(float(f2_flag[f2_fam == f].mean()), 4)
                 for f in fams if (f2_fam == f).sum() >= 10}
    f2_fpr = round(float(f2_flag[~y2].mean()), 4) if (~y2).sum() else 0.0
    f2_acc = prf(int((f2_flag & y2).sum()), int((f2_flag & ~y2).sum()),
                 int((~f2_flag & y2).sum()), int((~f2_flag & ~y2).sum()))

    # ── combinado (estimación por familia): zona gris → F2 ──
    # TP = ataques resueltos por F1 + zona gris de cada FAMILIA DE ATAQUE
    # (la de 'normal' aporta al FP, no al TP)
    gz_benign = int((gz & ~y).sum())
    est_tp = int(f1_flag[y].sum()) + sum(
        n_att * f2_recall.get(f, 0.0) for f, n_att in gz_comp.items()
        if f != "normal")
    est_fp = int(f1_flag[~y].sum()) + gz_benign * f2_fpr
    fn = int(y.sum()) - est_tp
    tn = int((~y).sum()) - est_fp
    combined = prf(int(round(est_tp)), int(round(est_fp)),
                   int(round(fn)), int(round(tn)))
    f1_only = prf(int(f1_flag[y].sum()), int(f1_flag[~y].sum()),
                  int((y & ~f1_flag).sum()), int((~y & ~f1_flag).sum()))

    lat_f1 = f1_lat[f1_flag & ~np.isnan(f1_lat)]
    res = {
        "day": day, "n_flows": n, "n_attack": int(y.sum()),
        "n_alerts": len(alerts),
        "f1": {"resolved": round(float(f1_flag.mean()), 4), **f1_only,
               "latency_s": {"median": round(float(np.median(lat_f1)), 3),
                             "p90": round(float(np.percentile(lat_f1, 90)), 3)}
               if len(lat_f1) else None},
        "f2": {"n_test": int(len(f2_flag)), "recall_by_fam": f2_recall,
               "fpr_benign": f2_fpr, "accuracy": f2_acc["accuracy"],
               "f1": f2_acc["f1"],
               "latency_us_p50": round(f2_lat_us, 1)},
        "gray_zone": {"n": int(gz.sum()),
                      "share_of_day": round(float(gz.mean()), 4),
                      "composition": gz_comp},
        "combined_estimate": combined,
        "per_class": per_class,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"cascade_{day}.json").write_text(json.dumps(res, indent=2))
    print(f"[e5] {day}: F1 acc={f1_only['accuracy']} "
          f"combinado acc={combined['accuracy']} "
          f"zona gris={res['gray_zone']['share_of_day']:.1%}")
    return res


def e4_latency() -> float:
    for p in sorted(pathlib.Path("results/e4").glob("e4_cic*.json")):
        return json.loads(p.read_text())["p50_us"]
    return 955.2  # unsw small (única medida E4 disponible)


def report() -> None:
    rows = [json.loads(p.read_text())
            for p in sorted(OUT_DIR.glob("cascade_*.json"))]
    if not rows:
        print("[e5] sin cascade_*.json"); return
    lines = ["# E5 — Cascada Fase1 (Suricata+ET) → Fase2 (VX-One)", "",
             "| día | flujos | ataque | F1 acc | F2 acc (test día) | combinado "
             "acc | F1 P | F1 R | comb P | comb R | zona gris |", "|---|---|---|"
             "---|---|---|---|---|---|---|---|"]
    for r in rows:
        f2 = r["f2"]
        lines.append(
            f"| {r['day']} | {r['n_flows']:,} | {r['n_attack']:,} "
            f"| {r['f1']['accuracy']:.4f} | {f2.get('accuracy', '—')} | "
            f"{r['combined_estimate']['accuracy']:.4f} "
            f"| {r['f1']['precision']:.4f} | {r['f1']['recall']:.4f} "
            f"| {r['combined_estimate']['precision']:.4f} "
            f"| {r['combined_estimate']['recall']:.4f} "
            f"| {r['gray_zone']['share_of_day']:.1%} |")
    (OUT_DIR / "e5.md").write_text("\n".join(lines) + "\n")
    print(f"[e5] reporte: {OUT_DIR}/e5.md ({len(rows)} días)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="2017-07-03 .. 2017-07-07")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if not args.day:
        ap.error("--day o --report")
    run_day(args.day, e4_latency())


if __name__ == "__main__":
    main()
