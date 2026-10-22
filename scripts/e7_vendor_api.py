#!/usr/bin/env python
"""E7 — vendor System One: Microsoft-Decision-1 / TypeSafe Jev vs VX-One.

Protocolo (EXPERIMENTS.md §9): zero-shot del modelo comercial vía API sobre
una muestra estratificada del test UNSW. Tres preguntas por request (una de
cada primitiva, MISMO state): gate (noul ¿es ataque?), choice (familia sobre
el MISMO candidate set congelado del run), score (rubrik de severidad 0-3).
Compara contra nuestros logits del npz oficial en las mismas filas.

Vendor:
- typesafe:  POST https://api.typesafe.ai/v1/systemone  (Bearer TYPESAFE_API_KEY,
             modelo jev-1.13.0 = alias estable en 2026-10; $42/Btok input>$0).
- opencode:  MISMA API System One pero vía el proxy/opencode:
             POST https://opencode.ai/zen/v1/systemone con OPENCODE_API_KEY
             (oc_sk_...) y modelo jev-1.13-free — costo $0 (tier free de
             OpenCode passthrough al Jev 1.13 real; verificado 2026-10-20).
- azure:     DESCARTADA por decisión del usuario (2026-10-20): la suscripción
             Azure está suspendida por compliance y no se usará. E7 corre solo
             por la ruta TypeSafe Jev.

Coste controlado: ~2,000 eventos × ~600 tok ≈ 1.2 Mtok ≈ $0.05 (Jev).

Uso:
    TYPESAFE_API_KEY=sk-... python scripts/e7_vendor_api.py                 # run
    python scripts/e7_vendor_api.py --dry-run                               # sin red
    python scripts/e7_vendor_api.py --n-normal 2000 --n-attack 2000         # escala
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "src")]

from src.features.build_canonical import UNSW_FEATURE_COLUMNS  # noqa: E402

PARQUET = Path("${HOME}/projects/paper/data/processed/unsw_nb15/unsw_nb15.parquet")
NPZ = Path("${HOME}/projects/paper/results/outputs/unsw_small_oficial.npz")
META = Path("${HOME}/projects/paper/results/outputs/unsw_small_oficial.json")
OUT = REPO / "results/e7_vendors"
SEED = 42

PRICE_PER_MTOK = 0.042  # jev-1.13/1.14: $42/Btok input; output gratis.
FAMILY_DESC = {
    "normal": "benign network traffic with no attack",
    "dos": "denial-of-service: exhausts a service with request volume",
    "recon": "reconnaissance: probing or scanning to map the network",
    "exploitation": "exploit of a vulnerable service to gain access",
    "shellcode": "shellcode injection executing attacker payloads",
    "backdoor": "backdoor channel granting remote control",
    "analysis": "file analysis/patched-binary payload inspection",
    "fuzzing": "fuzzing: malformed inputs to provoke crashes",
    "generic": "generic pattern-based attack (mass-scanning probes)",
    "worms": "self-propagating worm infection attempts",
}
SEV_LEVELS = [
    "benign: no attack (level 0)",
    "low severity: recon, fuzzing or generic probing (level 1)",
    "medium severity: dos or analysis, service impact (level 2)",
    "high severity: exploitation, shellcode, backdoor or worms, host compromise (level 3)",
]


# ── muestra estratificada ─────────────────────────────────────────────
def stratified_attack_quota(fams_vector: np.ndarray, n_attack: int) -> np.ndarray:
    """Cotas ∝ prueba con mínimo 15 por familia presente; devuelve índices."""
    fams, cnts = np.unique(fams_vector, return_counts=True)
    q = np.floor(n_attack * cnts / cnts.sum()).astype(int)
    q = np.maximum(q, 15)
    q = np.minimum(q, cnts)  # nunca pedir más de lo que hay
    rng = np.random.default_rng(SEED)
    out = []
    for f, k in zip(fams, q):
        idx = np.where(fams_vector == f)[0]
        out.append(rng.choice(idx, size=int(k), replace=False))
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


# ── request builder ───────────────────────────────────────────────────
def build_state(row: dict) -> dict:
    s = {k: (int(v) if float(v).is_integer() and abs(v) < 1e15 else round(float(v), 6))
         for k, v in zip(UNSW_FEATURE_COLUMNS, row["x"])}
    s.update(proto=row["proto"], service=row["service"], state=row["state"])
    return s


def build_questions(candidates: list[str], with_unknown: bool = False) -> dict:
    crit = {c: FAMILY_DESC[c] for c in candidates}
    if with_unknown:
        crit["unknown"] = "traffic that cannot be attributed to any listed family"
    return {
        "gate": {"type": "noul",
                 "instructions": "This network flow record is malicious traffic (part of an attack), not benign.",
                 "criteria": {"true": "the record is part of an attack",
                              "false": "the record is benign traffic"}},
        "family": {"type": "choice",
                   "instructions": "Which attack family or benign class does this flow record belong to?",
                   "criteria": crit},
        "severity": {"type": "score",
                     "instructions": "How severe is this flow record on the response-priority rubric?",
                     "criteria": SEV_LEVELS},
    }


# ── HTTP ──────────────────────────────────────────────────────────────
VENDORS = {
    "typesafe": {"base": "https://api.typesafe.ai", "env": "TYPESAFE_API_KEY"},
    "opencode": {"base": "https://opencode.ai/zen", "env": "OPENCODE_API_KEY"},
}


def ask_one(vendor: str, model: str, state: dict, questions: dict,
            api_key: str, tries: int = 8) -> tuple[dict, float, dict]:
    base = VENDORS[vendor]["base"]
    body = json.dumps({"model": model, "state": state, "questions": questions},
                      separators=(",", ":")).encode()
    req = urllib.request.Request(
        f"{base}/v1/systemone", data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json",
                 "User-Agent": "vxone-e7/0.1"})  # WAF bloquea Python-urllib
    delay = 2.0
    for i in range(tries):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dt = time.perf_counter() - t0
                return json.loads(r.read()), dt, {}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = float(e.headers.get("retry-after", delay))
                time.sleep(wait); delay = min(delay * 2, 60); continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(delay); delay = min(delay * 2, 60); continue
    raise RuntimeError(f"retries agotados ({tries})")


# ── métricas ──────────────────────────────────────────────────────────
def ece15(conf: np.ndarray, y: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 16)
    e, n = 0.0, len(y)
    for i in range(15):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum():
            e += abs(conf[m].mean() - y[m].mean()) * m.sum() / n
    return float(e)


def auc_from(y, s):
    # AUC por rankings (empates 0.5)
    from scipy.stats import rankdata
    r = rankdata(s)
    P = (y == 1).sum(); N = (y == 0).sum()
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N)) if P and N else float("nan")


def best_fpr_at_tpr95(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """FPR@95TPR idéntico al repo (src/eval/metrics.py: quantile de positivos)."""
    sys.path.insert(0, str(REPO))
    from src.eval.metrics import fpr_at_tpr
    return 0.95, float(fpr_at_tpr(score, y, 0.95))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        if tp == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", choices=["typesafe", "opencode"], default="opencode")
    ap.add_argument("--model", default=None,
                 help="default: jev-1.13.0 (typesafe) / jev-1.13-free (opencode)")
    ap.add_argument("--n-normal", type=int, default=1000)
    ap.add_argument("--n-attack", type=int, default=1000)
    ap.add_argument("--with-unknown", action="store_true",
                    help="añade opción 'unknown' al choice del vendor (abl)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    cand = None
    import pyarrow.parquet as pq

    tbl = pq.read_table(PARQUET, columns=["x_num", "split", "label_family",
                                          "label_severity", "x_txt_proto",
                                          "x_txt_service", "x_txt_state"])
    meta = json.load(open(META))
    cand = meta["candidates"]; Tn = meta["temperatures"]["noul"]; Ts = meta["temperatures"]["score"]

    fam_all = np.asarray(tbl.column("label_family").to_pylist())
    te_mask = np.asarray(tbl.column("split").to_pylist()) == "test"
    fam_te = fam_all[te_mask]
    y_bin = (fam_te != "normal").astype(np.int8)
    sev_te = np.asarray(tbl.column("label_severity").to_pylist())[te_mask].astype(np.int64)
    x = tbl.column("x_num").to_pylist()
    proto = np.asarray(tbl.column("x_txt_proto").to_pylist())[te_mask]
    service = np.asarray(tbl.column("x_txt_service").to_pylist())[te_mask]
    state_c = np.asarray(tbl.column("x_txt_state").to_pylist())[te_mask]

    npz = np.load(NPZ)
    assert (npz["te_y_bin"] == y_bin).all() and len(y_bin) == len(npz["te_y_bin"]), \
        "alineación npz↔parquet rota"

    # muestra estratificada
    rng = np.random.default_rng(SEED)
    idx_n = np.where(y_bin == 0)[0]
    idx_a = np.where(y_bin == 1)[0]
    q_att = stratified_attack_quota(fam_te[idx_a], args.n_attack)
    idx_a_s = idx_a[q_att]
    idx_n_s = rng.choice(idx_n, size=min(args.n_normal, len(idx_n)), replace=False)
    idx = np.sort(np.r_[idx_n_s, idx_a_s])
    rows = []
    for i in idx:
        rows.append({"x": x[i], "proto": str(proto[i]), "service": str(service[i]),
                     "state": str(state_c[i])})

    # nuestos outputs en las mismas filas
    our_gate = 1 / (1 + np.exp(-npz["te_noul_logits"][idx] / Tn))
    our_choice_pred = npz["te_choice_logits"][idx].argmax(1)
    choice_p = np.exp(npz["te_choice_logits"][idx] / Ts - np.log(
        np.exp(npz["te_choice_logits"][idx] / Ts).sum(1, keepdims=True)))
    # esperado ordinal del score head (COR: P(nivel>k)→nivel completo + E[V])
    sl = npz["te_corn_logits"][idx]
    if sl.ndim == 2 and sl.shape[1] == 3:
        lvl_p = np.zeros((len(idx), 4))
        ok = 1 / (1 + np.exp(-sl / Ts))  # P(nivel > k)
        lvl_p[:, 0] = 1 - ok[:, 0]
        lvl_p[:, 1] = ok[:, 0] - ok[:, 1]
        lvl_p[:, 2] = ok[:, 1] - ok[:, 2]
        lvl_p[:, 3] = ok[:, 2]
        our_sev = (lvl_p * np.arange(4)).sum(1)
    else:
        our_sev = np.full(len(idx), np.nan)

    fam_idx = {f: i for i, f in enumerate(cand)}
    label_idx = np.array([fam_idx[f] for f in fam_te[idx]])

    questions = build_questions(cand, with_unknown=args.with_unknown)

    # ── dry-run: valida TODO el pipeline sin red ──
    if args.dry_run:
        s0 = build_state(rows[0])
        print("state[0]:", json.dumps(s0)[:400])
        print("questions:", json.dumps(questions)[:600])
        print("filas:", len(idx), "normales:", len(idx_n_s), "ataques:", len(idx_a_s))
        our_auc = auc_from(y_bin[idx], our_gate)
        tpr95, fpr95 = best_fpr_at_tpr95(y_bin[idx], our_gate)
        print(f"OUR en la muestra: AUROC={our_auc:.4f} FPR@95={fpr95:.4f} "
              f"ECE15={ece15(our_gate, y_bin[idx]):.4f} macroF1_choice="
              f"{macro_f1(label_idx, our_choice_pred, len(cand)):.4f}")
        return

    if args.model is None:
        args.model = "jev-1.13.0" if args.vendor == "typesafe" else "jev-1.13-free"
    env = VENDORS[args.vendor]["env"]
    api_key = os.environ.get(env)
    assert api_key, f"{env} no definido"

    def work(k: int):
        s = build_state(rows[k])
        resp, lat, _ = ask_one(args.vendor, args.model, s, questions, api_key)
        return k, resp["answers"], lat, resp["usage"]["input_tokens"]

    n_gate = np.full(len(idx), np.nan)
    fam_p = np.zeros((len(idx), len(cand) + (1 if args.with_unknown else 0)))
    sev_p = np.zeros((len(idx), 4))
    lats, tokens = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for j, (k, ans, lat, tok) in enumerate(ex.map(work, range(len(idx)))):
            n_gate[k] = ans["gate"]["noul"]
            for name, p in ans["family"]["probabilities"].items():
                if name in fam_idx:
                    fam_p[k, fam_idx[name]] = p
            for lv, p in ans["severity"]["probabilities"].items():
                sev_p[k, int(lv)] = p
            lats.append(lat); tokens.append(tok)
            if (j + 1) % 100 == 0:
                print(f"  {j+1}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)

    vendor_choice = fam_p.argmax(1)
    if args.with_unknown:  # pred unknown → mapea a la clase menos log-boost? no: -1
        vendor_choice = np.where(fam_p[:, len(cand)] > fam_p[:, :len(cand)].max(1),
                                 -1, vendor_choice)

    y = y_bin[idx]
    v_tpr95, v_fpr = best_fpr_at_tpr95(y, n_gate)
    o_tpr95, o_fpr = best_fpr_at_tpr95(y, our_gate)
    res = {
        "vendor": args.vendor, "model": args.model,
        "sample": {"n_normal": int(len(idx_n_s)), "n_attack": int(len(idx_a_s)),
                   "seed": SEED, "stratified": True},
        "gate": {"vendor": {"auroc": auc_from(y, n_gate),
                            "fpr_at_95": v_fpr, "tpr95": v_tpr95,
                            "ece15": ece15(n_gate, y)},
                 "ours": {"auroc": auc_from(y, our_gate),
                          "fpr_at_95": o_fpr, "tpr95": o_tpr95,
                          "ece15": ece15(our_gate, y)}},
        "choice": {"vendor_macro_f1": macro_f1(label_idx, vendor_choice, len(cand)),
                   "our_macro_f1": macro_f1(label_idx, our_choice_pred, len(cand)),
                   "vendor_acc": float((vendor_choice == label_idx).mean()),
                   "our_acc": float((our_choice_pred == label_idx).mean())},
        "severity": {"vendor_mae": float(np.abs(sev_p @ np.arange(4) - sev_te[idx]).mean()),
                     "our_mae": float(np.abs(our_sev - sev_te[idx]).mean())},
        "latency_ms": {"p50": float(np.percentile(lats, 50)) * 1000,
                       "p95": float(np.percentile(lats, 95)) * 1000},
        "cost": {"input_tokens_total": int(sum(tokens)),
                 "usd": sum(tokens) * PRICE_PER_MTOK / 1e6},
        "notes": {"candidates": cand, "with_unknown": args.with_unknown,
                  "thresholds": {"noul": Tn, "score": Ts},
                  "alignment": "te npz filas == parquet test en orden natural (verificado)"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "unknown" if args.with_unknown else "main"
    npz_out = OUT / f"vendor_{args.vendor}_{tag}.npz"
    np.savez_compressed(npz_out, idx=idx, vendor_gate=n_gate,
                        vendor_family_probs=fam_p, vendor_sev_probs=sev_p,
                        # nuestras referencias en las mismas filas:
                        our_gate=our_gate, our_choice_pred=our_choice_pred,
                        our_sev=our_sev, y_bin=y, label_idx=label_idx)
    json.dump(res, open(OUT / f"vendor_{args.vendor}_{tag}.json", "w"), indent=2)
    print(json.dumps(res, indent=2)[:2000])


if __name__ == "__main__":
    main()
