#!/usr/bin/env python
"""E7-azure — vendor Microsoft-Decision-1 vía Azure OpenAI (ruta /decisions).

Requires caller-supplied ``AZURE_DECISION_ENDPOINT`` and
``AZURE_DECISION_API_KEY`` environment variables:
  POST {endpoint}/decisions
  headers: api-key: <caller-supplied value>
  body: {
    "model": "decision-1",            # nombre del DEPLOYMENT del resource
    "input": "<json del state como string>",
    "questions": [                     # ARRAY, sin id/criteria/options
      {"type": "predicate",            # (equivalente al 'noul' de Jev)
       "instructions": "..."},
      {"type": "choice",
       "instructions": "...",
       "choices": [{"value": "...", "description": "..."}, ...]},
      {"type": "score",
       "instructions": "...",
       "levels": [{"label": "...", "description": "..."}, ...]}
    ]}

ESTADO: el deployment `decision-1` existe pero la invocación devuelve
  404 model_not_found ("The model `Microsoft-Decision-1` does not exist or
  you do not have access to it") → falta entitlement/acceso del modelo en el
  resource (portal: AI Foundry → deployments → provisioning/access) o la
  región no lo sirve. Este runner queda listo; correr cuando el acceso
  funcione (test: --probe).

Protocolo E7 idéntico (EXPERIMENTS.md §9): misma muestra estratificada
(seed 42), MISMO state de features UNSW, tres preguntas (gate/family/
severity), comparación contra los logits del npz oficial.

Uso:
    python scripts/e7_azure_decision.py --probe            # 1 request de prueba
    python scripts/e7_azure_decision.py --run              # corrida completa
    python scripts/e7_azure_decision.py --run --n-normal 300 --n-attack 300
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

PARQUET = REPO / "data/processed/unsw_nb15/unsw_nb15.parquet"
NPZ = REPO / "results/outputs/unsw_small_oficial.npz"
META = REPO / "results/outputs/unsw_small_oficial.json"
OUT = REPO / "results/e7_vendors/azure"
SEED = 42
DEPLOYMENT = "decision-1"

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
    ("0", "benign: no attack (level 0)"),
    ("1", "low severity: recon, fuzzing or generic probing (level 1)"),
    ("2", "medium severity: dos or analysis, service impact (level 2)"),
    ("3", "high severity: exploitation, shellcode, backdoor or worms, host compromise (level 3)"),
]


def creds() -> tuple[str, str]:
    """Read caller-supplied Azure settings; never read a credential file."""
    return (os.environ["AZURE_DECISION_ENDPOINT"],
            os.environ["AZURE_DECISION_API_KEY"])


def build_state(row: dict) -> str:
    s = {k: (int(v) if float(v).is_integer() and abs(v) < 1e15 else round(float(v), 6))
         for k, v in zip(UNSW_FEATURE_COLUMNS, row["x"])}
    s.update(proto=row["proto"], service=row["service"], state=row["state"])
    return json.dumps(s, separators=(",", ":"))


def build_questions(candidates: list[str]) -> list[dict]:
    return [
        {"type": "predicate",
         "instructions": "This network flow record is malicious traffic (part of an attack), not benign. "
                         "true = the record is part of an attack; false = the record is benign traffic."},
        {"type": "choice",
         "instructions": "Which attack family or benign class does this flow record belong to?",
         "choices": [{"value": c, "description": FAMILY_DESC[c]} for c in candidates]},
        {"type": "score",
         "instructions": "How severe is this flow record on the response-priority rubric? "
                         "Answer with the level number.",
         "levels": [{"label": lab, "description": desc} for lab, desc in SEV_LEVELS]},
    ]


def ask(state_json: str, questions: list[dict], ep: str, key: str,
        tries: int = 6) -> tuple[dict, float]:
    body = json.dumps({"model": DEPLOYMENT, "input": state_json,
                       "questions": questions}, separators=(",", ":")).encode()
    req = urllib.request.Request(f"{ep}/decisions", data=body, method="POST",
                                 headers={"api-key": key,
                                          "Content-Type": "application/json"})
    delay = 2.0
    for _ in range(tries):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read()), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(float(e.headers.get("retry-after", delay)))
                delay = min(delay * 2, 60)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(delay); delay = min(delay * 2, 60)
    raise RuntimeError("retries agotados")


def probe() -> None:
    ep, key = creds()
    qs = build_questions(["normal", "dos", "recon"])
    try:
        out, dt = ask(build_state({"x": np.zeros(len(UNSW_FEATURE_COLUMNS)),
                                   "proto": "tcp", "service": "http", "state": "FIN"}),
                      qs, ep, key)
        print(json.dumps(out, indent=1)[:2000])
        print(f"[probe] latencia {dt:.2f}s")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        print(f"[probe] HTTP {e.code}: {body}")
        if "model_not_found" in body:
            print("[probe] → el deployment 'decision-1' existe pero el resource no "
                  "tiene acceso al modelo Microsoft-Decision-1 aún. En el portal: "
                  "AI Foundry deployment 'decision-1': review "
                  "provisioning state; y Models → Microsoft-Decision-1 → Request "
                  "access (preview) / región con capacidad. Reintentar --probe.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-normal", type=int, default=1010)
    ap.add_argument("--n-attack", type=int, default=1011)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.probe:
        probe()
        return
    raise SystemExit("corrida completa: habilitar tras --probe OK "
                     "(mismo protocolo que scripts/e7_vendor_api.py; "
                     "reutilizar sus funciones de métricas)")


if __name__ == "__main__":
    main()
