# E7 fleet runbook (ejecutado 2026-10-20)

La corrida completa (2,021 filas) se ejecutó con coste $0 vía el passthrough
System One del Zen de OpenCode (`jev-1.13-free`). El free tier limita por IP
(~250 req/ventana diaria) → distribución:

1. Genera slices del sample seed-42 (con las questions congeladas en la línea
   meta k=-1) con `e7_resumable.py --metrics`-compatible: `build_sample` + 
   `build_questions` (ver e7_launch_fleet.sh caller en el historial) →
   `rest[i::13]`.
2. `bash e7_launch_fleet.sh` scp + nohup en 13 nodos del fleet (keys
   OPENCODE key1/key2 alternadas; SSH con `IdentitiesOnly=yes`).
3. Pull de `e7_results_<nodo>.jsonl` + merge dedupe por k en
   `results/e7_vendors/opencode/jev_free_partial.jsonl`.
4. `python scripts/e7_resumable.py --metrics` → `vendor_jev_free_full.json`.

Nota: `pkill -f e7_...` en ssh remoto se suicide por patrón (está en su propia
línea de comando); usar `pgrep | xargs kill` en llamada separada.
