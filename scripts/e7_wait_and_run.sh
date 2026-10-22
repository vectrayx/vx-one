#!/usr/bin/env bash
# E7 wrapper: espera el cooldown del free tier (respetando retry-after) y
# lanza el runner resumible. Al terminar escribe log limpio.
set -u
KEY="$(cat ${HOME}/projects/servers/opencode.txt)"
export OPENCODE_API_KEY="$KEY"
PY=/tmp/opencode/venv/bin/python
cd ${HOME}/projects/paper-fix-glm

sleep_probe () {
  local wait
  for i in $(seq 1 200); do
    wait=$(timeout 60 curl -s -o /dev/null -D - -X POST \
      -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      "https://opencode.ai/zen/v1/systemone" \
      -d '{"model":"jev-1.13-free","state":"ping","questions":{"q":{"type":"noul","instructions":"ok?"}}}' \
      | grep -i "^retry-after:" | tr -d '\r' | awk '{print $2}')
    if [ -n "$wait" ]; then
      echo "$(date +%H:%M:%S) cooldown restante: ${wait}s"
      sleep "$wait"
    else
      echo "$(date +%H:%M:%S) endpoint desbloqueado"
      return 0
    fi
  done
  echo "no se desbloqueo en 200 intentos"; return 1
}

sleep_probe || exit 1
$PY -u scripts/e7_resumable.py --pace 1.2
