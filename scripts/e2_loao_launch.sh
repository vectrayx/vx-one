#!/usr/bin/env bash
# Launch LOAO jobs from any checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -m scripts.e2_loao "$@"
