#!/usr/bin/env bash
# F1 bootstrap — desbloquea el pipeline de datos completo en un solo paso:
#   [1] fija signify==0.7.1 (thrember usa la API vieja de signify.authenticode;
#       pip instaló 0.9.3 donde la restructuraron — bug de empaquetado de thrember)
#   [2] materializa UNSW-NB15 → parquet canónico (foreground, ~1-2 min)
#   [3] arranca EN BACKGROUND:
#         a) descarga EMBER2024 subset Win32 + challenge (~29 GB, HF joyce8/EMBER2024)
#         b) materialización CIC-IDS2017 (2.83M flujos, ~5-15 min)
#       Logs: logs/ember2024_download.log y logs/cic_materialize.log
#
# Falla ruidosamente en el primer error (set -e): si [1] o [2] revienta,
# los backgrounds NO arrancan y se ve el traceback aquí.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== [1/4] fijando signify==0.7.1 (API que usa thrember) =="
.venv/bin/pip install -q 'signify==0.7.1'
.venv/bin/python -I -c "import thrember; print('THREMBER_OK')"

echo "== [2/4] materializando UNSW-NB15 (parquet canónico) =="
.venv/bin/python -I src/features/materialize.py --dataset unsw

echo "== [3/4] descarga EMBER2024 Win32 en background =="
mkdir -p data/raw/ember2024 logs
nohup .venv/bin/python -I -c "
import thrember
thrember.download_dataset('data/raw/ember2024', split='all', file_type='Win32')
print('EMBER2024_DESCARGA_COMPLETA')
" > logs/ember2024_download.log 2>&1 &
echo "   PID $! — log: logs/ember2024_download.log"

echo "== [4/4] materialización CIC-IDS2017 en background =="
nohup .venv/bin/python -I src/features/materialize.py --dataset cic > logs/cic_materialize.log 2>&1 &
echo "   PID $! — log: logs/cic_materialize.log"

echo "== LISTO — los dos jobs largos corren detached; logs vigilables =="
