#!/usr/bin/env bash
# lanzador LOAO baselines (protocolo idéntico a run_tabular de E1)
set -u
cd /home/jsantillana/projects/paper-fix-glm
PY=/home/jsantillana/projects/paper/.venv/bin/python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
SRC=${1:-unsw}
FAMS_unsw="analysis backdoor dos exploitation fuzzing generic recon shellcode worms"
FAMS_cic="botnet bruteforce dos_ddos exploitation infiltration recon webattack"
FAMS=$(eval "echo \$FAMS_$SRC")
SCHEME=oficial; [ "$SRC" != unsw ] && SCHEME=strat
for fam in $FAMS; do
  echo "=== fold $fam $(date +%T) ==="
  $PY -m src.baselines.run_tabular --source $SRC --scheme "$SCHEME" \
      --exclude-family "$fam" --out results/baselines_loao \
      >> /tmp/opencode/loao_logs/${SRC}_${fam}.log 2>&1
  echo "=== fold $fam exit=$? $(date +%T) ==="
done
echo "LOAO $SRC DONE"
