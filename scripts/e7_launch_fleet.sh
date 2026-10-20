#!/usr/bin/env bash
# Lanzador E7 fleet: distribuye slices entre 13 nodos. No usar pkill -f aquí.
set -u
K1="$(cat /home/jsantillana/projects/servers/opencode.txt)"
K2="$(cat /home/jsantillana/projects/servers/opencode-other.txt)"
AWS_K=/home/jsantillana/projects/servers/newrsa.pem
AZ_K=~/.ssh/azurekey
SSHO="-o StrictHostKeyChecking=accept-new -o IdentitiesOnly=yes -o ConnectTimeout=20"
LOG=/tmp/opencode/e7_launch.log
: > $LOG

# name|user|keyfile|ip|keyvalue
NODES=(
 "a0|ec2-user|$AWS_K|3.20.227.35|$K1"
 "a1|ec2-user|$AWS_K|3.20.226.13|$K2"
 "a2|ec2-user|$AWS_K|18.224.169.246|$K1"
 "a3|ec2-user|$AWS_K|77.113.58.21|$K2"
 "a4|ec2-user|$AWS_K|18.191.219.245|$K1"
 "a5|ubuntu|$AWS_K|18.191.245.48|$K2"
 "a6|ubuntu|$AWS_K|18.222.132.248|$K1"
 "a7|ubuntu|$AWS_K|13.59.138.213|$K2"
 "z0|ubuntu|$AZ_K|20.124.130.213|$K1"
 "z1|ubuntu|$AZ_K|40.87.109.47|$K2"
 "z2|ubuntu|$AZ_K|40.76.237.22|$K1"
 "z3|ubuntu|$AZ_K|20.163.130.17|$K2"
 "z4|ubuntu|$AZ_K|172.176.124.82|$K1"
)

i=0
for n in "${NODES[@]}"; do
  IFS='|' read -r name user keyfile ip key <<< "$n"
  slice=/tmp/opencode/slices/slice_$(printf '%02d' $i).jsonl
  echo "== $name $ip slice $i: $(wc -l < $slice) líneas" | tee -a $LOG
  scp -i "$keyfile" $SSHO -q /tmp/opencode/e7_remote_worker.py "$slice" "$user@$ip:/tmp/" 2>>$LOG \
    || { echo "  scp FAIL $name" | tee -a $LOG; i=$((i+1)); continue; }
  ssh -i "$keyfile" $SSHO "$user@$ip" \
    "nohup env OPENCODE_API_KEY='$key' PACE=1.0 python3 /tmp/e7_remote_worker.py /tmp/slice_$(printf '%02d' $i).jsonl /tmp/e7_results_$name.jsonl > /tmp/e7_w_$name.log 2>&1 < /dev/null & echo pid \$!" 2>>$LOG \
    | tee -a $LOG
  i=$((i+1))
done
echo "LANZADOS (revisar $LOG)"
