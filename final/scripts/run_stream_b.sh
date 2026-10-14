#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}

mkdir -p logs

echo "=== stream B: M0 final fold ==="
$PY src/run.py fit --model M0 --fold final --run-id "$RUN_ID" \
  --draws 3000 --tune 3000 --chains 4 --cores 4 2>&1 | tee logs/streamB_M0_final.log
$PY src/run.py predict --model M0 --fold final --run-id "$RUN_ID" 2>&1 | tee -a logs/streamB_M0_final.log
$PY src/run.py effects --model M0 --fold final --run-id "$RUN_ID" 2>&1 | tee -a logs/streamB_M0_final.log
$PY src/run.py residuals --model M0 --fold final --run-id "$RUN_ID" 2>&1 | tee -a logs/streamB_M0_final.log

echo "=== stream B: M0 full-period 25-city fit ==="
$PY src/run.py fit --model M0 --fold full --roster primary --run-id "$RUN_ID" \
  --draws 3000 --tune 3000 --chains 4 --cores 4 2>&1 | tee logs/streamB_M0_full.log
$PY src/run.py effects --model M0 --fold full --roster primary --run-id "$RUN_ID" 2>&1 | tee -a logs/streamB_M0_full.log
$PY src/run.py residuals --model M0 --fold full --roster primary --run-id "$RUN_ID" 2>&1 | tee -a logs/streamB_M0_full.log

echo "=== stream B: pooling ablation on dev1 ==="
for model in A0 A1; do
  $PY src/run.py fit --model "$model" --fold dev1 --run-id "$RUN_ID" \
    --draws 3000 --tune 3000 --chains 4 --cores 4 2>&1 | tee "logs/streamB_${model}_dev1.log"
  $PY src/run.py predict --model "$model" --fold dev1 --run-id "$RUN_ID" 2>&1 | tee -a "logs/streamB_${model}_dev1.log"
  $PY src/run.py effects --model "$model" --fold dev1 --run-id "$RUN_ID" 2>&1 | tee -a "logs/streamB_${model}_dev1.log"
done

echo "=== stream B complete ==="
