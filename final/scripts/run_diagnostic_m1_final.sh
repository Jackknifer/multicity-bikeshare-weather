#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}

mkdir -p logs

echo "=== diagnostic: M1 on the final fold ==="
$PY src/run.py fit --model M1 --fold final --run-id "$RUN_ID" \
  --draws 3000 --tune 3000 --chains 4 --cores 4 2>&1 | tee logs/diagnostic_M1_final.log
$PY src/run.py predict --model M1 --fold final --run-id "$RUN_ID" 2>&1 | tee -a logs/diagnostic_M1_final.log
$PY src/run.py effects --model M1 --fold final --run-id "$RUN_ID" 2>&1 | tee -a logs/diagnostic_M1_final.log
echo "=== diagnostic complete ==="
