#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}

mkdir -p logs

for fold in dev1 final; do
  echo "=== diagnostic M1nt (no city trend) on $fold ==="
  $PY src/run.py fit --model M1nt --fold "$fold" --run-id "$RUN_ID" \
    --draws 3000 --tune 3000 --chains 4 --cores 4 2>&1 | tee "logs/diagnostic_M1nt_${fold}.log"
  $PY src/run.py predict --model M1nt --fold "$fold" --run-id "$RUN_ID" 2>&1 | tee -a "logs/diagnostic_M1nt_${fold}.log"
  $PY src/run.py effects --model M1nt --fold "$fold" --run-id "$RUN_ID" 2>&1 | tee -a "logs/diagnostic_M1nt_${fold}.log"
done

echo "=== M1nt diagnostic complete ==="
