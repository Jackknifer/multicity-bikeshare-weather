#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}
SELECTED=${2:-M2}

mkdir -p logs

echo "=== stream C: $SELECTED final fold ==="
$PY src/run.py fit --model "$SELECTED" --fold final --run-id "$RUN_ID" \
  --draws 3000 --tune 3000 --chains 4 --cores 4 2>&1 | tee "logs/streamC_${SELECTED}_final.log"
$PY src/run.py predict --model "$SELECTED" --fold final --run-id "$RUN_ID" 2>&1 | tee -a "logs/streamC_${SELECTED}_final.log"
$PY src/run.py effects --model "$SELECTED" --fold final --run-id "$RUN_ID" 2>&1 | tee -a "logs/streamC_${SELECTED}_final.log"
$PY src/run.py residuals --model "$SELECTED" --fold final --run-id "$RUN_ID" 2>&1 | tee -a "logs/streamC_${SELECTED}_final.log"

echo "=== stream C complete ==="
