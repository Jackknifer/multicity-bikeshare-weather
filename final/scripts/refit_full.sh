#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}
SELECTED=${2:-M2}

mkdir -p logs

rm -rf "runs/${RUN_ID}/${SELECTED}/full"
echo "=== refit $SELECTED full-period with the non-redundant weekday basis ==="
$PY src/run.py fit --model "$SELECTED" --fold full --roster primary --run-id "$RUN_ID" \
  --draws 3000 --tune 3000 --chains 4 --cores 4 2>&1 | tee "logs/fit_${SELECTED}_full.log"
$PY src/run.py effects --model "$SELECTED" --fold full --roster primary --run-id "$RUN_ID" \
  2>&1 | tee -a "logs/fit_${SELECTED}_full.log"
$PY src/run.py residuals --model "$SELECTED" --fold full --roster primary --run-id "$RUN_ID" \
  2>&1 | tee -a "logs/fit_${SELECTED}_full.log"
echo "=== full-period refit complete ==="
