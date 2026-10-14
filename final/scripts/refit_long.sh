#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}
MODEL=${2:-M2}
FOLD=${3:-dev2}
DRAWS=${4:-4000}
TUNE=${5:-4000}

mkdir -p logs

rm -rf "runs/${RUN_ID}/${MODEL}/${FOLD}"
echo "=== refit ${MODEL} ${FOLD} with ${TUNE}+${DRAWS} ==="
$PY src/run.py fit --model "$MODEL" --fold "$FOLD" --run-id "$RUN_ID" \
  --draws "$DRAWS" --tune "$TUNE" --chains 4 --cores 4 2>&1 | tee "logs/fit_${MODEL}_${FOLD}.log"
$PY src/run.py predict --model "$MODEL" --fold "$FOLD" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${MODEL}_${FOLD}.log"
$PY src/run.py effects --model "$MODEL" --fold "$FOLD" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${MODEL}_${FOLD}.log"
$PY src/run.py residuals --model "$MODEL" --fold "$FOLD" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${MODEL}_${FOLD}.log"
echo "=== refit complete ==="
