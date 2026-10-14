#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}
SELECTED=${2:-M2}
SCALE=${3:-2.0}

tag=$(echo "$SCALE" | tr -d '.')
run_name="${SELECTED}_s${tag}"

mkdir -p logs
rm -rf "runs/${RUN_ID}/${run_name}"

echo "=== sensitivity ${run_name} scale=${SCALE} ==="
$PY src/run.py fit --model "$SELECTED" --fold dev1 --run-name "$run_name" \
  --weather-scale "$SCALE" --run-id "$RUN_ID" --draws 2000 --tune 2000 --chains 4 --cores 4 \
  2>&1 | tee "logs/fit_${run_name}_dev1.log"
$PY src/run.py predict --model "$SELECTED" --fold dev1 --run-name "$run_name" \
  --weather-scale "$SCALE" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${run_name}_dev1.log"
$PY src/run.py effects --model "$SELECTED" --fold dev1 --run-name "$run_name" \
  --weather-scale "$SCALE" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${run_name}_dev1.log"
echo "=== sensitivity complete ==="
