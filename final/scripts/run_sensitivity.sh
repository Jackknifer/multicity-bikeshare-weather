#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}
SELECTED=${2:-M2}

mkdir -p logs

for scale in 0.5 2.0; do
  tag=$(echo "$scale" | tr -d '.')
  run_name="${SELECTED}_s${tag}"
  echo "=== fit $run_name fold=dev1 scale=$scale ==="
  $PY src/run.py fit --model "$SELECTED" --fold dev1 --run-name "$run_name" \
    --weather-scale "$scale" --run-id "$RUN_ID" --draws 2000 --tune 2000 --chains 4 --cores 4 \
    2>&1 | tee "logs/fit_${run_name}_dev1.log"
  $PY src/run.py predict --model "$SELECTED" --fold dev1 --run-name "$run_name" \
    --weather-scale "$scale" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${run_name}_dev1.log"
  $PY src/run.py effects --model "$SELECTED" --fold dev1 --run-name "$run_name" \
    --weather-scale "$scale" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${run_name}_dev1.log"
done

echo "=== sensitivity complete ==="
