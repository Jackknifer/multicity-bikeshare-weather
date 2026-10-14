#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}
DRAWS=3000
TUNE=3000

mkdir -p logs

run_model () {
  local model=$1
  local fold=$2
  echo "=== fit $model $fold ==="
  $PY src/run.py fit --model "$model" --fold "$fold" --run-id "$RUN_ID" \
    --draws "$DRAWS" --tune "$TUNE" --chains 4 --cores 4 2>&1 | tee "logs/fit_${model}_${fold}.log"
  echo "=== predict $model $fold ==="
  $PY src/run.py predict --model "$model" --fold "$fold" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${model}_${fold}.log"
  echo "=== effects $model $fold ==="
  $PY src/run.py effects --model "$model" --fold "$fold" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${model}_${fold}.log"
  echo "=== residuals $model $fold ==="
  $PY src/run.py residuals --model "$model" --fold "$fold" --run-id "$RUN_ID" 2>&1 | tee -a "logs/fit_${model}_${fold}.log"
}

for fold in dev1 dev2; do
  echo "=== baselines $fold ==="
  $PY src/run.py baselines --fold "$fold" --run-id "$RUN_ID" 2>&1 | tee "logs/baselines_${fold}.log"
done

for model in M0 M1 M2; do
  for fold in dev1 dev2; do
    run_model "$model" "$fold"
  done
done

$PY src/run.py prior-predictive --model M2 --fold dev1 --run-id "$RUN_ID" --draws 1000 2>&1 | tee logs/prior_predictive.log

echo "=== development folds complete ==="
$PY src/check_diagnostics.py --run-id "$RUN_ID"
