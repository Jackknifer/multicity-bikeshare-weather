#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=formal_20260930

mkdir -p logs

for fold in dev1 dev2; do
  echo "=== baselines $fold ==="
  $PY src/run.py baselines --fold "$fold" --run-id "$RUN_ID" 2>&1 | tee "logs/baselines_${fold}.log"
done

for model in M0 M1; do
  for fold in dev1 dev2; do
    echo "=== fit $model $fold ==="
    $PY src/run.py fit --model "$model" --fold "$fold" --run-id "$RUN_ID" \
      --draws 2000 --tune 2000 --chains 4 --cores 4 2>&1 | tee "logs/fit_${model}_${fold}.log"
  done
done

for fold in dev1 dev2; do
  echo "=== fit M2 $fold ==="
  $PY src/run.py fit --model M2 --fold "$fold" --run-id "$RUN_ID" \
    --draws 2000 --tune 2000 --chains 4 --cores 4 2>&1 | tee "logs/fit_M2_${fold}.log"
done

echo "=== dev folds complete ==="
