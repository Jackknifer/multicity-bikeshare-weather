#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}
SELECTED=${2:?usage: run_stage_final.sh <run_id> <selected_model>}
DRAWS=3000
TUNE=3000

mkdir -p logs

fit () {
  local model=$1
  local fold=$2
  local roster=$3
  local run_name=$4
  local scale=$5
  echo "=== fit $run_name ($model) fold=$fold roster=$roster scale=$scale ==="
  $PY src/run.py fit --model "$model" --fold "$fold" --roster "$roster" \
    --run-name "$run_name" --weather-scale "$scale" --run-id "$RUN_ID" \
    --draws "$DRAWS" --tune "$TUNE" --chains 4 --cores 4 2>&1 | tee "logs/fit_${run_name}_${fold}.log"
}

evaluate () {
  local model=$1
  local fold=$2
  local roster=$3
  local run_name=$4
  local scale=$5
  $PY src/run.py predict --model "$model" --fold "$fold" --roster "$roster" \
    --run-name "$run_name" --weather-scale "$scale" --run-id "$RUN_ID" \
    2>&1 | tee -a "logs/fit_${run_name}_${fold}.log"
  $PY src/run.py effects --model "$model" --fold "$fold" --roster "$roster" \
    --run-name "$run_name" --weather-scale "$scale" --run-id "$RUN_ID" \
    2>&1 | tee -a "logs/fit_${run_name}_${fold}.log"
}

echo "=== final fold ==="
for model in M0 "$SELECTED"; do
  fit "$model" final prediction "$model" 1.0
  evaluate "$model" final prediction "$model" 1.0
  $PY src/run.py residuals --model "$model" --fold final --run-id "$RUN_ID" \
    2>&1 | tee -a "logs/fit_${model}_final.log"
done

echo "=== pooling ablation on dev1 ==="
for model in A0 A1; do
  fit "$model" dev1 prediction "$model" 1.0
  evaluate "$model" dev1 prediction "$model" 1.0
done

echo "=== full-period 25-city fits ==="
for model in M0 "$SELECTED"; do
  fit "$model" full primary "$model" 1.0
  $PY src/run.py effects --model "$model" --fold full --roster primary --run-id "$RUN_ID" \
    2>&1 | tee -a "logs/fit_${model}_full.log"
  $PY src/run.py residuals --model "$model" --fold full --roster primary --run-id "$RUN_ID" \
    2>&1 | tee -a "logs/fit_${model}_full.log"
done

echo "=== weather prior-scale sensitivity for the selected model ==="
for scale in 0.5 2.0; do
  tag=$(echo "$scale" | tr -d '.')
  run_name="${SELECTED}_s${tag}"
  echo "=== fit $run_name ($SELECTED) fold=dev1 scale=$scale draws=2000 ==="
  $PY src/run.py fit --model "$SELECTED" --fold dev1 --roster prediction \
    --run-name "$run_name" --weather-scale "$scale" --run-id "$RUN_ID" \
    --draws 2000 --tune 2000 --chains 4 --cores 4 \
    2>&1 | tee "logs/fit_${run_name}_dev1.log"
  evaluate "$SELECTED" dev1 prediction "$run_name" "$scale"
done

echo "=== aggregate ==="
$PY src/aggregate.py --run-id "$RUN_ID" 2>&1 | tee logs/aggregate.log
$PY src/check_diagnostics.py --run-id "$RUN_ID"
echo "=== stage final complete ==="
