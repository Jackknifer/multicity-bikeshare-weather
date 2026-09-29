#!/bin/zsh
set -euo pipefail

project_dir=${0:A:h:h}
export PYTENSOR_FLAGS='cxx='
exec "$project_dir/.venv/bin/python" "$project_dir/code/fit_pilot_model.py" "$@"
