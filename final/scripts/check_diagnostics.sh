#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python src/check_diagnostics.py --run-id "${1:-formal_20260930}"
