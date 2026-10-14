#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUN_ID=${1:-formal_20260930}

mkdir -p logs report/build slides/build

echo "=== aggregate ==="
$PY src/aggregate.py --run-id "$RUN_ID" 2>&1 | tee logs/aggregate.log

echo "=== figures ==="
$PY src/make_figures.py --run-id "$RUN_ID" 2>&1 | tee logs/figures.log

echo "=== report inputs ==="
$PY src/make_report_inputs.py --run-id "$RUN_ID" 2>&1 | tee logs/report_inputs.log

echo "=== report ==="
( cd report && mkdir -p build && \
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build final_report.tex > build/pass1.log && \
  bibtex build/final_report > build/bibtex.log && \
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build final_report.tex > build/pass2.log && \
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build final_report.tex > build/pass3.log && \
  cp build/final_report.pdf final_report.pdf )

echo "=== slides ==="
( cd slides && mkdir -p build && \
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build final_slides.tex > build/pass1.log && \
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build final_slides.tex > build/pass2.log && \
  cp build/final_slides.pdf final_slides.pdf )

echo "=== verification ==="
$PY src/verify_deliverables.py --run-id "$RUN_ID" 2>&1 | tee logs/verify.log

echo "=== build complete ==="
