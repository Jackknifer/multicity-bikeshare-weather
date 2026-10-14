#!/usr/bin/env zsh
# Rebuild every midterm deliverable from the frozen daily panel.
#
#   1. exploratory analysis and figures
#   2. report (pdflatex + bibtex, three passes)
#   3. slides (xelatex, two passes for the cover coordinates)
set -euo pipefail

project_dir=${0:A:h:h}
cd "$project_dir"

echo "== 1. exploratory analysis =="
.venv/bin/python code/eda_midterm.py > /dev/null

echo "== 2. report =="
cd report
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_report.tex > /dev/null
bibtex build/midterm_report > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_report.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_report.tex > /dev/null
cp build/midterm_report.pdf midterm_report.pdf
cd ..

echo "== 3. slides =="
cd slides
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_slides.tex > /dev/null
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_slides.tex > /dev/null
cp build/midterm_slides.pdf midterm_slides.pdf
cd ..

echo "midterm build finished"
