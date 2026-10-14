#!/bin/sh
set -eu
project_dir="$(dirname "$0")/.."
cd "$project_dir"

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build -cd report/final_report.tex
cp report/build/final_report.pdf report/final_report.pdf
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build -cd slides/final_slides.tex
cp slides/build/final_slides.pdf slides/final_slides.pdf
echo "报告与演示编译完成。"
