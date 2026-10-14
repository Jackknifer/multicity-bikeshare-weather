# 期中项目

期中材料包括研究背景、数据清理、数据字典、探索性分析和期末模型方案。

| 材料 | 位置 |
|---|---|
| 报告 | [midterm_report.pdf](report/midterm_report.pdf) |
| 演示 | [midterm_slides.pdf](slides/midterm_slides.pdf) |
| 项目说明与数据字典 | [project-reference.md](docs/project-reference.md) |
| 数据处理 | `code/prepare_bikeshare_weather.py` |
| 探索性分析 | `code/eda_midterm.py` |
| 日度数据 | `data/clean/bikeshare_weather_daily.csv` |
| 结果与图表 | `results/`、`figures/` |

所有期中命令均在本目录执行。Python 依赖记录在 `requirements.txt`。
数据处理和图表复现命令见项目说明。只编译已有文档时可以分别执行：

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build midterm_report.tex
```

```bash
cd slides
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build midterm_slides.tex
```
