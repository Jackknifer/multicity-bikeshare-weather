# 期中项目文件结构

| 目录或文件 | 内容 |
|---|---|
| [code/](code/) | 数据处理、探索性分析与文档编译脚本 |
| [data/raw/](data/raw/) | 原始数据 |
| [data/clean/](data/clean/) | 清理后的日度数据 |
| [results/](results/) | 数据审计与分析结果 |
| [figures/](figures/) | 图表 |
| [report/](report/) | 报告源文件、参考文献与 PDF |
| [slides/](slides/) | 演示源文件、主题与 PDF |
| [requirements.txt](requirements.txt) | Python 依赖 |

## 软件依赖

Python 3.12，依赖版本见 `requirements.txt`。文档编译需要 TeX Live、
pdfLaTeX、BibTeX、XeLaTeX，以及演示主题使用的字体。

## 主要结果复现

在 `midterm/` 目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/prepare_bikeshare_weather.py
.venv/bin/python code/eda_midterm.py
```

清理数据写入 `data/clean/`，审计与分析结果写入 `results/`，图表写入 `figures/`。
编译报告与演示可执行 `zsh code/build_midterm.sh`，该脚本同时重新生成探索性分析图表。
