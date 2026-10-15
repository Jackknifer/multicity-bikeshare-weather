# 期末项目文件结构

| 目录或文件 | 内容 |
|---|---|
| [configs/](configs/) | 路径、模型与评估窗口配置 |
| [data/](data/) | 日度数据、审计与日期切分记录 |
| [src/](src/) | 建模、预测、评估与核验代码 |
| [tests/](tests/) | 数据隔离与指标测试 |
| [runs/formal_20260930/](runs/formal_20260930/) | 按模型与评估窗口保存的正式实验结果 |
| [tables/](tables/) | 汇总表与模型选择记录 |
| [figures/](figures/) | PDF 与 PNG 图表 |
| [report/](report/) | 报告源文件、表格、参考文献与 PDF |
| [slides/](slides/) | 演示源文件、主题与 PDF |
| [docs/](docs/) | 课程要求、分析协议、讲稿与核验记录 |
| [logs/review-20261001/](logs/review-20261001/) | 补充核验脚本与结果 |
| [scripts/build_documents.sh](scripts/build_documents.sh) | 文档编译脚本 |
| [requirements.txt](requirements.txt) | Python 依赖 |

## 软件依赖

Python 3.12，主要使用 PyMC、nutpie、ArviZ、NumPy、pandas 和 SciPy。
完整版本见 `requirements.txt`。文档编译需要 TeX Live、latexmk、pdfLaTeX、
BibTeX、XeLaTeX，以及演示主题使用的字体。

## 主要结果复现

在 `final/` 目录准备环境与路径：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export BIKESHARE_PATHS=configs/paths.github.yaml
```

已有天气关联、逐日预测与诊断保存在 `runs/formal_20260930/`。
重新拟合主要模型并计算预测评分时，使用新的运行标识：

```bash
RUN_ID=reproduce_20261015
for fold in dev1 dev2; do
  .venv/bin/python src/run.py baselines --fold "$fold" --run-id "$RUN_ID"
done
for model in M0 M1 M2; do
  for fold in dev1 dev2 final; do
    .venv/bin/python src/run.py fit --model "$model" --fold "$fold" \
      --run-id "$RUN_ID" --draws 3000 --tune 3000 --chains 4 --cores 4
    .venv/bin/python src/run.py predict --model "$model" --fold "$fold" --run-id "$RUN_ID"
    .venv/bin/python src/run.py residuals --model "$model" --fold "$fold" --run-id "$RUN_ID"
  done
done
.venv/bin/python src/aggregate.py --run-id "$RUN_ID" --out "tables/$RUN_ID"
```

全期天气关联使用 25 个城市，单独执行：

```bash
for model in M0 M2; do
  .venv/bin/python src/run.py fit --model "$model" --fold full --roster primary \
    --run-id "$RUN_ID" --draws 3000 --tune 3000 --chains 4 --cores 4
  .venv/bin/python src/run.py effects --model "$model" --fold full --roster primary --run-id "$RUN_ID"
done
```

新结果写入 `runs/$RUN_ID/` 与 `tables/$RUN_ID/`。每次复现使用不同的 `RUN_ID`。
只编译现有报告与演示时，执行 `./scripts/build_documents.sh`。
编译直接使用 `report/generated/` 中已整理的表格与数字。
