# STA5001 期末项目工作目录

本目录包含 STA5001 期末报告、演示、代码、配置、正式结果与交付说明。
期中材料位于 `../midterm/`，课程考纲位于 `../STA5001-syllabus.pdf`。

## GitHub 材料与使用方式

在本目录执行复现命令前，设置仓库路径配置：

```bash
export BIKESHARE_PATHS=configs/paths.github.yaml
```

数据 CSV、259 个正式运行产物、汇总表、图表及六个独立核验证据文件均保留原有相对位置。
后验样本保存在各运行目录的 `posterior.nc`，已有结果可以直接阅读，报告和演示可以直接编译。
本仓库版本通过文档编译和文件核对，尚未在新环境重新执行统计实验。

XUE Yantong 负责数据搜寻、课题确定、实验设计、代码编写、实验运行、主要统计分析和
材料编写。WANG Ziyu 与 ZHAO Jiashen 共同参与数据处理、代码编写与检查、报告和演示核验。

## 一、数据来源与许可

- 日度面板由 Mendeley Data 数据集 `10.17632/2nxpvtz935.1`（Version 1，CC BY 4.0）经期中
  处理流程生成，对应论文 Bean, Pojani 与 Corcoran (2021)，*Journal of Transport Geography*
  95:103155，DOI `10.1016/j.jtrangeo.2021.103155`。
- 冻结文件 `bikeshare_weather_daily.csv` 的 SHA-256 为
  `6e30157ee8061f76142b95dd9ca7615bdea068f57c2fbd82796c7ac5e62af33e`，每次拟合前校验。
- 交付副本位于 `data/bikeshare_weather_daily.csv`，与期中目录中的冻结输入逐字节一致
  （SHA-256 相同）。原始数据保存在 `../midterm/data/raw/`，后验保存在本目录的 `runs/`。
- 数据集的教师批准状态只能依据既有记录，本项目不声称已获批准。

## 二、目录结构

```
final/
  README.md                 本文件
  requirements.txt          Python 依赖版本
  configs/
    paths.local.yaml        本机绝对路径（当前机器的运行配置）
    paths.github.yaml       仓库用的相对路径配置
    splits.json             两个开发折与末段评估的日期切分
    models.json             M0/M1/M2 与汇聚消融规格及先验尺度
  data/
    bikeshare_weather_daily.csv  清理后的日度面板（交付副本）
    city_data_audit.csv          城市审计表
    integrity_report.json        冻结输入完整性检查
    frozen_input.json            输入哈希
    roster_comparison.json       三个城市范围的行数对比
    split_index.csv              逐行切分标记
    split_summary.json           切分汇总
  src/                      实现（见下方入口说明）
  tests/test_pipeline.py    防泄漏与指标测试
  runs/formal_20260930/     正式结果，按 <model>/<fold>/ 组织
  tables/                   汇总表与选模记录
  figures/                  报告与演示使用的 10 张图，各含 PDF 与 300 dpi PNG
  report/                   期末报告（NeurIPS 2026 模板）
  slides/                   期末演示（CUHK-SZ 主题）
  docs/                     课程要求、审计、协议、讲稿、验收与交付状态
  scripts/build_documents.sh 文档编译入口
  logs/                     运行日志与独立核验证据（提交范围见第三节）
```

## 三、环境与依赖

独立虚拟环境 `.venv`，Python 3.12。依赖版本记录在 `requirements.txt` 与每次运行的
`config.json`。关键版本：PyMC 6.3.2、nutpie 0.16.11、PyTensor 3.3.2、ArviZ 1.3.0、
NumPy 2.5.3、pandas 3.0.6、Matplotlib 3.11.2、SciPy 1.18.1。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### 提交范围

不提交的内容：

- `.venv/`，依赖按 `requirements.txt` 重建。
- LaTeX 的 `report/build/` 与 `slides/build/`，它们只保存每次编译产生的中间文件。
- `logs/` 下的普通运行日志，例如 `fit_*.log`、`streamB_*.log`、`figures.log`、`aggregate.log`。
  它们记录耗时与完成状态，不是结果来源。
- `logs/review-20261001/` 下的缓存目录 `compiler-cache/`、`matplotlib/`、`numba/`。

必须提交的补充证据位于 `logs/review-20261001/`，保持下列相对路径：

| 文件 | 内容 |
|---|---|
| `logs/review-20261001/followup_checks.json` | 末段基线、AR-OLS 创新累积系数、预测积分稳定性、完整递推先验预测 |
| `logs/review-20261001/recomputed_metrics.csv` | 从已保存逐日预测独立复算的 48 组指标 |
| `logs/review-20261001/supplementary_final_baselines.csv` | 末段窗口基线的逐条预测记录 |
| `logs/review-20261001/independent_checks.json` | 独立核验结果，含冻结输入哈希、后验诊断复算与预测隔离检查 |
| `logs/review-20261001/audit_followup.py` | 生成 `followup_checks.json` 与 `supplementary_final_baselines.csv` 的脚本 |
| `logs/review-20261001/audit_saved_results.py` | 生成 `independent_checks.json` 与 `recomputed_metrics.csv` 的脚本 |

报告使用的末段基线点预测与完整递推先验预测来自这些文件，因此它们属于提交材料，不能随
普通运行日志一起排除。两个脚本依赖的包内文件见第 7.3 节。

## 四、数据路径配置

`src/common.py` 的 `load_paths` 从 `configs/paths.local.yaml` 读取路径，也可以用环境变量
`BIKESHARE_PATHS` 指向其他配置文件。`load_paths` 要求配置中存在 `project_root`、
`midterm_root`、`work_root`、`data_csv`、`data_sha256` 五个字段，缺少任何一个都会终止；
其余字段可以省略。

`verify_frozen_input` 只读取 `data_csv` 指向的文件，并核对其 SHA-256 是否等于
`data_sha256`。`src/run.py audit` 随后用同一个 CSV 生成城市审计表、完整性报告与城市范围
对比，写入 `data/`。当前 `audit` 不打开 `raw_zip` 与 `syllabus_pdf`。

- 在本机继续使用现有输出时，保持 `configs/paths.local.yaml` 不变，它指向期中目录里的
  原始冻结文件。
- 使用仓库材料时，在 `final` 目录设置 `BIKESHARE_PATHS=configs/paths.github.yaml`。

`configs/paths.github.yaml` 中的字段分两类。必须保留的配置字段由 `load_paths` 决定，
程序是否读取该字段指向的文件是另一件事。

| 字段 | `load_paths` 是否要求 | 程序是否读取该字段指向的文件 | 指向 |
|---|---|---|---|
| `data_csv` | 是 | 是，`verify_frozen_input` 与 `audit` 读取 | `data/bikeshare_weather_daily.csv`，包内 |
| `data_sha256` | 是 | 是，作为校验目标 | 包内副本的哈希 |
| `work_root` | 是 | 否 | `.`，包内 |
| `project_root` | 是 | 否 | `..`，仓库根目录 |
| `midterm_root` | 是 | 否 | `../midterm`，仓库内 |
| `raw_zip` | 否 | 否 | 期中目录下的原始压缩包，仓库内 |
| `raw_zip_sha256` | 否 | 否 | 原始压缩包的哈希 |
| `syllabus_pdf` | 否 | 否 | `../STA5001-syllabus.pdf`，仓库内 |

`raw_zip` 与 `syllabus_pdf` 属于仓库内的辅助材料，当前程序不会打开它们。

只使用交付包内的数据复现统计结果时，`data_csv` 指向包内副本、`data_sha256` 与副本一致
即可。完整仓库同时保留原始压缩包与课程考纲。

## 五、代码入口

| 文件 | 用途 |
|---|---|
| `src/run.py` | 命令行入口，子命令见下 |
| `src/common.py` | 路径加载、哈希校验、环境快照 |
| `src/data.py` | 冻结数据读取、训练期标准化、设计数组 |
| `src/splits.py` | 切分构造与泄漏断言 |
| `src/model.py` | PyMC 模型构造与后验访问 |
| `src/predict.py` | 块预测与一步预测 |
| `src/metrics.py` | WIS、对数预测密度、覆盖率、标准化误差 |
| `src/baselines.py` | 逐城常数、7 日季节朴素、逐城 OLS、逐城 AR-OLS |
| `src/effects.py` | 天气效应、温度曲线峰值、支撑范围 |
| `src/residuals.py` | 训练残差、自相关、后验预测核验 |
| `src/audit.py` | 城市审计 |
| `src/aggregate.py` | 结果汇总、选模规则、配对自助区间 |
| `src/figures.py`、`src/make_figures.py` | 图表 |
| `src/check_diagnostics.py` | 收敛验收汇总 |
| `src/verify_deliverables.py` | 交付核验 |

`src/run.py` 的子命令：`audit`、`splits`、`baselines`、`fit`、`predict`、`effects`、
`residuals`、`prior-predictive`。

## 六、主要结果与来源文件

| 结果 | 来源 |
|---|---|
| 降水与温度关联（M0、M2 全期，25 城市） | `runs/formal_20260930/M0/full/effects.json`、`runs/formal_20260930/M2/full/effects.json` |
| 城市层面的关联与区间 | 同上文件的 `cities` 数组；`figures/fig02_rain_forest.pdf` |
| 温度曲线与观察范围 | `figures/fig03_temperature_curves.pdf`、`figures/figA2_all_city_curves.pdf` |
| 残差自相关诊断 | `runs/formal_20260930/<model>/<fold>/residual_summary.json`、`figures/fig04_residual_acf.pdf` |
| 预测评分与覆盖率 | `runs/formal_20260930/<model>/<fold>/metrics_overall.json`、`tables/all_metrics.csv` |
| 末段窗口的模型点预测 MAE | `logs/review-20261001/recomputed_metrics.csv` |
| 末段窗口的基线点预测 MAE | `logs/review-20261001/followup_checks.json` 的 `final_baselines_existing_implementation` |
| 配对自助区间 | `tables/paired_comparison.json` |
| 收敛诊断 | `runs/formal_20260930/<model>/<fold>/diagnostics.json`、`logs/review-20261001/independent_checks.json` |
| 选模记录 | `tables/selection.json`、`tables/selection_dev_folds.csv` |
| 先验尺度敏感性（M2，开发折 1） | `runs/formal_20260930/M2_s05/dev1/effects.json`、`runs/formal_20260930/M2_s20/dev1/effects.json` |
| 完整递推先验预测（M2，开发折 1） | `logs/review-20261001/followup_checks.json` 的 `joint_prior_predictive` |
| 预测积分稳定性 | `logs/review-20261001/followup_checks.json` 的 `predictive_integration_stability` |

表中列出的文件全部属于提交材料。`runs/`、`tables/`、`figures/`、`report/`、`slides/`、
`docs/` 与 `logs/review-20261001/` 下的这六个补充证据文件都要随交付材料提供；`logs/` 下
其他位置的普通运行日志不在此列，提交范围见第三节。

## 七、复现步骤

以下命令按当前源文件的入口编写。本轮交付只做了阅读核对，没有执行其中任何一条。

### 7.1 只使用已有结果

报告与演示中的数字来自三处，逐项来源见第六节。

- 正式运行目录 `runs/formal_20260930/<model>/<fold>/`：天气关联、预测评分与覆盖率、收敛
  诊断、残差自相关。
- 汇总表 `tables/`：跨模型对比、选模记录与配对自助区间。
- 补充核验文件 `logs/review-20261001/`：末段窗口的基线点预测、末段窗口的模型点预测 MAE、
  完整递推先验预测与预测积分稳定性。

只阅读结果、核对数字或编译文档时，不需要运行任何统计步骤。

`report/generated/` 下的五个文件是人工整理输入，已经包含报告需要的全部数字，编译报告与
演示时直接使用。

### 7.2 重新生成统计结果

**先读这一条。** `fit` 在 `posterior.nc` 已经存在时会直接终止，但 `predict`、`effects`、
`residuals`、`baselines`、`prior-predictive` 会覆盖同名输出。要重新生成结果，请另给一个
运行标识，例如 `--run-id reproduce_20261001`，不要复用 `formal_20260930`，否则正式运行
记录会被覆盖。

```bash
export BIKESHARE_PATHS=configs/paths.github.yaml
RUN_ID=reproduce_20261001

# 1. 数据审计与切分
.venv/bin/python src/run.py audit
.venv/bin/python src/run.py splits

# 2. 防泄漏与指标测试
.venv/bin/python tests/test_pipeline.py

# 3. 开发折的无 MCMC 基线，块预测与一步预测一次写出
for fold in dev1 dev2; do
  .venv/bin/python src/run.py baselines --fold "$fold" --run-id "$RUN_ID"
done

# 4. 开发折的正式拟合与正文所需诊断
for model in M0 M1 M2; do
  for fold in dev1 dev2; do
    .venv/bin/python src/run.py fit --model "$model" --fold "$fold" --run-id "$RUN_ID" \
      --draws 3000 --tune 3000 --chains 4 --cores 4
    .venv/bin/python src/run.py predict --model "$model" --fold "$fold" --run-id "$RUN_ID"
    .venv/bin/python src/run.py effects --model "$model" --fold "$fold" --run-id "$RUN_ID"
    .venv/bin/python src/run.py residuals --model "$model" --fold "$fold" --run-id "$RUN_ID"
  done
done

# 5. 末段窗口：M0、M1、M2 三个规格都要拟合与预测
for model in M0 M1 M2; do
  .venv/bin/python src/run.py fit --model "$model" --fold final --run-id "$RUN_ID" \
    --draws 3000 --tune 3000 --chains 4 --cores 4
  .venv/bin/python src/run.py predict --model "$model" --fold final --run-id "$RUN_ID"
  .venv/bin/python src/run.py effects --model "$model" --fold final --run-id "$RUN_ID"
done
# M0 与 M2 另外需要残差诊断（正文用到自相关与训练条件预测检查）
for model in M0 M2; do
  .venv/bin/python src/run.py residuals --model "$model" --fold final --run-id "$RUN_ID"
done

# 6. 25 城市全期拟合与关联结果
#    --roster primary 决定使用 25 城市名单，缺少它就会退回 23 城市预测名单。
#    全期折没有测试分区，predict 会失败，因此只运行 effects 与 residuals。
for model in M0 M2; do
  .venv/bin/python src/run.py fit --model "$model" --fold full --roster primary \
    --run-id "$RUN_ID" --draws 3000 --tune 3000 --chains 4 --cores 4
  .venv/bin/python src/run.py effects --model "$model" --fold full --roster primary \
    --run-id "$RUN_ID"
  .venv/bin/python src/run.py residuals --model "$model" --fold full --roster primary \
    --run-id "$RUN_ID"
done

# 7. 天气先验尺度敏感性
#    保存的模型名称为 M2_s05 与 M2_s20，抽样设置为 2,000 + 2,000。
for scale in 0.5 2.0; do
  tag=$(echo "$scale" | tr -d '.')
  run_name="M2_s${tag}"
  .venv/bin/python src/run.py fit --model M2 --fold dev1 --run-name "$run_name" \
    --weather-scale "$scale" --run-id "$RUN_ID" --draws 2000 --tune 2000 --chains 4 --cores 4
  .venv/bin/python src/run.py predict --model M2 --fold dev1 --run-name "$run_name" \
    --weather-scale "$scale" --run-id "$RUN_ID"
  .venv/bin/python src/run.py effects --model M2 --fold dev1 --run-name "$run_name" \
    --weather-scale "$scale" --run-id "$RUN_ID"
done

# 8. 汇总、选模与收敛验收
#    --run-id 指定要汇总的运行目录，--out 指定汇总结果的保存目录。
#    省略 --out 时 aggregate.py 会写入公共的 tables/，覆盖正式汇总文件，
#    因此复现时把结果写到以运行标识命名的独立目录。
.venv/bin/python src/aggregate.py --run-id "$RUN_ID" --out "tables/$RUN_ID"
.venv/bin/python src/check_diagnostics.py --run-id "$RUN_ID"
```

统计复现按上述命令逐步执行，使用独立的运行标识与输出目录。

正式报告使用 `tables/` 下现有的正式汇总文件，它们由 `--run-id formal_20260930` 生成。
复现得到的汇总文件保存在 `tables/<运行标识>/`，两套文件互不覆盖。

### 7.3 需要单独说明的结果

**末段窗口的基线点预测。** 正式基线目录 `runs/formal_20260930/baselines/` 只包含 `dev1` 与
`dev2`。报告末段基线表中的季节朴素、城市 OLS、城市 AR-OLS、城市常数四项来自已有补充检查
`logs/review-20261001/followup_checks.json` 的 `final_baselines_existing_implementation`，
逐条预测记录保存在 `logs/review-20261001/supplementary_final_baselines.csv`。生成它们的
实现是 `logs/review-20261001/audit_followup.py`，它调用 `src/baselines.py` 的同一套基线
函数与 `src/metrics.py` 的汇总函数，因此与开发折基线口径一致。若要重新生成，运行该脚本；
**它会覆盖 `logs/review-20261001/followup_checks.json`，并在同一次运行中重算 AR-OLS 创新
累积系数、预测积分稳定性和完整先验预测**，运行前请先备份该文件。报告只使用其中末段基线的
MAE 一列；基线的区间评分、密度与覆盖率不用于模型优劣结论，原因是基线的多步区间没有累计
未来创新。

**独立核验脚本与产物。** `logs/review-20261001/audit_saved_results.py` 从已保存的逐日预测
与后验复算指标和诊断，输出 `recomputed_metrics.csv` 与 `independent_checks.json`；报告
引用的末段窗口模型点预测 MAE 来自前者。这两个脚本及其四个已有输出随交付材料提供，保持
原有相对路径，清单见第三节。

两个脚本依赖以下包内文件：

- `src/` 下的模块。脚本用 `Path(__file__).resolve().parents[2]` 定位 `final` 目录后导入，
  因此脚本文件必须留在 `logs/review-20261001/`。
- `configs/splits.json` 与 `configs/models.json`，由 `load_splits` 与 `load_model_config`
  按默认路径读取。
- 一份路径配置。两个脚本调用 `load_paths()` 时不带参数，默认读取
  `configs/paths.local.yaml`；只拿到交付包时需要设置
  `BIKESHARE_PATHS=configs/paths.github.yaml`。
- 该路径配置中 `data_csv` 指向的 CSV。
- `runs/formal_20260930/`。`audit_saved_results.py` 读取其中每个 `<model>/<fold>/` 的
  `predictions.csv`、`metrics_overall.json`、`posterior.nc`、`config.json`、
  `diagnostics.json`、`preprocessing.json` 与 `effects.json`；`audit_followup.py` 读取
  `M2/dev1` 与 `M2/final` 的 `posterior.nc` 与 `metrics_overall.json`。

**完整递推先验预测。** 正文引用的 3.50% 来自同一份 `followup_checks.json` 的
`joint_prior_predictive` 字段，由 `audit_followup.py` 生成：以 M2 规格与 `dev1` 折为对象，
用 `pm.sample_prior_predictive(draws=1000, random_seed=27)` 抽先验，把观测到的天气与日历
代入均值函数，再递推 AR 状态并模拟 Student-$t$ 创新。
`src/run.py prior-predictive` 保存的是参数分位数与截距负值比例
（`intercept_negative_fraction`），不含这一步递推，因此它对应正文中的截距抽样比例，不是
完整递推先验预测的复现入口。

**M0 的完整先验预测未评估。** 报告写明这一点，仓库中也没有对应产物。

### 7.4 编译报告与演示

这一步只编译已有文档。需要 latexmk、pdfLaTeX、BibTeX 与 XeLaTeX。

```bash
./scripts/build_documents.sh
```

`report/generated/` 下的五个文件是人工整理输入，编译时直接使用。
可从本目录核验已有交付文件：

```bash
BIKESHARE_PATHS=configs/paths.github.yaml .venv/bin/python src/verify_deliverables.py --out docs/deliverable-verification.json
```

## 八、运行输出

每个 `runs/<run_id>/<model>/<fold>/` 至少保存：

```
config.json             输入哈希、切分、模型规格、先验、随机种子、软件版本
model_spec.md           数学规格、初始状态、支撑与解释
preprocessing.json      训练期均值、尺度、城市顺序与趋势中心
posterior.nc            后验样本
posterior_summary.csv   区间、ESS、Rhat、MCSE
diagnostics.json        通过与否与具体阈值
predictions.csv         逐日各任务的中位数、区间、对数预测密度、实际值
metrics_city.csv        逐城指标
metrics_overall.json    总体指标与积分稳定性检查
residual_acf.csv        逐城自相关
residual_summary.json   残差汇总
ppc_train.csv           训练期条件预测检查
events.jsonl            耗时、错误、完成状态
```

完整后验文件体积较大（`runs/` 约 0.7 GB），是否随提交材料附带取决于课程平台容量。

## 九、文档

| 文件 | 内容 | 对应版本 |
|---|---|---|
| `docs/course_requirements.md` | 课程交付要求核对，逐条给出考纲页码 | 不随源文件变化 |
| `docs/data_audit.md` | 冻结数据审计结果 | 当前 |
| `docs/analysis_protocol.md` | 分析协议与参数化处置记录 | 正式运行 |
| `docs/local-review-20261001.md` | 2026-10-01 的本地核验记录 | 2026-10-01 13:19 的源文件与运行结果 |
| `docs/presentation-notes.md` | 讲稿与预设问答 | 当前 |
| `docs/final-delivery-status.md` | 交付状态、证据来源、阅读核对结果与待确认事项 | 当前 |
| `docs/deliverable-verification.json` | 当前文件、数据哈希、诊断、PDF 与图片元数据的核验结果 | 当前 |

### 报告与演示的当前版本

两份 PDF 已于 2026-10-15 根据当前源文件编译。报告共 15 页，正文截至第 7 页，
参考文献从第 7 页开始，附录从第 9 页开始。演示共 12 页。两份 PDF 的字体全部嵌入，
文本可以提取，最终编译记录没有未定义引用、缺失字符或 Overfull。
页面视觉效果与演示计时尚未验证。本轮只运行文档编译和 PDF 检查工具，统计结果保持原样。

报告正文使用 `figures/fig02_rain_forest.pdf`（降雨森林图）、`figures/fig04_residual_acf.pdf`
（残差自相关）与 `report/generated/table_models.tex`（预测评分表）。其余图表在附录：
`figures/fig01_city_overview.pdf`、`figures/fig03_temperature_curves.pdf`、
`figures/figA2_all_city_curves.pdf`、`figures/fig05_forecast_scores.pdf`、
`figures/fig06_city_trajectories.pdf`、`figures/figA3_convergence.pdf`、
`figures/figA4_sensitivity.pdf`、`figures/figA5_posterior_predictive.pdf`，以及
`report/generated/table_effects.tex`、`report/generated/table_baselines.tex`、
`report/generated/table_baselines_compact.tex`。

演示为封面、8 页主讲与 3 页备答。逐页讲稿、讲述目标与预设问答见
`docs/presentation-notes.md`。
