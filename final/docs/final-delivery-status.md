# 期末交付状态

更新日期：2026-10-15。运行标识：`formal_20260930`。工作目录：`STA5001/期中期末项目/final/`。

本轮根据用户授权编译报告与演示，更新两份 PDF，核对页数、文本、字体与编译记录。
演示使用适合页面宽度的公式、表格和文字排列，保留封面、8 页主讲和 3 页备答。
本轮没有运行统计代码、生成器、测试或绘图脚本，没有重新拟合或重新计算任何指标。
本文件只记录当前文件状态、证据来源与必要的待确认事项。

## 一、文件状态

### 1.1 当前源文件

| 文件 | 状态 |
|---|---|
| `report/final_report.tex` | 当前版本，已编译 |
| `report/references.bib` | 当前版本 |
| `report/generated/macros.tex` | 人工整理输入，顶部有中文来源注释 |
| `report/generated/table_effects.tex` | 人工整理输入 |
| `report/generated/table_models.tex` | 人工整理输入 |
| `report/generated/table_baselines.tex` | 人工整理输入 |
| `report/generated/table_baselines_compact.tex` | 人工整理输入 |
| `slides/final_slides.tex` | 当前版本，已编译 |
| `docs/presentation-notes.md` | 讲稿与预设问答 |
| `docs/final-delivery-status.md` | 本文件 |
| `README.md` | 当前版本 |
| `configs/paths.github.yaml` | 仓库用的相对路径配置 |
| `data/bikeshare_weather_daily.csv` | 清理后数据的交付副本 |

### 1.2 必须提交的补充证据

报告使用的部分结果不在正式运行目录内，而在独立核验目录中。以下文件属于提交材料，保持
原有相对路径：

| 文件 | 内容 |
|---|---|
| `logs/review-20261001/followup_checks.json` | 末段基线、AR-OLS 创新累积系数、预测积分稳定性、完整递推先验预测 |
| `logs/review-20261001/recomputed_metrics.csv` | 从已保存逐日预测独立复算的 48 组指标，报告末段模型点预测 MAE 的来源 |
| `logs/review-20261001/supplementary_final_baselines.csv` | 末段窗口基线的逐条预测记录 |
| `logs/review-20261001/independent_checks.json` | 独立核验结果，含冻结输入哈希、后验诊断复算与预测隔离检查 |
| `logs/review-20261001/audit_followup.py` | 生成 `followup_checks.json` 与 `supplementary_final_baselines.csv` |
| `logs/review-20261001/audit_saved_results.py` | 生成 `independent_checks.json` 与 `recomputed_metrics.csv` |

这六个文件都存在于当前目录。两个脚本的包内依赖与运行方式见 `README.md` 第 7.3 节。

不提交的内容：`.venv/`、`report/build/`、`slides/build/`、`logs/` 下的普通运行日志，
以及 `logs/review-20261001/` 下的缓存目录 `compiler-cache/`、`matplotlib/`、`numba/`。

**提交清单已明确，交付压缩包尚未生成，打包结果未验证。**

### 1.3 未修改的内容

- 统计代码 `src/` 全部文件、测试 `tests/`、驱动脚本 `scripts/` 均保持原样。
- 运行配置 `configs/paths.local.yaml`、`configs/splits.json`、`configs/models.json` 保持原样。
- `runs/`、`tables/`、`figures/`、`logs/` 的内容保持原样，没有覆盖任何正式运行记录。
- 期中目录与已归档的云端汇总保持原样。

### 1.4 历史材料及其对应版本

当前交付保留正式分析协议和与结果相关的技术核验记录。

| 文件 | 对应版本 |
|---|---|
| `docs/local-review-20261001.md` | 2026-10-01 13:19 的本地核验记录，针对当时的源文件与运行结果 |
| `docs/analysis_protocol.md` | 正式运行的分析协议，不随本轮文档修订变化 |

技术核验记录用于说明证据来源，当前 PDF 状态见下一节。

### 1.5 PDF 状态

| 文件 | 状态 |
|---|---|
| `report/final_report.pdf` | 当前源文件编译产物，共 15 页，正文截至第 7 页 |
| `slides/final_slides.pdf` | 当前源文件编译产物，共 12 页 |

报告参考文献从第 7 页开始，附录从第 9 页开始，正文满足 9 页上限。
两份 PDF 的字体全部嵌入，文本可以提取。最终编译记录没有未定义引用、缺失字符或
Overfull。报告保留浮动图表位置调整和 Underfull 提示。
超链接点击行为、页面视觉效果与演示计时尚未验证。提交准备仍需完成第五节的事项。

编译与核对步骤见 `README.md` 第 7.4 节，该步骤只更新文档，不需要运行统计代码。

### 1.6 报告与演示的结构

报告正文使用 `figures/fig02_rain_forest.pdf`、`figures/fig04_residual_acf.pdf` 与
`report/generated/table_models.tex`。附录集中放置
`figures/fig01_city_overview.pdf`、`figures/fig03_temperature_curves.pdf`、
`figures/figA2_all_city_curves.pdf`、`figures/fig05_forecast_scores.pdf`、
`figures/fig06_city_trajectories.pdf`、`figures/figA3_convergence.pdf`、
`figures/figA4_sensitivity.pdf`、`figures/figA5_posterior_predictive.pdf`，以及
`report/generated/table_effects.tex`、`report/generated/table_baselines.tex`、
`report/generated/table_baselines_compact.tex`。

演示为封面、8 页主讲与 3 页备答，讲述目标合计 360 秒，逐页讲稿见
`docs/presentation-notes.md`。图表文件与共享宏文件保持原样。演示备答表格使用短标题，
数字依据现有共享表格逐项保留；模型公式采用换行排列。

主讲第 4～6 页采用左侧图片区和右侧说明区。降雨森林图使用竖向空间；
温度曲线与残差图使用较宽的图片区。图片保持原始比例，数值和研究结论保持原样。
图片排版修订后已重新编译，演示仍为 12 页，最终记录没有 Overfull。
页面视觉效果尚未验证。

演示整体采用统一的紫色与金色主题、标题字号、表格间距和说明文字颜色。
研究问题与结论页采用相同的三栏结构。预测页区分开发窗口的评分与最终窗口的覆盖率。
封面标题分行排列，页脚显示课程与项目名称。三页备答材料依次编号。
讲稿已同步页面结构，并说明温度曲线只展示 M2 全期拟合。

## 二、证据来源

| 数值 | 来源文件 |
|---|---|
| 降水与温度关联（M0、M2 全期，25 城市） | `runs/formal_20260930/M0/full/effects.json`、`runs/formal_20260930/M2/full/effects.json` |
| 城市层面的关联、区间与 `p_rain_negative` | 同上文件的 `cities` 数组 |
| 内部峰值城市数、超范围城市数 | 同上文件的 `cities_with_interior_maximum_probability_above_0_95`、`cities_with_temp_10c_outside_support` |
| 开发折 1 的温度与降水规格差异 | `runs/formal_20260930/{M0,M1,M2}/dev1/effects.json` |
| 预测 WIS、对数密度、覆盖率、区间宽度 | `runs/formal_20260930/<model>/<fold>/metrics_overall.json` |
| 逐城与总体 MAE（复算） | `logs/review-20261001/recomputed_metrics.csv` |
| 末段窗口基线点预测 MAE | `logs/review-20261001/followup_checks.json` 的 `final_baselines_existing_implementation` |
| 配对七天区块区间 | `tables/paired_comparison.json` |
| 收敛诊断 | `runs/formal_20260930/<model>/<fold>/diagnostics.json` |
| 残差自相关 | `runs/formal_20260930/<model>/<fold>/residual_summary.json` |
| 完整递推先验预测（3.50%） | `logs/review-20261001/followup_checks.json` 的 `joint_prior_predictive`，实现见 `logs/review-20261001/audit_followup.py` |
| 预测积分稳定性 | `logs/review-20261001/followup_checks.json` 的 `predictive_integration_stability` |
| 城市趋势后验 | `runs/formal_20260930/M2/final/posterior_summary.csv` |
| 趋势的时间尺度与中心日 | `runs/formal_20260930/M2/final/preprocessing.json`、`src/data.py` |
| 先验分布与尺度 | `src/model.py`、`configs/models.json` |
| 城市行数与名单 | `data/city_data_audit.csv`、`data/roster_comparison.json` |

## 三、阅读核对结果

来源与内容核对依据源文件和已保存结果。PDF 核对使用 LaTeX 编译记录、pdfinfo、
pdftotext、pdffonts 和 SHA-256。正式 PDF 与 build 目录中的编译产物哈希相同。

### 3.1 文件存在性

| 检查项 | 结果 |
|---|---|
| 报告与演示源文件存在 | 通过 |
| `report/generated/` 五个共享文件存在 | 通过 |
| 报告引用的 10 张图存在 | 通过 |
| 报告与演示 `\input` 的表格文件存在 | 通过 |
| `report/neurips_2026.sty`、`report/references.bib` 存在 | 通过 |
| 1.2 节列出的六个补充证据文件存在 | 通过 |
| `configs/paths.github.yaml` 的 `data_csv` 指向包内文件且存在 | 通过 |
| 同一文件的 `data_sha256` 与包内副本一致 | 通过，均为 `6e30157e…f33e` |
| 同一文件的 `project_root`、`midterm_root`、`raw_zip`、`syllabus_pdf` | 指向仓库内材料，当前代码不读取 |

### 3.2 来源核对

| 检查项 | 结果 |
|---|---|
| 关联数值逐项对应 `effects.json`，模型与窗口明确 | 通过 |
| 预测评分与覆盖率逐项对应 `metrics_overall.json` | 通过 |
| 末段基线 MAE 逐项对应 `followup_checks.json` 与 `recomputed_metrics.csv` | 通过 |
| 配对区间逐项对应 `paired_comparison.json` | 通过 |
| 收敛数值逐项对应 `diagnostics.json` | 通过 |
| 先验表逐项对应 `src/model.py`，含噪声层级的 0.8 与 0.5 | 通过 |
| 趋势先验按实际生效值 `Normal(0, 0.3)` 与 `HalfNormal(0.3)` 记录，并说明配置中的 `beta_trend_city_sd` 未被读取 | 通过 |
| AR(1) 初始尺度按 `sigma * sqrt(nu / ((nu - 2) * (1 - phi^2)))` 记录，并声明未验证平稳性与敏感性 | 通过 |
| 完整递推先验预测的 3.50% 标注了来源文件与实现脚本 | 通过 |
| 截距抽样比例与完整递推比例分开陈述 | 通过 |
| 23 城市行数与 `data/city_data_audit.csv` 逐行加总一致（8,393） | 通过 |
| `README.md` 第 4 节的路径说明与 `load_paths`、`verify_frozen_input`、`command_audit` 的实际行为一致 | 通过 |
| `README.md` 第 7.3 节列出的两个补充核验脚本的包内依赖与脚本实际读取的文件一致 | 通过 |

### 3.3 内容一致性

| 检查项 | 结果 |
|---|---|
| `report/generated/macros.tex` 内没有重复定义 | 通过 |
| 报告引用的宏全部有定义 | 通过 |
| 演示引用的宏全部有定义 | 通过 |
| 报告中的 `\ref` 与 `\label` 一一对应 | 通过 |
| 引用键全部存在于 `references.bib`，且没有未被引用的条目 | 通过 |
| 温度比较分别写明窗口与城市名单，不再把不同窗口的数值并列成同一批数据的变化 | 通过 |
| 先验表的「两个尺度一致」限定在实际共用同一尺度参数的层级系数，噪声层级单独说明 | 通过 |
| 收缩机制写成一般性层级说明，不再依据城市使用量规模断言收缩力度 | 通过 |
| 共线只写成可能解释，没有据规格变化确认为原因 | 通过 |
| 贡献声明不主张任何未经确认的个人或共同贡献 | 通过 |
| 主讲 8 页、备答 3 页 | 通过 |
| 讲述时间标注为目标，未标注为已验证 | 通过 |
| 三个预测窗口与最终窗口限制保留 | 通过 |
| 主要降雨发现与其可信区间保留 | 通过 |
| `README.md` 第 3 节的提交范围与第 6 节的来源表、第 7.1 节的数字来源相互一致 | 通过 |
| 第 7.2 节的汇总命令同时包含 `--run-id` 与独立的 `--out` | 通过 |
| 第 7.2 节说明正式汇总文件继续由 `tables/` 提供，复现汇总写入独立目录 | 通过 |

### 3.4 本轮精简的验收项

| 检查项目 | 结果 |
|---|---|
| 三个研究问题含义一致，正文分别回答 | 通过，分别见第 5、6、7 节与讨论部分 |
| 降雨证据含定义、估计、区间、城市差异与森林图 | 通过 |
| 温度证据保留全期正关联与峰值不确定性，窗口名单清楚 | 通过 |
| 模型与先验含公式、规格差别、实际先验表与选择理由 | 通过 |
| 计算可靠性含抽样设置、验收阈值与实际结果 | 通过 |
| 时间诊断区分拟合残差与创新残差，不独立归因于 AR(1) | 通过 |
| 预测评估保留三个窗口，最终 WIS 与覆盖率完整可见 | 通过 |
| 基线比较在正文保留关键点预测结果，附录保留完整表格 | 通过 |
| 限制完整并与对应主张相连 | 通过 |
| 图表文件未删除，标签唯一，引用位置正确 | 通过，无重复标签，无未被引用的标签 |
| 无重复宏、无缺失宏、无失效引用 | 通过 |
| 演示为封面、8 页主讲、3 页备答，目标总时间 360 秒 | 通过 |
| 交付说明的证据清单、依赖、路径与独立汇总目录保持正确 | 通过 |
| 编译、页数、字体、版式、计时与打包均有准确状态 | 通过 |

### 3.5 本轮文字修订的验收项

| 检查项目 | 结果 |
|---|---|
| 全文没有把结构比较的结果单独归因于 AR(1) | 通过，报告摘要、第 4 节、第 7 节、演示第 6 页与第 8 页、讲稿第 3、6、8 页与问答第 4 题均使用统一含义 |
| 拟合残差与创新残差的名称准确 | 通过，报告、演示与讲稿中 M0 与 M1 写为 fitted residual 或拟合残差，M2 写为 innovation residual 或创新残差；讲稿问答第 4 题分别标注 0.317 与 0.066 属于哪一类残差 |
| 点预测简写没有形成错误排名 | 通过，报告与演示均改为逐项列出数值 |
| 位置误差和趋势解释均保持待验证的范围 | 通过，位置误差写为 possible contributor，趋势写为 possible source 且未做去除趋势的拟合 |
| 积分稳定性的文字与保存数值一致 | 通过，保留 0.33942 与 0.33731，并写明所列小数位上不完全一致 |
| 报告、幻灯片、讲稿和问答表述一致 | 通过 |
| 图表位置、标签、引用及共享宏保持有效 | 通过，无重复标签、无未被引用的标签、无缺失宏 |
| 最终窗口 WIS、覆盖率和点预测结果完整保留 | 通过 |
| 两份 PDF 与当前源文件一致 | 已编译更新 |
| 编译、页数、字体与页面溢出记录 | 已核对；视觉效果、实际计时和打包尚未验证 |

### 3.6 未执行的运行检查

以下检查本轮没有执行，因此没有结论。

| 未执行的检查 | 影响 |
|---|---|
| `src/verify_deliverables.py` | 本轮没有重新运行完整交付核验 |
| `tests/test_pipeline.py` | 测试结果沿用先前记录 |
| 图表与演示的视觉检查 | 图片可读性与幻灯片现场可读性未确认 |
| 演示计时 | 6 分钟为讲述目标，未经实际彩排计时 |

## 四、本轮未处理的统计问题

以下问题来自 `docs/local-review-20261001.md`，都需要重新计算。报告已按更窄的口径表述，
原始结果文件保留。

1. **汇聚消融的对比含额外变化。** `src/model.py` 的 `independent` 模式仍以可学习的
   population 参数作为城市系数均值，且天气系数的 `pooling` 设置同时作用于城市趋势。
   报告不把该消融作为研究贡献。
2. **趋势先验配置未进入模型。** 见 3.2 节。配置与运行记录保持原样。
3. **AR 初始分布的方差说明。** 报告已改为准确表述，初始分布的敏感性未做检验。
4. **时间基线的多步区间缺少创新累积。** 报告只用基线的点预测做比较，并写明该限制。
5. **完整先验预测与截距抽样应分别报告。** 报告已区分，并注明 M0 的完整先验预测未评估。
6. **驱动脚本缺陷。** `scripts/run_stage_final.sh` 给 `predict` 传入 `--roster`，而该子命令
   没有此参数。本轮未修改该脚本，`README.md` 第 7.2 节改为直接给出逐步命令。
7. **演示页面可读性。** 当前演示使用明确的字号和内容排列，最终编译记录没有 Overfull。
   页面视觉效果与现场可读性尚未确认。

## 五、待确认事项

以下事项缺少可核对的记录，本轮不填写、不标记为完成。

1. **学号与邮箱。** 报告首页与演示封面仍为占位内容（`00000000` 与
   `studentN@example.com`）。考纲第 2 页要求标题页列出全部组员姓名、学号与邮箱。
2. **演示分工。** 贡献声明已按组员提供的信息填写：XUE Yantong 负责数据搜寻、课题确定、
   实验设计、代码编写、实验运行、主要分析和材料编写；WANG Ziyu 与 ZHAO Jiashen
   共同参与数据处理、代码编写与检查、材料核验。
   具体讲述页次与问答分工尚待确认。
3. **演示当日出席安排。** 全体成员是否出席以及问答中的参与方式需要确认。
4. **数据集批准。** 项目沿用 Mendeley Data `10.17632/2nxpvtz935.1`（Version 1，CC BY 4.0）。
   教师是否已批准该数据集只能依据既有记录，本文件不声称已获批准。
5. **页面视觉效果与演示计时。** 文档已编译，页面视觉效果和实际讲述时间尚未确认。
6. **UTCI 数据条目的著者信息。** 本地核验记录写明原论文引用 di Napoli (2020) 与 DOI
   `10.24381/cds.553b7518`。`references.bib` 按数据提供方 Copernicus Climate Change
   Service 记为数据集条目，未凭推测补全个人著者名单。
7. **课程通知。** 演示日期按 2026-12-15、提交截止按 2026-12-20 23:59 编排；若教师另有
   通知，按最新通知执行。
