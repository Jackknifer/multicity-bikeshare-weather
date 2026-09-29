# 多城市共享单车天气研究参考文档

## 项目概况

本项目研究 29 个城市共享单车日使用量与通用热气候指数（UTCI）、降水量之间的关系。项目题目为“基于层级贝叶斯模型的多城市共享单车天气效应研究”，课程为 STA5001 Bayesian & Computational Statistics。

主分析预先选取 25 个全年持续产生使用记录的城市，包含 9,120 个城市日。分析关注当地热舒适度、降水量与日使用量的关系，以及城市间天气效应差异。所有研究结论均按统计关联解释。

项目包含期中报告、课堂演示、探索性分析、数据处理脚本和层级 Student-t 模型试运行。期末计划使用 No-U-Turn Sampler（无掉头采样器，NUTS）完成正式推断、后验预测、时间外检验与敏感性分析。

## 研究问题与来源

### 研究问题

1. 相对当地研究期平均水平的热舒适度变化，与共享单车日使用量有何关系？
2. 降水量增加时，各城市的日使用量变化有多大差异？
3. 城市的热环境和降水背景，能否解释部分城市间差异？

### 核心论文与数据

核心论文为 Bean、Pojani 和 Corcoran 于 2021 年发表于《Journal of Transport Geography》的研究《How does weather affect bikeshare use? A comparative analysis of forty cities across climate zones》，DOI 为 [`10.1016/j.jtrangeo.2021.103155`](https://doi.org/10.1016/j.jtrangeo.2021.103155)。

原论文逐城拟合频率学广义加性模型（GAM），分析小时、季节、UTCI 和降水量。论文报告的温度响应通常在 UTCI 约 27～28 °C 达到峰值，之后下降；降水量通常与较低使用量相关。

数据来自 Mendeley Data 数据仓库，数据集题名为《How does weather affect bicycle use? A comparative analysis of forty cities across climate zones》，DOI 为 `10.17632/2nxpvtz935.1`，版本为 Version 1，发布日期为 2021 年 8 月 5 日，许可为 CC BY 4.0。项目固定使用 Version 1；Mendeley 页面将 2021 年 9 月发布的 Version 2（DOI `10.17632/2nxpvtz935.2`）标为最新版本。库存型系统的小时使用量来自每分钟站点库存快照；UTCI 与降水数据来自 Copernicus Climate Data Store。

本地原始数据包为 [`data/raw/bikeshare_weather_40cities.zip`](../data/raw/bikeshare_weather_40cities.zip)，大小为 6,400,538 字节，SHA-256 为 `fcee01fe98b7f4a12b6ffef2e71262a7c048b113f8a41940c65d6c1d37157740`。Version 1 的处理脚本、校验和及探索性结果均以此数据快照为准。

### 课程内容

项目涉及 Bayesian regression（贝叶斯回归）、hierarchical models（层级模型）、posterior computation（后验计算）、Markov chain Monte Carlo（马尔可夫链蒙特卡洛，MCMC）、Hamiltonian Monte Carlo（哈密顿蒙特卡洛，HMC）、收敛诊断、先验选择和 posterior predictive check（后验预测核验）。对应课程内容包括 Gaussian regression（高斯回归）、hierarchical Gaussian models（层级高斯模型）、MCMC foundations、Monte Carlo error、HMC 和 posterior predictive checks。Hoff 教材相关内容包括第 8 章多群体层级模型、第 10 章非共轭回归与 Metropolis-Hastings，以及第 11 章层级回归。

研究领域对应考纲中的 transportation and network data 与 environmental data。课程资料包括 [STA5001 课程考纲](../../../大纲与说明/STA5001-syllabus.pdf)、[STA5001 课程讲义](../../../教材或PPT/STA5001-notes.pdf) 和 [Hoff 教材](../../../教材或PPT/Hoff.pdf)。课程考纲要求小组在开始分析前取得教师的数据集认可；正式写作前应向教师提交本方案、核心论文和数据来源信息。

## 数据范围与处理

### 原始数据文件

原始数据包包含城市元数据、库存使用量、UTCI、降水量和论文 R 代码。处理所用的四个数据文件如下：

- `data/raw/bikeshare-weather-40cities/bs/bs-ll.csv`：40 个城市的国家、气候代码、经纬度及时区。
- `data/raw/bikeshare-weather-40cities/bs/stock-data.csv`：262,800 行小时使用量，含 30 个城市，每城 8,760 行。
- `data/raw/bikeshare-weather-40cities/utci/f_utci.csv`：40 个城市、8,760 个小时 UTCI。
- `data/raw/bikeshare-weather-40cities/utci/f_rain.csv`：40 个城市、8,760 个小时降水量。

论文 R 代码按元数据第 7～34 行和第 36 行选取 29 个库存型城市，名称均能与 `stock-data.csv` 完整匹配。另一库存城市 Stockholm 属于单站系统，原论文没有纳入分析；本项目沿用论文的 29 城市范围。

原始库存记录共 254,040 行，即 29 个城市各 8,760 行。27 个城市覆盖 2017 年；Melbourne 和 Kaohsiung 覆盖 2016 年 7 月至 2017 年 6 月。

### 时间处理与质量规则

论文代码按原始行序读取每城 8,760 个小时值，并从研究期首日 00:00 UTC 建立连续 UTC 小时序列。项目保持这一规则，将连续 UTC 时间转换为当地时间，再按当地日期汇总。该规则使使用量和天气数组逐小时对应，也保留源序列。

源文件有 3 个重复的城市、日期和小时标签。Brisbane 在 2017-10-29 02:00 有 2 条记录，使用量均为 143；Dublin 同一日期时刻有 2 条记录，使用量均为 1.5；Melbourne 在 2017-04-02 02:00 有 2 条记录，使用量分别为 32.5 和 41。Dublin 与 Melbourne 的日期符合夏令时切换时间，Brisbane 不实行夏令时，其重复原因未定。项目按论文规则保留小时数组顺序，不合并重复标签。

使用量没有缺失值和负值；其中 97,459 个小时值含 0.5 小数部分，40,623 个小时值为 0。UTCI 数组有 23 个缺失小时，涉及 Kaohsiung 和 Marseille。降水数组没有缺失值或负值。原始降水单位为米，处理时乘以 1,000 转为毫米。

UTC 时间转换为当地时间后产生 10,613 个城市日组合。日度记录须同时满足以下规则：

- 日期位于该城市源文件的名义研究期内。
- 当地日期包含 23～25 个小时，以容纳夏令时变化。
- 有效 UTCI 小时不少于 20 个。
- 降水小时数与该日小时数一致。

最终保留 10,578 个城市日。各项未通过原因可以同时发生：28 个日期位于名义研究期以外，34 个日期的小时数超出 23～25，33 个日期少于 20 个有效 UTCI 小时，没有降水记录不完整的日期。最终数据含 47 个 23 小时日期、10,506 个 24 小时日期和 25 个 25 小时日期；没有缺失单元、负使用量、负降水量或重复城市日期。

### 主分析样本

完整日度数据包含 575 个零使用日期。零值主要来自 Goteborg、Kazan、Lillestrom 和 Vilnius，分别有 47、236、148 和 140 天。持续零值可能来自季节停运、站点记录状态或实际没有使用，现有数据无法区分这些原因。

主分析按有效日期中 `usage > 0` 的占比至少 99% 选取城市，共 25 个城市、9,120 个城市日。完整 29 城市数据仍保存在同一日度 CSV 中。主分析包含 4 个零使用日期：Creteil 有 3 天，Lund 有 1 天。响应值采用 `log1p` 变换，Student-t 观测分布用于容纳少量极端日期。

## 日度数据字典

日度文件为 [`data/clean/bikeshare_weather_daily.csv`](../data/clean/bikeshare_weather_daily.csv)，共 10,578 行、28 列。每行代表一个城市在一个当地日期的估计使用量和天气信息。

### 城市与日期

| 字段 | 定义 |
|---|---|
| `city` | 城市名称。 |
| `country` | 国家或地区名称。 |
| `climate_code` | 数据源提供的 Trewartha 气候代码。 |
| `latitude` | 城市纬度，单位为十进制度。 |
| `longitude` | 城市经度，单位为十进制度。 |
| `timezone` | IANA 时区名称。 |
| `date` | 城市当地日期，格式为 `YYYY-MM-DD`。 |

### 使用量

| 字段 | 定义 |
|---|---|
| `usage` | 当地日期内小时估计使用量之和，来源为站点库存变化，可含 0.5 小数。 |
| `usage_log1p` | `log(1 + usage)`，主模型的响应变量。 |
| `active_day_share` | 该城市有效日期中 `usage > 0` 的比例，即非零使用日占比。 |
| `continuous_system` | `active_day_share` 达到 99% 时为 1，否则为 0；该字段表示项目筛选结果，不代表已核实运营日历。 |

### UTCI

| 字段 | 定义 |
|---|---|
| `utci_mean_c` | 当地日期内小时 UTCI 平均值，单位为 °C。 |
| `utci_min_c` | 当地日期内小时 UTCI 最低值，单位为 °C。 |
| `utci_max_c` | 当地日期内小时 UTCI 最高值，单位为 °C。 |
| `utci_city_mean_c` | 该城市所有有效日期的 `utci_mean_c` 平均值。 |
| `utci_anomaly_10c` | `(utci_mean_c - utci_city_mean_c) / 10`。 |
| `utci_anomaly_sq` | `utci_anomaly_10c` 的平方，未在城市内中心化。 |
| `utci_within_city_z` | `utci_mean_c` 在城市内标准化后的值。 |

模型二次项由 `code/fit_pilot_model.py` 计算：对 `utci_anomaly_10c` 的平方减去城市内均值。该项与模型中的 $Q_{it}$ 一致。模型降水项使用 `precipitation_log1p` 减去城市内均值，再除以城市内标准差。

### 降水、日历与完整性

| 字段 | 定义 |
|---|---|
| `precipitation_mm` | 当地日期内小时降水量之和，单位为 mm。 |
| `precipitation_log1p` | `log(1 + precipitation_mm)`。 |
| `weekday` | 星期序号，星期一为 0，星期日为 6。 |
| `weekend` | 星期六或星期日为 1，其余日期为 0。 |
| `day_of_year` | 一年中的日期序号，通常为 1～365，闰年可为 366。 |
| `season_sin` | `sin(2π × day_of_year / 365.25)`。 |
| `season_cos` | `cos(2π × day_of_year / 365.25)`。 |
| `observed_hours` | 当地日期包含的连续 UTC 小时数，保留值为 23、24 或 25。 |
| `utci_observed_hours` | 当日有效 UTCI 小时数，至少为 20。 |
| `rain_observed_hours` | 当日有效降水小时数，与 `observed_hours` 相同。 |

## 模型方案与试运行

### 层级 Student-t 回归

令城市 $i$ 在日期 $t$ 的估计使用量为 $U_{it}$，响应变量为

$$
y_{it}=\log(1+U_{it}), \qquad y_{it}\sim t_{\nu}(\mu_{it},\sigma_i).
$$

位置参数为

$$
\begin{aligned}
\mu_{it}={}&\alpha_i+\beta_{1i}T_{it}+\beta_{2i}Q_{it}+\beta_{3i}R_{it}+\beta_{4i}W_{it}\\
&+\beta_{5i}\sin(2\pi d_{it}/365.25)+\beta_{6i}\cos(2\pi d_{it}/365.25).
\end{aligned}
$$

其中，$T_{it}$ 为日均 UTCI 与该城市研究期均值之差，以 10 °C 为单位；$Q_{it}$ 为 $T_{it}^2$ 的城市内中心化值；$R_{it}$ 为 `log1p` 降水量减去城市均值后除以城市内标准差；$W_{it}$ 为周末指示变量；$d_{it}$ 为一年中的日期序号。

城市截距、天气系数、周末系数和季节系数均有城市专属取值，并共享正态群体分布。该结构通过部分汇聚估计城市差异。温度线性项与二次项的群体中心随城市研究期平均 UTCI 改变；降水系数的群体中心随城市研究期平均降水量改变。各城市有独立残差尺度。模型使用非中心参数化，且 $\nu=2+\operatorname{Exponential}(0.1)$。

温度项与季节项存在较强共线关系。仅用年周期正弦和余弦解释城市内 UTCI 异常值时，25 个城市的 $R^2$ 从 Dublin 的 0.62 到 Toyama 的 0.92，平均为 0.76。季节项用于控制年周期；温度系数须在季节项纳入模型时解释。

### 先验与正式计算设置

先验均在 `log(1 + usage)` 尺度上：群体截距为 `Normal(7, 2)`，城市截距标准差为 `HalfNormal(2)`；10 °C 温度变化的群体系数为 `Normal(0, 0.5)`；温度二次项为 `Normal(0, 0.3)`；标准化降水量为 `Normal(0, 0.3)`；周末系数为 `Normal(0, 0.4)`；两个季节系数为 `Normal(0, 0.8)`。温度与降水背景调节系数使用标准差为 0.2～0.3 的零均值正态先验；城市间标准差使用相应尺度的 HalfNormal 先验；`log(σ_i)` 使用层级正态先验。

正式后验推断计划使用 PyMC 6.3.2、nutpie 0.16.11 和 NUTS，随机种子为 27，`target_accept=0.92`。设置为 4 条链，每条链 2,000 次 warmup 和 2,000 个保留样本。计算核验包括 $\widehat R$、bulk ESS、tail ESS、Monte Carlo standard error（蒙特卡洛标准误，MCSE）、divergence（发散）和 tree depth（树深度）。目标为 $\widehat R\le1.01$、bulk ESS 与 tail ESS 均至少为 400、没有 divergence，并确认 MCSE 相对后验标准差足够小。

正式分析将进行 prior predictive check（先验预测核验），检查日使用量范围、天气变化对应的倍数变化和城市差异。若先验预测大量超出数据与领域知识支持的范围，将在查看观测结果前调整先验尺度。

### 已有短链结果

项目已在 9,120 个观测上运行 2 条链，每条链含 200 次 warmup 和 200 个保留样本。短链的 divergence 为 0。8 个关键超参数的最大 $\widehat R$ 为 1.037，最低 bulk ESS 为 94.2；421 个参数整体的最大 $\widehat R$ 为 1.321，最低 bulk ESS 为 6.1。城市截距项的混合情况最弱；400 个抽样中有 18 个触及最大 tree depth 10。短链只用于模型运行可行性检查，不能支持正式研究结论。

### 正式模型核验计划

1. 先验预测：检查先验隐含的使用量、天气效应和城市差异。
2. 后验预测：比较城市日使用量的均值、标准差、分位数、极低值、周末差异和季节变化。
3. 时间外检验：每个城市保留研究期末 8 周，比较预测区间覆盖率、MAE 和 log predictive density。
4. 残差检查：按城市检查时间相关性、季节结构和极端天气日期。
5. 样本范围分析：用全部 29 个城市并保留零使用日，比较天气系数与 25 城市主分析的结果。
6. 先验敏感性：将天气系数先验尺度分别乘以 0.5 和 2，检查核心后验量。
7. 观测分布敏感性：使用 Gaussian（高斯）观测分布重复主分析，检查 Student-t 选择的影响。
8. 季节项敏感性：去除年周期正弦和余弦项，检查温度系数的变化。

### 主要后验量

- UTCI 高于城市研究期均值 10 °C 时，日使用量的百分比变化。
- 日降水量高于该城市研究期典型水平 10 mm 时，日使用量的百分比变化。
- 各城市观测温度范围内的预测使用量最高点及其可信区间。
- 群体降水效应低于 0 的后验概率。
- 温度和降水效应的城市间标准差。
- 热环境与降水环境调节系数的后验分布。
- 小系统城市参数经过部分汇聚后的不确定度变化。

## 期中探索性结果

以下数值记录在 `results/eda_midterm.json`，用于描述性分析。

| 项目 | 结果 |
|---|---|
| 分析日期范围 | 2016-07-02～2017-12-31。 |
| 日使用量城市中位数范围 | Créteil 为 11.5，Paris 为 76,103，相差约 6,618 倍。 |
| 城市平均 UTCI 范围 | −2.57～26.58 °C。 |
| 城市平均降水量范围 | 0.76～7.15 mm。 |
| UTCI 绝对温度曲线峰值 | 27.5 °C，中心化 `log1p` 值约 +0.26。 |
| 最冷与最热分箱 | −22.5 °C 分箱约 −0.26；32.5 °C 分箱约 −0.14。 |
| UTCI 异常值曲线峰值 | 高于当地研究期均值约 22.5 °C，中心化 `log1p` 值约 +0.45。 |
| 最重降水分箱 | 相对城市均值约 −30%；相对干燥分箱约 −41%。 |
| 季节峰谷 | 峰值约在 10 月中旬，谷值约在 12 月底，幅度约 0.84 个中心化 `log1p` 单位。 |
| 周末差异 | 25 个城市中有 23 个周末使用量较低；城市平均差异约 −0.55 个中心化 `log1p` 单位。 |
| 气候代码 | Aw、Cf、Cfb、Cs、Dc、Do，共 6 类。 |

降水分箱的“相对城市均值”与“相对干燥分箱”使用不同基准。UTCI 绝对温度曲线便于与论文报告的 27～28 °C 峰值比较；模型使用城市内 UTCI 异常值，以城市截距表示气候背景。

## 期中交付与问答

### 报告与演示

期中报告为 NeurIPS 2026 模板，正文 4 页，参考文献 1 页。演示共 13 页，前 9 页用于 5 分钟讲述，后 4 页用于问答。12 张图均有 PDF 与 300 dpi PNG，报告使用 3 张，幻灯片使用 5 张。幻灯片封面使用学校原始双语校名图形，其中包含校徽和中英文校名；组员姓名、邮箱及日期的版面位置与本地留存版一致。页脚分为等宽三栏，显示 CUHK-Shenzhen、组员姓名和页码。

期中交付核验 136 项全部通过。检查范围包括样本、分箱曲线、季节幅度、周末效应、温度与季节项关系、图表文件、报告页数、字体嵌入、文字边界、引用解析和原始校名图形。

报告中的 Contribution statement 只有标题，正文仍待填写；提交前应补充每位组员的工作，并重新编译报告。

### 5 分钟讲述时间

| 页码 | 内容 | 时长 |
|---|---|---|
| 1 | 封面 | 10 秒 |
| 2 | 研究问题与逐城建模的两个局限 | 32 秒 |
| 3 | 数据来源、许可与 29 城市范围 | 24 秒 |
| 4 | 小时数据处理与质量控制 | 24 秒 |
| 5 | 城市规模差异与持续系统筛选 | 24 秒 |
| 6 | 温度与降水响应 | 52 秒 |
| 7 | 季节和周末效应的城市差异 | 33 秒 |
| 8 | 期末模型结构与先验 | 43 秒 |
| 9 | 结论与提问 | 18 秒 |

合计 260 秒，即 4 分 20 秒，为 5 分钟上限留出 40 秒供换页、停顿和现场交流。讲述重点是 27.5 °C 附近的温度曲线峰值、降水与使用量的负向关系，以及城市间差异为何需要层级模型。

### 问答要点

**为何主分析使用 25 个城市？** 完整面板有 575 个零使用日，其中 571 个集中在 4 个城市。长时间零值可能来自季节停运、站点记录状态或实际无使用，现有资料无法区分。预先采用使用日占比至少 99% 的规则，得到 25 个城市和 9,120 个城市日。完整 29 城市数据仍在日度 CSV 中，期末将使用全部城市进行样本范围分析。

**温度曲线为何同时报告绝对 UTCI 与异常值？** 绝对温度便于与论文的 27～28 °C 峰值比较。模型使用相对城市均值的异常值，让城市截距表示气候差异；异常尺度描述天气相对当地常态的关系。

**日度数据如何从小时数据生成？** 每城读取源文件中的 8,760 个使用量，按论文规则从研究期首日 00:00 UTC 建立连续小时序列。之后转换到城市当地时间并按当地日期汇总。三个重复时间标签分别位于 Brisbane、Dublin 的 2017-10-29 02:00 和 Melbourne 的 2017-04-02 02:00；按原始序列保留。

**使用量数据有哪些限制？** 使用量由每分钟站点库存快照估计，29 个城市中有 97,459 个小时值带 0.5 小数部分。库存变化也可能包含人工调度造成的车辆移动。

**为何采用 Student-t 观测分布？** 日使用量可能受节假日、大型活动和数据异常影响。短链估计的自由度约为 2.4，低于先验均值 12，提示重尾观测分布可能适用；该短链未达到正式推断要求。期末将用 Gaussian 观测分布重复主分析。

**模型目前运行到什么程度？** 层级 Student-t 模型已在 9,120 个观测上完成 2 条链、每条 200 次抽样的试运行，divergence 为 0。8 个关键超参数的最大 $\widehat R$ 为 1.037、最低 bulk ESS 为 94.2；全部 421 个参数的最大 $\widehat R$ 为 1.321、最低 bulk ESS 为 6.1；400 个抽样中有 18 个触及最大 tree depth。正式推断计划使用 4 条链，每条链 2,000 次 warmup 和 2,000 个保留样本，并逐项检查收敛。增加抽样次数可降低 Monte Carlo 误差；树深和混合情况仍须单独检查。

## 研究范围与解释限制

数据适合分析多城市天气与共享单车日使用量的关联：城市、天气和使用量按时间对应；29 个城市具有相同的 8,760 小时结构；25 个连续系统提供 9,120 条主分析记录；城市规模差异可由层级模型部分汇聚。UTCI 汇总温度、湿度、风和太阳辐射对热感受的影响。

研究结论受以下范围限制：

- 使用量为估计值，可能包含人工调度造成的车辆移动。
- 每个城市只有一年数据，研究期平均天气不能代表长期气候常态。
- 城市系统规模、基础设施、收费和运营政策缺少完整协变量。
- 日度汇总无法呈现小时通勤高峰。
- 相邻日期可能存在时间相关性，须在模型核验中检查。
- 当前研究设计支持关联分析，无法单独识别天气变化的因果效应。
- 主分析针对持续产生使用记录的系统，不代表长时间零使用的系统。

## 期末项目安排

期末报告正文计划控制在 9 页以内，涵盖研究问题与已有论文、层级 Student-t 回归和先验、NUTS 实现与计算诊断、群体天气效应和城市差异、后验预测、时间外检验、敏感性分析，以及城市交通含义与研究限制。期末演示计划控制在 6 分钟内，附加页用于先验、诊断和城市结果。

## 环境与运行命令

当前项目环境使用 Python 3.12。主要依赖版本记录在 [`requirements.txt`](../requirements.txt)：ArviZ 1.3.0、matplotlib 3.11.2、netCDF4 1.7.4、NumPy 2.5.3、nutpie 0.16.11、pandas 3.0.6 和 PyMC 6.3.2。

创建环境并安装依赖：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

生成日度数据、审计结果、探索性分析与图表：

```bash
.venv/bin/python code/prepare_bikeshare_weather.py
.venv/bin/python code/eda_midterm.py
```

运行模型试验的入口：

```bash
code/run_pilot_model.sh --draws 200 --tune 200 --chains 2
```

正式模型计划采用以下参数：

```bash
code/run_pilot_model.sh --draws 2000 --tune 2000 --chains 4
```

报告使用 `pdflatex` 和 `bibtex`。在项目根目录执行：

```bash
cd report
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_report.tex
bibtex build/midterm_report
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_report.tex
cp build/midterm_report.pdf midterm_report.pdf
cd ..
```

幻灯片使用 `xelatex`、`fontspec` 和 `xeCJK`。主题使用 Palatino、Helvetica Neue、Songti SC、Heiti SC 和 TeX Live 自带的 FandolKai 字体。Linux 或云端编译时，应为前四种字体指定可用字体。已生成的 PDF 使用嵌入字体。

```bash
cd slides
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_slides.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build midterm_slides.tex
cp build/midterm_slides.pdf midterm_slides.pdf
cd ..
```

重建探索性分析、报告和幻灯片：

```bash
./code/build_midterm.sh
```
