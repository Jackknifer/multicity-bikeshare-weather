# 冻结数据审计

审计对象：`期中期末项目/midterm/data/clean/bikeshare_weather_daily.csv`
SHA-256 `6e30157ee8061f76142b95dd9ca7615bdea068f57c2fbd82796c7ac5e62af33e`，与计划给出的冻结值一致。
审计输出：`data/city_data_audit.csv`、`data/integrity_report.json`、`data/roster_comparison.json`。
复算命令：`.venv/bin/python src/run.py audit`。

## 完整性与规模

| 项目 | 数值 |
|---|---:|
| 行数 | 10,578 |
| 城市数 | 29 |
| 重复城市日期 | 0 |
| 缺失单元格 | 0 |
| 负使用量行 | 0 |
| 负降水量行 | 0 |
| 零使用日 | 575 |
| 含 0.5 小数的使用量行 | 4,936 |
| 研究期早于 2017-01-01 的城市 | Kaohsiung、Melbourne |
| 研究期早于 2017-12-31 结束的城市 | Kaohsiung、Melbourne |

## 城市研究期与日期间断

27 个城市覆盖 2017 年全年。Kaohsiung 与 Melbourne 覆盖 2016-07-02 至 2017-06-30。

- Brisbane、Toyama、Kazan、Vilnius 的研究期自 2017-01-02 起，缺少 2017-01-01 一天。
- **Kaohsiung 在 2016-09-27 缺失一天**，是全部 29 个城市中唯一在研究期内出现日期缺口的城市；缺口长度为 1 天，前后最近观测相隔 2 个日历日。
- Melbourne 在其研究期内无缺口（364 行对应 364 天）。

因此按日连续片段处理时间结构时，只有 Kaohsiung 需要在 2016-09-27 处切断并重新初始化。预测评估使用的 23 城市研究期内没有缺口。

## 城市名单

三种范围由冻结字段与日期范围直接确定，不由结果筛选：

| 范围 | 定义 | 城市数 | 行数 |
|---|---|---:|---:|
| 完整面板 | 全部城市 | 29 | 10,578 |
| 主分析 25 城市 | `continuous_system == 1` | 25 | 9,120 |
| 预测评估 23 城市 | `continuous_system == 1` 且研究期覆盖 2017-01-02 至 2017-12-31 | 23 | 8,393 |

- 25 城市减去 23 城市得到的两个城市是 Kaohsiung 与 Melbourne，原因是研究期只到 2017-06-30。
- 23 城市中没有出现新的城市。Goteborg、Kazan、Lillestrom、Vilnius 因为 `continuous_system == 0` 不在任一范围。
- `continuous_system` 由全期使用日占比得出，因此 23 城市名单以“全期持续产生使用记录”为条件。新的预测评估不按测试期使用量重新挑选城市，但这一条件本身如实披露。

## 使用量、温度与降水范围

- 日使用量中位数最低为 Créteil 的 11.5，最高为 Paris 的 76,103。
- 单日平均 UTCI 最低为 Kazan 的 −37.8 °C，最高为 Seville 的 34.3 °C；城市研究期平均 UTCI 的范围见 `data/city_data_audit.csv` 的 `utci_mean_c_median` 列。
- 城市平均日降水量最低为 Valence 的 0.76 mm，最高为 Toyama 的 7.15 mm。
- 主分析 25 城市中有 4 个零使用日：Créteil 3 天，Lund 1 天。响应使用 `log1p`，这 4 天对应 `log1p(0) = 0`。

## 与预测评估相关的结论

- 23 城市的训练窗与测试窗内均无日期缺口，块预测的连续日假设成立。
- 测试窗固定为 56 天，三个折的测试行数均为 23 × 56 = 1,288。
- 训练行数：开发折 1 为 4,529；开发折 2 为 5,817；末段为 7,105。
