# 阶段二—三组件级 A/B 评测报告

## 评测边界

Offline counterfactual component benchmark on perturbations derived from the five normal training demonstrations; not an independent test set and not an end-to-end simulator success-rate evaluation. V3 admits break-link trials only inside a formal LINK origin interval, excludes LINK_PENDING and isolated per-state linked evidence, and restores the original eight-state perturbation plus four-state aftermath window after event-consistent relation propagation makes valid intervals available; no failure-set metric is used for parameter selection.

覆盖 8 个任务、12 套机械臂模型、5 条正常示范索引。时间试验 300 组，关系试验 190 组。

## 时间扰动结果

| 场景 | 试验数 | 固定时钟 MAE | 在线估计 MAE | 相对降低 | 新方法胜率 |
|---|---:|---:|---:|---:|---:|
| normal | 60 | 0.0000 | 0.0636 | 0.00% | 0.00% |
| skip | 120 | 1.9843 | 0.1804 | 90.91% | 100.00% |
| stutter | 120 | 3.0955 | 0.1569 | 94.93% | 100.00% |

## 关系扰动结果

固定流掩码基线没有在线关系失配判定，因此其检测率按机制定义为0；对照组用于衡量动态角色误报。

| 场景 | 试验数 | 基线检测率 | 动态角色检测率 | 平均阻断率 | 检测延迟中位数 |
|---|---:|---:|---:|---:|---:|
| break_link_control | 45 | 0.00% | 0.00% | 0.93% | — |
| break_link_perturbed | 45 | 0.00% | 100.00% | 99.81% | 0.0 |
| false_coupling_control | 50 | 0.00% | 0.00% | 0.00% | — |
| false_coupling_perturbed | 50 | 0.00% | 100.00% | 100.00% | 0.0 |

## 可复现性

配置、逐试验指标、按任务/机械臂聚合的统计、压缩逐周期轨迹、源码/模型/示范哈希、环境信息、图表和 SHA256SUMS 均与本报告同目录保存。

## 解释限制

该结果只支持在线进度估计、动态角色和推进阻断的组件级结论。输入由训练用正常示范派生，且离线动作不会改变后续环境；不得将本结果表述为独立测试集泛化、恢复成功率或完整任务成功率。

## 未构造的关系条件

- no stable formal linked interval: 15
- no stable selected external segment: 10
