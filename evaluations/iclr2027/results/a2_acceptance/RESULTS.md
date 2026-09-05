# A2 公共评测层验收结果

验收日期：2026-09-04  
结论：**PASS**

本阶段只冻结公共评测基础设施、正常校准回放和 M4 故障训练池；这些结果不是论文 sealed-test 数值。

## 验收摘要

| 项目 | 结果 |
|---|---:|
| Main-10 development dry run | 200/200 回合完成，0 基础设施错误 |
| Development 物理触发 | 94/94 eligible 回合触发；五个故障族均为 100% |
| Normal calibration | 每任务 50 条成功 nominal 回放，共 500 条 |
| M4 failure-train | 每任务 200 条，共 2,000 条 |
| Failure-train 物理触发 | 1,629/1,637 eligible，99.51% |
| 完整周期工件审计 | 2,700/2,700 通过 |
| 并发准入 | 48 workers，峰值 48，0 遗留基础设施错误 |
| sealed test | 未执行 |

Failure-train 条件触发统计：

| 故障族 | 总回合 | Eligible | 物理触发 | 条件触发率 |
|---|---:|---:|---:|---:|
| Actuation delay | 480 | 480 | 480 | 100.00% |
| Coordination delay | 180 | 154 | 154 | 100.00% |
| Environment change | 380 | 380 | 379 | 99.74% |
| Missed interaction | 480 | 363 | 363 | 100.00% |
| Relation loss | 480 | 260 | 253 | 97.31% |

未达到物理触发条件的计划回合仍按 intention-to-treat 协议保留；监控器输入不包含故障名称、强度、触发时刻或 auditor 标签。

## 验证内容

- Manifest、split 身份与固定规模经过统一校验；manifest 本身与方法无关，方法身份只写入结果。
- 每条周期记录均通过因果 feature schema、时间戳对齐、连续 cycle 编号、独立 audit 字段及文件哈希检查。
- Shadow monitor 无动作权限；Skill-Retry 使用共同且有界的恢复预算。
- 48 个槽使用独立 DISPLAY、临时目录、simulator 进程和单回合文件，通过全局动态队列调度。
- 相关测试共 60 项通过：ICLR/A2 基础设施 20 项，阶段六 fault/formal 兼容回归 40 项。

## 本阶段修复

Development dry run 中补齐了通用的物理触发时钟、阶段相关交互谓词、组合故障的双事件确认和增量周期日志。生成 failure-train 尾部时发现一个 X11 中断回合；调度器原本会在任务原队列耗尽后丢失该重试条目。现已改为重试时同时恢复任务在调度环中的成员资格，并增加非筛选作业的最终完整性检查。缺失回合单独补跑后，池规模恢复为 2,000/2,000，临时工件为 0。

这些修复均作用于公共任务/调度语义，没有加入任务专属成功规则或改变技术路线阈值。

交付前复核进一步冻结了 M4 的因果监督连接：周期行保存物理步前的 `feature_t` 与物理步后的 `audit_t`，公共读取器以 `audit_{t-1}` 的活动 violation 状态生成 `feature_t` 标签，首周期标签为0。读取器完整重读了2,000条序列和433,290个周期，得到97,211个因果正标签；全部身份、周期、哈希和可移植路径检查通过。原始轨迹和 A2 验收数值均未改变。

## 冻结工件

- 机器可读验收：[A2_ACCEPTANCE.json](A2_ACCEPTANCE.json)
- Manifests：`evaluations/iclr2027/manifests/`
- A-only normal calibration 候选与冻结只读视图：`evaluations/iclr2027/datasets/normal_calibration_candidates/`、`evaluations/iclr2027/manifests/main10_normal_calibration.jsonl`
- M4 failure-train：`evaluations/iclr2027/datasets/failure_train/`
- B 接口交付清单：`B_INTERFACE_HANDOFF.json`（10 个规范路径文件，包含统一 failure-train 读取器）
- B failure-train 交付清单：`B_FAILURE_TRAIN_HANDOFF.json`（4,001 个规范路径文件）
- B failure-train 唯一数据源：`evaluations/iclr2027/manifests/main10_failure_train.jsonl` 与 `evaluations/iclr2027/datasets/failure_train/{episodes,cycles}/`

两份清单只记录仓库相对路径、大小和 SHA256，不复制数据；均明确记录 `contains_normal_calibration=false`、`contains_sealed_test=false`。episode 摘要中的周期文件引用也已统一为相对自身位置的可移植路径。`artifacts/` 仅保留给 checkpoint、训练和校准产物。A2 到此停止；A3 需经用户确认后才启动。

## 交付状态

截至 2026-09-04，A2 接口与 failure-train 文件**尚未实际发送给服务器 B**。这是用户确认的延后交付安排，不影响 A3 在服务器 A 上继续进行 development-only 开发与验证。首次交付前若发现并修复了真实的数据或协议缺陷，必须重新生成并审计两份 handoff 清单；B 只接收最终冻结版本，避免无效重训。
