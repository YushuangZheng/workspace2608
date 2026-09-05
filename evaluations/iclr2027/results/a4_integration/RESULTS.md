# A4 交付与冻结：A 侧部分验收

状态：**A 侧 PASS，等待服务器 B 交付；A4 尚未整体验收。**  
机器可读记录：`A4_PARTIAL_STATUS.json`

本轮仅完成 A4 中不依赖 M3/M4 的工作。没有读取 sealed test，没有启动 A5，也没有生成论文正式成功率。

## 已完成

- 冻结统一监控器正常误报口径：每任务 50 条成功正常回放、5% episode 级预算；阈值是连续 `H` 周期统计量的有限样本 split-conformal 上分位数。不可用观测会打断连续计数。
- 完成服务器 B 交付验证器：只接受 B 所有路径，逐文件校验大小和 SHA256，绑定 M4 checkpoint、方法配置、训练预算、训练种子和 feature schema，并拒绝覆盖 A 所有目录。
- 完成通用 shadow replay：监控器只接收因果 `feature`，不接收 `audit`；输出逐周期 score/alarm/threshold/persistence/metadata，并逐回合验证动作字节等价。
- 完成 M2 正式正常标定：10 个任务 × 50 条，共 500 条成功回放和 106,579 个周期；模型权重未更新，故障标签与 sealed test 均未读取。
- 更新并复核尚未发给 B 的两份交付清单：接口清单现为 12 个文件；failure-train 清单仍为 4,001 个文件。两者逐文件校验均为 0 错误，且不包含 normal calibration 或 sealed test。A/B 交付方式已冻结为保持仓库相对路径的清单限定文件拷贝，不再使用 Git patch 或整仓同步。

## M2 正式标定结果

| 任务 | 阈值 | H | 标定集报警回合 | 可用/总周期 |
|---|---:|---:|---:|---:|
| Bimanual Handover Item | 9.95624254 | 5 | 1/50 | 12,528/12,528 |
| Bimanual Lift Tray | 8.54342902 | 5 | 1/50 | 5,400/5,400 |
| Bimanual Put Bottle in Fridge | 25.1473344 | 5 | 1/50 | 13,650/13,650 |
| Bimanual Sweep to Dustpan | 36.786654 | 5 | 1/50 | 6,600/6,600 |
| Close Jar | 1789.46446 | 5 | 1/50 | 8,100/8,100 |
| Insert Onto Square Peg | 11504.2468 | 5 | 1/50 | 8,550/8,550 |
| Open Drawer | 40.8245372 | 5 | 1/50 | 4,602/4,602 |
| Place Cups / 3 | 261.167732 | 5 | 1/50 | 30,550/30,550 |
| Stack Cups | 221.438542 | 5 | 1/50 | 11,150/11,150 |
| Sweep to Dustpan | 2.35226117 | 5 | 1/50 | 5,449/5,449 |

5% 是冻结的允许上限；有限样本秩和严格大于阈值的报警规则使实际标定集结果为每任务 1/50，即 2%。这不是用故障表现或单个任务成功率反向调参。

原 500 条成功回放早于 M2 世界系流 marginal 的持久化。为避免用新模拟器回放替换已冻结成功轨迹，本轮只依据每周期已记录的 StateId、流掩码、PoE 权重和同一冻结模型，在内存中重建遗漏的确定性 marginal；不修改任何源记录。该重建在 10 个任务的当前格式日志上逐帧交叉验证，PoE 权重最大绝对误差为 `9.16e-15`。

## Development 集成检查

- M2 加载正式 calibration artifact 后完成 Main-10 每任务 1 条 nominal dry run：10/10 落盘，0 基础设施错误，8/10 任务成功。
- OpenDrawer 与 PlaceCups 两条骨干未完成任务，并分别触发 M2 报警和一次共享 Skill-Retry；该小样本只用于接口与日志验收，不用于估计方法性能或调整阈值。
- 对上述 10 条、2,223 个周期重新执行 shadow replay：报警、阈值、持久计数和 metadata 与在线记录完全一致；分数最大绝对差为 `2.33e-10`，来自 JSON 浮点往返；动作输入/输出 SHA256 全部一致。
- A2/A3/A4 聚焦测试 `29 passed`，相关目录通过 `compileall`。

## 尚待 B 交付后完成

1. 核验 M3/M4 代码、方法配置、M4-200 三种子 checkpoints、训练记录和 golden outputs；
2. 在 A 上完成 checkpoint 前向兼容、golden score 复现与 M3/M4 的 10 nominal + 10 perturbed dry run；
3. 仅用 A 保存的 500 条正常回放标定 M3 和每个 M4 checkpoint；
4. 用最终监控器、auditor 和正式日志负载完成 48-worker 准入；
5. 生成完整 `A4_ACCEPTANCE.json` 并向用户汇报。只有再次得到用户确认后才能进入 A5。

服务器 A 尚未把 A2/A4 交付清单列出的文件发送给 B，服务器 B 也尚未向 A 交付 M3/M4。本状态不会被表述为 A4 已全部完成。
