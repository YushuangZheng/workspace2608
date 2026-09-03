# 阶段四联合事务集成验证结果

正式结果位于 `results/v1/`。评测使用真实 LiftTray 任务模型、正式阶段四运行配置和五条正常示范，并对右臂进度后验施加两周期受控延迟来覆盖异步就绪分支。

关键结论：

- 真实联合事务端到端通过 5/5；
- 左臂先达到入口许可时，5/5 条示范中的两臂游标都保持在技能 1；
- 20 个验证周期中没有发生任何单臂部分提交；
- 右臂恢复真实终端信念并满足 `H=2` 后，5/5 条示范均在同一 tick 将两臂原子提交到技能 2；
- `results/v1/SHA256SUMS` 全部通过。

该结果验证 `BeliefUpdater → ClosedLoopExecutionController → MultiArmBoundaryController → TransitionTransactionCoordinator` 的真实模型集成闭环，不代表完整 RLBench 在线动作执行成功率。
