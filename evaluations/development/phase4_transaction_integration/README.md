# 阶段四联合事务集成验证

本目录验证真实 LiftTray 任务模型中的联合边界是否贯通以下正式运行链：

```text
五条正常示范的技能 1 末端观测
→ BeliefUpdater 终端保持信念
→ ClosedLoopExecutionController pre-action 更新
→ MultiArmBoundaryController 同快照入口守卫
→ TransitionTransactionCoordinator 原子提交
```

为覆盖异步就绪路径，评测只把右臂进度后验在前两个控制周期设为同技能内的非末端状态；几何观测、关系后验、场景状态和左臂信念仍来自正常示范。预期行为是：

- 左臂先达到入口许可时，事务组整体保持，两臂游标均不改变；
- 右臂恢复真实末端信念并连续满足正式 `H` 后，两臂在同一 tick 原子提交；
- 不允许任何单臂部分提交。

运行命令：

```bash
python -m evaluations.development.phase4_transaction_integration.run \
  --config evaluations/development/phase4_transaction_integration/config.json \
  --output evaluations/development/phase4_transaction_integration/results/v1
```

运行器拒绝覆盖已有输出目录。正式结果见 `RESULTS.md` 和 `results/v1/`。
