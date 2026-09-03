# 阶段四边界参数标定

本目录只使用正常训练示范，对每个技能边界标定 `theta_local` 和连续确认周期 `H`。回放使用原始 RLBench 控制周期序列，不使用经过 TAPAS/MiDiGaP 对齐后的状态数量代替真实时间。

标定规则：

- `theta_local/H` 只读取本地完成度，不以边界关系或场景守卫是否已经放行筛选本地正样本；
- 正常末端保持分数的最低值形成正样本支持下界；
- 若正常边界前分数与末端支持可分，阈值取二者中点；
- 若二者有重叠，阈值取末端支持下界的保守比例，并由连续确认消除边界前偶发脉冲；
- `H` 至少覆盖配置的最短确认时间，并严格大于正常边界前最长连续就绪脉冲；
- 所有正常示范的末端保持必须能够连续满足最终 `H`，否则该边界标记为不可标定；
- 最终守卫复核区分“应放行”和“单向跨臂条件尚未满足、应继续等待”，两者都必须可观测且与 BoundaryModel 一致。

运行：

```bash
python -m evaluations.development.phase4_boundary_calibration.run \
  --config evaluations/development/phase4_boundary_calibration/config.json \
  --output evaluations/development/phase4_boundary_calibration/results/v5
```

当前正式完整结果见 `RESULTS.md` 和 `results/v5/`。对侧末端执行依赖筛选与通用协作就绪性验收见 `evaluations/development/peer_execution_dependency/results/v1/`。旧标定结果及 WipeDesk 单任务过程性标定已删除，避免与最终 40 个边界配置混用。运行器拒绝覆盖已存在的输出目录。
