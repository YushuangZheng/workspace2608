# 阶段四边界参数标定

本目录只使用正常训练示范，对每个技能边界标定 `theta_local` 和连续确认周期 `H`。回放使用原始 RLBench 控制周期序列，不使用经过 TAPAS/MiDiGaP 对齐后的状态数量代替真实时间。

标定规则：

- 正常末端保持分数的最低值形成正样本支持下界；
- 若正常边界前分数与末端支持可分，阈值取二者中点；
- 若二者有重叠，阈值取末端支持下界的保守比例，并由连续确认消除边界前偶发脉冲；
- `H` 至少覆盖配置的最短确认时间，并严格大于正常边界前最长连续就绪脉冲；
- 所有正常示范的末端保持必须能够连续满足最终 `H`，否则该边界标记为不可标定。

运行：

```bash
python -m evaluations.phase4_boundary_calibration.run \
  --config evaluations/phase4_boundary_calibration/config.json \
  --output evaluations/phase4_boundary_calibration/results/v3
```

正式结果见 `RESULTS.md` 和 `results/v3/`。运行器拒绝覆盖已存在的输出目录。
