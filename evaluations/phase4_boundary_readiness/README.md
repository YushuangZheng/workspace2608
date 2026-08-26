# 阶段四正常边界就绪性检查

该检查使用五条正常训练示范回放阶段二—三闭环链，并逐技能边界审计阶段一保存的 `BoundaryModel`。它回答“现有正常数据和后验能否为阶段四标定与守卫实现提供输入”，不是入口守卫实现、任务成功率实验或故障评测。

为避免早期边界误差级联，进入每个新技能时假设上一边界已经正常提交：进度初始化到该技能首状态，同时携带此前已确认关系后验和稳定决策。固定离线回放使用产生当前观测的上一条示范动作；控制器查询仅作审计，不反馈到未执行的轨迹。

检查内容包括：

- 正常末端是否存在观测支持且进度后验进入终止窗口；
- 最终机器人目标是否位于阶段一保存的支持域；
- 本臂必要关系、边界关系和场景条件是否可观测并符合正常边界；
- 是否具备后续标定 `theta_local` 和控制周期确认长度 `H` 的输入。

正式配置保持 `minimum_explanation_score=0.001`。连续高斯支持以自身峰值为 1；离散关系在绝对解释度中同样除以当前软先验的可达峰值，但原始关系内积仍原样参与进度后验。

```bash
python -m evaluations.phase4_boundary_readiness.run \
  --config evaluations/phase4_boundary_readiness/config.json \
  --output evaluations/phase4_boundary_readiness/results/v7_reproduction
```

输出保存评测配置、实际阶段二/三运行配置、逐边界/逐条件/逐状态记录、环境与输入哈希、中文报告和 `SHA256SUMS`。当前结论见 [RESULTS.md](RESULTS.md)。
