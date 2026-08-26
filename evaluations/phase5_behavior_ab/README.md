# 阶段五受控行为 A/B

该评测在冻结 V4 任务模型和五条正常示范场景上，以确定性理想执行器模拟阶段五组件行为：

- 静止 Pending 被动等待 vs 主动反向探测；
- 不执行恢复 vs 事件级 LINK/UNLINK 恢复；
- 直接恢复旧时钟 vs 当前完整状态重入；
- 无守卫跨技能跳转 vs 阶段四许可约束。

运行：

```bash
PYTHONPATH="$PWD:$PWD/source" python evaluations/phase5_behavior_ab/run.py \
  --output evaluations/phase5_behavior_ab/results/v5
```

正式结果保存在 `results/v5/`。评测使用7个真实 Pending、11个正式 LINK、
3个正式 UNLINK及其五条正常示范场景，并同时模拟 linked/external 两种已知物理结果、
无响应执行器、正向时钟滞后和反向恢复。它是组件级受控 A/B，不等同于阶段六完成后的
RLBench 在线任务成功率实验。重入评测还在同一次正常回放中对比参考机器人兼容度阈值
0.01和正式标定阈值0.001，并保存选择、精确命中、错误选择和MAE变化。
