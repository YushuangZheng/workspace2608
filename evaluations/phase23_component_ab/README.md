# 阶段二—三组件级 A/B 评测

该评测从冻结 DynaMAC V4 的五条正常训练示范派生确定性的受控扰动，对比：

- 固定一步一状态时钟与阶段二在线进度估计；
- DynaMAC 固定流掩码与阶段三动态关系角色、精度权重及推进阻断。

它不会训练模型、启动 RLBench/CoppeliaSim、修改原始示范或覆盖 V4 正式结果。输入来自训练示范，且反事实动作不会改变后续离线环境，因此结果只支持组件级能力，不代表独立测试集泛化、仿真任务成功率或恢复成功率。

当前正式结果为 `results/v7`。运行时显式读取并在结果目录保存：

- `configs/closed_loop_belief.json`；
- `configs/closed_loop_execution.json`；
- `config_v3.json` 中的固定评测协议。

断连候选必须位于正式 `link_origin` 区间；`LINK_PENDING` 和没有事件来源的逐状态 linked 脉冲不能构造断连样本。正式 linked 关系即使当前不是动作专家也继续接受关系监控，但执行权重始终为零。伪连接只注入当前已选的 external 动作流。

复现时必须写入不存在的新目录：

```bash
python -m evaluations.phase23_component_ab.run \
  --config evaluations/phase23_component_ab/config_v3.json \
  --output evaluations/phase23_component_ab/results/v7_reproduction
```

调试时可使用 `--tasks stack_wine --demonstrations 0` 写入单独目录。正式输出包括实际评测配置、阶段二/三运行配置、源码/模型/示范哈希、逐试验与逐周期记录、聚合摘要、SVG 图表、中文报告和 `SHA256SUMS`。

最新数值和声明边界见 [RESULTS.md](RESULTS.md)。
