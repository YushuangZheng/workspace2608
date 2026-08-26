# 阶段二—三组件级 A/B 评测

该评测从冻结 DynaMAC V4 的五条正常训练示范派生确定性的受控扰动输入，对比：

- DynaMAC 固定一步一状态时钟，与阶段二在线进度估计；
- DynaMAC 离线固定流掩码，与阶段三动态关系角色及推进阻断。

评测不会训练模型、启动 RLBench/CoppeliaSim、修改原始示范或覆盖 V4 正式结果。它是组件能力和消融实验草稿，不是独立测试集上的最终任务成功率实验。

## 协议版本

- `results/v1`：首次固定的8状态关系窗口；当时无法在“当前已选流”中构造正式断链条件。
- `results/v2`：把关系窗口缩短为2状态，但断链候选仍来自逐状态 linked 证据，尚未限定为正式 LINK 事件，因此其断链负结果只作为历史诊断保留。
- `results/v3`：当前正式结果。断链候选必须位于正式 `link_origin` 区间，排除 `LINK_PENDING` 和孤立 linked 脉冲；正式关系即使对应专家当前未选中也继续监控，但其动作权重始终为0。事件一致传播使原始8状态扰动和4状态观察尾窗重新可用。

V3 没有根据故障检测结果调整在线阈值。事件先验使用阶段一既有 `external≤0.3 / linked≥0.7` 口径，在线关系滤波、角色阈值和扰动长度均由机制或原始协议确定。

复现完整 V3 时应写入一个不存在的新目录，例如：

```bash
python -m evaluations.phase23_component_ab.run \
  --config evaluations/phase23_component_ab/config_v3.json \
  --output evaluations/phase23_component_ab/results/v3_reproduction
```

调试时可用 `--tasks stack_wine --demonstrations 0` 写入单独临时目录。输出包含：

```text
config.json                 实际运行配置副本
environment.json            Git、Python、依赖和源码/模型/示范哈希
progress_trials.csv         逐时间扰动试验指标
relation_trials.csv         逐关系扰动试验指标
task_summary.csv            按任务/机械臂聚合的异质性结果
tick_trace.csv.gz           逐控制周期审计轨迹
summary.json / summary.csv  聚合结果
report.md                   中文解释和声明边界
figures/*.svg               论文草稿图
SHA256SUMS                  输出文件校验和
```

输出目录必须不存在，以避免覆盖已有评测记录。当前结果、逐任务异质性和结论边界见 [RESULTS.md](RESULTS.md)。
