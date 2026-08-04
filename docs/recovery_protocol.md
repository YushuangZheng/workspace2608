# 单臂关系恢复实验预注册协议

## 冻结时点与研究问题

本协议在查看 seeds 6500–6519 的任何 rollout 前创建。协议对应机器可读配置 `configs/experiments/recovery_protocol_v1.json`；实现和协议提交后建立本地 tag `recovery-protocol-v1`。held-out 结果无论成功或失败都不得用于修改关系阈值、恢复状态机、扰动几何、最大重抓次数或成功定义。

研究问题分三层：

1. 关系检测能否及时识别空抓和不同阶段的掉落？
2. 相同恢复图在在线关系与 Oracle relation 下能否完成重抓并继续任务？
3. 恢复是否以正常条件误触发、额外路径或不安全动作作为代价？

## 数据和 seed 分区

| 分区 | seed | 用途 |
| --- | --- | --- |
| calibration | 2608–2612 | 冻结五示范；标定关系估计与重抓位姿 |
| historical/debug | 6200–6202、6300–6309 | 旧单臂开发和冻结评测，不进入本协议统计 |
| development | 6400–6402 | Phase 1/2 集成与反例修复，不进入 held-out 统计 |
| held-out test | 6500–6519 | 20 个新 seed；协议冻结后只运行一次主分析 |

同一 seed 下的 method/condition 是配对试验；不同条件仍共享同一训练数据，不能写成独立训练集。

## 固定方法

- `full_dynamac`：Legacy Full，无恢复图；
- `relation_dynamac`：双向在线关系估计，无恢复图；
- `relation_dynamac_recovery`：在线关系估计＋相同恢复图；
- `oracle_relation_recovery`：当前 privileged 单臂抓持谓词＋相同恢复图。

Oracle 不读取未来、phase 或扰动控制器，也不提供目标动作。其几何定义和限制见 `docs/oracle_recovery_ablation.md`。

## 固定条件

Drop 条件把物体放回支撑面，时点分别为 phase 4 step 8、phase 5 step 12、phase 5 step 22。距离由 `seed % 4` 固定映射到 5/10/15/20 cm，方向由 `(seed // 4) % 4` 映射到前/后/左/右；奇数 seed 同时强制夹爪意外张开 3 个控制 step，偶数 seed 保持原命令。20 seeds 必须覆盖全部距离、方向和两种夹爪行为。

Miss 条件都在 phase 3 step 0 移动物体：

- `miss_small_shift`：3.0 cm；
- `miss_large_shift`：10.0 cm；
- `edge_grasp`：1.8 cm，用于产生边缘或单侧抓持反例；
- 方向使用同一 seed 映射。

`normal_no_failure` 不施加事件，用于测量最终成功和恢复误触发。全部对象移动只发生一次，恢复期间不重复注入扰动。

## 固定控制与成功标准

- 控制周期 20 ms，最多 1000 steps；
- 最大笛卡尔动作变化 20 mm/step；
- 最大重抓 2 次；单状态最多 120 steps，总恢复最多 450 steps；
- MISS 验证窗 12 steps，loss 候选确认最多 4 steps；
- 最终成功要求：策略完成、环境未异常终止、夹爪释放、物体在支撑面稳定，且最终物体—目标 XY 误差小于 10 mm；
- 5/10/20 mm XY 阈值只作敏感性分析，主结论固定使用 10 mm。

## 固定指标与失败分类

逐 trial 保存并汇总：

- 最终成功率及 Wilson 95% 区间；
- 事件到关系断开的 detection latency；
- recovery success、首次触发到 `RESUME_TASK` 的 time-to-recover；
- 相对同 method/seed 的 `normal_no_failure` 额外路径；
- 重抓次数、再次关系丢失和 `RECOVERY_FAILED`；
- 正常条件恢复误触发率；
- 最大末端速度、原始动作跳变和限幅后动作跳变；
- 最终 XY 误差、支撑、释放、稳定性；
- 互斥 failure taxonomy：`success`、`environment_terminated`、`recovery_failed`、`policy_incomplete`、`not_released`、`not_on_support`、`unstable_after_release`、`placement_xy_above_threshold`。

主比较为同 seed 配对：Relation 无恢复 vs Relation-Recovery 回答恢复图贡献；Relation-Recovery vs Oracle-Recovery 区分检测与控制瓶颈。报告每个条件的原始计数，不用小样本 p 值代替效应量和区间。

## 执行与不可变性

正式命令固定为：

```bash
conda run -n env_isaaclab python scripts/eval_single_arm.py --headless \
  --methods full_dynamac relation_dynamac \
  relation_dynamac_recovery oracle_relation_recovery \
  --conditions drop_lift_early drop_transport_middle drop_before_lower \
  miss_small_shift miss_large_shift edge_grasp normal_no_failure \
  --seeds 6500 6501 6502 6503 6504 6505 6506 6507 6508 6509 \
  6510 6511 6512 6513 6514 6515 6516 6517 6518 6519 \
  --max_steps 1000 --no-resume \
  --output_dir outputs/recovery_scientific/v1
```

预计 4×7×20＝560 个隔离 worker 和 560 对 JSON/NPZ。运行前必须是干净工作树；每个 trial 必须有 schema 7、源提交、源 SHA-256、冻结数据 SHA-256、方法配置和含 seed 具体位移的扰动配置。缺失或指纹重复时整组报告为基础设施不完整，不静默删 trial。

held-out 后允许修复的只有报告/聚合代码错误，且不得改变 rollout；任何策略或条件修改必须升为 `v2` 协议、新目录和新 test seeds。
