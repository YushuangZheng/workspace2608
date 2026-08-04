# 单臂代表性轨迹可视化审计

## 审计边界

本轮只读取冻结结果 `outputs/single_arm_scientific/v1/trials`，没有重跑、覆盖或修改该目录。新产物写入 `outputs/recovery_scientific/trace_audit_v1`，其 `manifest.json` 记录每个源 JSON/NPZ、视频和接触表的 SHA-256 及实验指纹。

冻结 v1 只保存状态和动作轨迹，没有保存模拟器相机帧。因此视频是依据 NPZ 数值生成的俯视 XY **轨迹重建视图**，不是相机录像。它能检查动作目标、实际末端、物体、目标、阶段、关系状态、夹爪和扰动的时间一致性，但不能证明指尖接触几何或视觉外观。视频标题和 manifest 均明确记录这一限制。

## 固定样本

| 审计角色 | 方法、条件和 seed | 冻结结论 |
| --- | --- | --- |
| 普通成功 | SkillDynaMAC / arm_offset / 6300 | success，XY 误差 0.00709 m |
| 普通失败 | SkillDynaMAC / smooth_object / 6301 | placement_xy_above_threshold，0.01270 m |
| 普通成功 | Legacy Full / arm_offset / 6300 | success，0.00209 m |
| 普通失败 | Legacy Full / arm_offset / 6303 | placement_xy_above_threshold，0.01195 m |
| 普通成功 | RelationDynaMAC / arm_offset / 6300 | success，0.00216 m |
| 普通失败 | RelationDynaMAC / arm_offset / 6303 | placement_xy_above_threshold，0.01071 m |
| 掉落反例 | Legacy Full / drop_after_grasp / 6300 | placement_xy_above_threshold，0.10389 m |
| 掉落反例 | RelationDynaMAC / drop_after_grasp / 6300 | placement_xy_above_threshold，0.10238 m |
| 空抓反例 | Legacy Full / close_without_grasp / 6300 | placement_xy_above_threshold，0.64017 m |
| 空抓反例 | RelationDynaMAC / close_without_grasp / 6300 | environment_terminated，0.43132 m |

“普通”在此指原六类泛化条件，不包含专门加入的因果反例；它并不表示无扰动静态环境。

## Overlay 与可追溯性

每段 MP4 同时显示：任务 phase、关系状态与置信度、connected、物体—目标 XY 误差、夹爪开度、扰动事件、策略动作目标、原始动作目标和实际末端。恢复状态固定显示 `NOT_IMPLEMENTED (baseline)`，用于防止把检测能力误写成恢复能力。

冻结 NPZ 未逐步保存 `active_frames`。若 JSON 中存在 `frame_switch_diagnostics`，工具从第一条 `before` 和后续 `after` 精确还原；若整个 trial 没有切换事件，则没有可用锚点，视频显示 `UNAVAILABLE in frozen v1`，不依据方法名称猜测。以后新实验应逐步持久化 active frames 和 recovery state。

## 审计结论

自动审计从 NPZ 最后一帧重新计算 XY 误差，并核对 JSON 的成功语义、环境终止、支撑、释放、稳定性和阈值。十个样本的 failure taxonomy 语义全部通过，但同时发现一个旧 trace 完整性缺口：冻结 v1 保存的是动作前观测，而 JSON 指标使用动作后的终端观测。普通结束时差异很小；Relation 空抓因物体在终止步骤继续下落，冻结 NPZ 最后一帧与 JSON 终端 XY 误差不一致。`environment_terminated` 分类仍与 JSON 的终止标志一致，但旧视频不能显示终止后的精确落点。

评测代码现已修复：新运行会在 action-aligned 序列之外单独保存 `terminal_ee_position`、`terminal_object_position` 和 `terminal_target_position`，并在 JSON 中保存最终物体和目标位置。这样既不伪造一个没有对应动作的额外 step，也能复现终止瞬间。冻结 v1 按约束保持不变，manifest 为每个旧 trial 分别报告 `terminal_snapshot_persisted` 和 `terminal_trace_alignment`。

其余审计结论如下：

- 六个普通样本的成败边界都由 1 cm XY 阈值区分，失败样本仍满足完成、释放、支撑和稳定条件；
- 两个掉落样本最终物体均停在目标约 10 cm 外，RelationDynaMAC 虽在事件后解除关系，却没有改变任务执行图，因此仍失败；
- Legacy 空抓继续执行到任务结束，物体距目标约 64 cm；Relation 空抓不建立连接并提前触发环境终止，说明分类差异来自实际 rollout 终止状态，不是统计代码误标；
- 当前证据支持“关系检测更可信”，不支持“已经具备恢复能力”。

运行方式：

```bash
/home/zys/miniconda3/envs/env_isaaclab/bin/python scripts/render_trace_audit.py
```

检查 `outputs/recovery_scientific/trace_audit_v1/manifest.json` 的 `all_failure_taxonomies_consistent` 必须为 `true`；终端对齐字段用于显式保留冻结 v1 的已知证据边界。
