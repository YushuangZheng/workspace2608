# RelationDynaMAC 单臂恢复图

## 目标与边界

`RelationDynaMACRecoveryPolicy` 把在线关系生命周期第一次转化为执行期动作。它由原 RelationDynaMAC 生成正常任务动作和关系估计，再由独立的 `RelationRecoveryController` 决定是否覆盖动作并暂停任务 phase clock。恢复层不修改 Gaussian 模型，不读取未来状态，也不读取仿真真值关系。

当前实现只回答单臂空抓和抓取后掉落；它不是 Oracle，也不证明双臂物理交接。

## 两层状态

任务层保持原十阶段：`rest → approach_above_object → approach_object → grasp_object → lift_object → move_above_target → lower_to_target → release_object → retreat → complete`。

恢复监督层为：

```text
NORMAL
├─ phase 4 闭合后 12 step 仍未 CONNECTED → MISS_DETECTED
└─ phase 4/5/6 中 CONNECTED → CANDIDATE_LOST → LOSS_DETECTED

MISS_DETECTED / LOSS_DETECTED
→ SAFE_RETREAT
→ RELOCALIZE
→ REAPPROACH
→ REGRASP
→ VERIFY_GRASP
├─ CONNECTED → RESUME_TASK → NORMAL（从 phase 4 恢复）
└─ 验证超时 → 重试；达到 2 次 → RECOVERY_FAILED
```

`CANDIDATE_LOST` 首先冻结搬运动作；若关系恢复为 `CONNECTED`，直接取消恢复，不产生撤离动作。正常 release phase 7 的关系解除不触发 loss recovery。

## 动作与安全约束

- 恢复激活时暂停 `phase_step`，但继续累计真实控制 step；
- 所有覆盖动作仍经过原有 20 mm/step 笛卡尔限幅；
- `SAFE_RETREAT` 张开夹爪并垂直撤到物体上方 120 mm；
- `RELOCALIZE` 使用当前物体位姿，而不是扰动前位姿；
- `REAPPROACH` 保持张开，到物体上方 65 mm；
- `REGRASP` 先保持张开下降，进入 6 mm 容差后才闭合，防止在物体上方空闭合；
- `VERIFY_GRASP` 生成 80 mm 小幅验证抬升，让关系估计获得共同运动证据；
- 单状态、总恢复时长和重抓次数均有上界，失败进入显式 `RECOVERY_FAILED`。

## 标定来源

重抓末端—物体偏移只使用五条冻结训练示范 phase 3 的后半段标定，不使用 evaluation seed。当前中位数为：

```text
[-0.00240064, -0.00048160, 0.00968410] m
```

标定来源、样本数和数值均写入实验 fingerprint。其余几何量和时间窗在大规模恢复协议之前固定；Phase 3 的 held-out test 结果禁止用于修改这些阈值。

## Trace 与指标

新 schema 逐 step 保存 `active_frames`、`recovery_state`、`recovery_trigger` 和 `regrasp_attempts`，另存动作后的 terminal snapshot。汇总指标新增：

- 是否触发、触发类型和状态切换 step；
- `time_to_recover_s`；
- 重抓次数与 `recovery_failed`；
- 正常条件下 `false_recovery_trigger`；
- 原有最终成功、路径、末端速度和动作跳变。

## Phase 1 development 集成

在未用于旧评测的 development seeds 6400–6402 上运行 9 个 trial：

| 条件 | 成功 | 恢复触发 | 平均恢复时间 | 平均重抓次数 | 最大限幅后动作跳变 |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 3/3 | 0/3 | 不适用 | 0 | 0.020 m |
| drop_after_grasp | 3/3 | 3/3 | 1.920 s | 1 | 0.020 m |
| close_without_grasp | 3/3 | 3/3 | 1.373 s | 1 | 0.020 m |

三条 static 均没有误触发恢复。drop 条件在 40 ms 关系解除后重抓、验证并从 phase 4 恢复；miss 条件重新定位被移动的物体后完成任务。结果位于 `outputs/recovery_scientific/phase1_dev_v1`。

第一次烟测使用未经示范标定的 25 mm 抓取高度，出现“状态机走通但夹爪在物体上方闭合”的反例。修复为训练示范标定偏移，并把下降与闭合拆开后，同一 seed 的两个反例均成功。该结果只证明 Phase 1 集成可行，不替代预注册的多 seed 检验。
