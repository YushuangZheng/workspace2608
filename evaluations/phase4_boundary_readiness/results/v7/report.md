# 阶段四正常边界就绪性检查结果

## 1. 结论边界

本检查由五条正常训练示范回放构成。每个新技能假设其上一边界已经正常提交，将进度初始化到该技能首状态，并继续携带已有关系后验，以免早期边界滞后污染后续边界的自身就绪性。检查用于确认离线边界模型和阶段二后验能否为阶段四提供可用的标定输入；它不实现在线入口守卫、不注入故障、不使用故障集调参，也不衡量仿真任务成功率。

本次回放使用的阶段二最低解释度为 `0.001`；完整运行配置保存在同目录 `belief_config.json`。

共检查 40 个技能边界；其中 9 个满足当前严格的标定输入检查。`theta_inputs_ready` 只说明正常边界末端的进度、目标和本臂关系输入均有效；`P_end` 的正负间隔另行保留，供联合完成度标定时判断。`h_inputs_ready` 只说明至少存在配置要求数量的连续正常样本；本检查不自行选择最终阈值。

## 2. 优先任务

| 任务/机械臂 | 边界 | 末端有效 | 末端概率最小值 | 最终目标 | 本臂关系 | 边界关系 | 场景 | 连续样本 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| bimanual_put_bottle_in_fridge/left | left:0->1 | 5/5 | 1.0000 | 100% | 100% | 100% | 100% | 1 | `ready_theta_limited_h` |
| bimanual_put_bottle_in_fridge/left | left:1->2 | 5/5 | 1.0000 | 100% | 100% | 100% | 100% | 1 | `ready_theta_limited_h` |
| bimanual_put_bottle_in_fridge/right | right:0->1 | 5/5 | 1.0000 | 100% | 100% | 100% | 100% | 1 | `ready_theta_limited_h` |
| bimanual_put_bottle_in_fridge/right | right:1->2 | 5/5 | 1.0000 | 100% | 100% | 100% | 100% | 2 | `ready_for_calibration` |
| wipe_desk | single:0->1 | 5/5 | 1.0000 | 100% | 100% | 100% | 100% | 1 | `ready_theta_limited_h` |
| wipe_desk | single:1->2 | 5/5 | 1.0000 | 100% | 100% | 100% | 100% | 1 | `ready_theta_limited_h` |
| wipe_desk | single:2->3 | 5/5 | 1.0000 | 100% | 100% | 100% | 100% | 1 | `ready_theta_limited_h` |

## 3. 全部边界

| 任务/机械臂 | 边界 | 终止窗 NPS | P_end 正负间隔 | theta 输入 | H 输入 | 守卫输入 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| bimanual_put_bottle_in_fridge/left | left:0->1 | 0.0% | 0.6750 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_put_bottle_in_fridge/left | left:1->2 | 0.0% | 0.2454 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_put_bottle_in_fridge/right | right:0->1 | 0.0% | 0.2123 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_put_bottle_in_fridge/right | right:1->2 | 0.0% | 1.0000 | 是 | 是 | 是 | `ready_for_calibration` |
| wipe_desk | single:0->1 | 0.0% | 0.5340 | 是 | 否 | 是 | `ready_theta_limited_h` |
| wipe_desk | single:1->2 | 0.0% | 0.4413 | 是 | 否 | 是 | `ready_theta_limited_h` |
| wipe_desk | single:2->3 | 0.0% | 0.9860 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_handover_item/left | left:0->1 | 0.0% | 0.1007 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_handover_item/left | left:1->2 | 0.0% | 0.4745 | 是 | 是 | 是 | `ready_for_calibration` |
| bimanual_handover_item/left | left:2->3 | 0.0% | 0.6753 | 是 | 是 | 是 | `ready_for_calibration` |
| bimanual_handover_item/left | left:3->4 | 0.0% | 0.3380 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_handover_item/left | left:4->5 | 0.0% | -0.5876 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_handover_item/left | left:5->6 | 0.0% | 0.7004 | 是 | 是 | 是 | `ready_for_calibration` |
| bimanual_handover_item/right | right:0->1 | 0.0% | -0.1715 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_handover_item/right | right:1->2 | 0.0% | 0.7000 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_handover_item/right | right:2->3 | 0.0% | 0.7011 | 是 | 是 | 是 | `ready_for_calibration` |
| bimanual_handover_item/right | right:3->4 | 0.0% | 0.7010 | 是 | 是 | 是 | `ready_for_calibration` |
| bimanual_handover_item/right | right:4->5 | 0.0% | 0.7011 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_handover_item/right | right:5->6 | 0.0% | 0.1536 | 是 | 是 | 是 | `ready_for_calibration` |
| bimanual_lift_tray/left | left:0->1 | 0.0% | 0.3450 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_lift_tray/left | left:1->2 | 0.0% | 0.5497 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_lift_tray/right | right:0->1 | 0.0% | -0.1556 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_lift_tray/right | right:1->2 | 0.0% | 0.5096 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_sweep_to_dustpan/left | left:0->1 | 0.0% | -0.0849 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_sweep_to_dustpan/left | left:1->2 | 0.0% | 0.9606 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_sweep_to_dustpan/left | left:2->3 | 0.0% | 0.7009 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_sweep_to_dustpan/left | left:3->4 | 0.0% | 0.4981 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_sweep_to_dustpan/right | right:0->1 | 0.0% | -0.0000 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_sweep_to_dustpan/right | right:1->2 | 6.7% | 0.9847 | 是 | 否 | 是 | `ready_theta_limited_h` |
| bimanual_sweep_to_dustpan/right | right:2->3 | 0.0% | 0.9617 | 是 | 是 | 是 | `ready_for_calibration` |
| bimanual_sweep_to_dustpan/right | right:3->4 | 0.0% | 0.6501 | 是 | 是 | 是 | `ready_for_calibration` |
| open_microwave | single:0->1 | 0.0% | 0.0811 | 是 | 否 | 是 | `ready_theta_limited_h` |
| open_microwave | single:1->2 | 0.0% | 0.4746 | 是 | 否 | 是 | `ready_theta_limited_h` |
| place_cups | single:0->1 | 0.0% | 0.9842 | 是 | 否 | 是 | `ready_theta_limited_h` |
| place_cups | single:1->2 | 0.0% | 0.4031 | 是 | 否 | 是 | `ready_theta_limited_h` |
| place_cups | single:2->3 | 0.0% | -0.2169 | 是 | 否 | 是 | `ready_theta_limited_h` |
| place_cups | single:3->4 | 0.0% | 0.3447 | 是 | 否 | 是 | `ready_theta_limited_h` |
| stack_wine | single:0->1 | 0.0% | 0.4196 | 是 | 否 | 是 | `ready_theta_limited_h` |
| stack_wine | single:1->2 | 0.0% | 0.9994 | 是 | 否 | 是 | `ready_theta_limited_h` |
| stack_wine | single:2->3 | 0.0% | 0.3033 | 是 | 否 | 是 | `ready_theta_limited_h` |

## 4. 字段解释

- `NO_PLAUSIBLE_STATE` 时的 `P_end` 只来自传播后的名义先验，本检查不把它计作有效末端证据。
- `P_end 正负间隔` 是全部正常终止窗口样本的最小末端概率减去全部更早样本的最大末端概率；正值表示仅按该分量存在正常完成/未完成分界。
- 目标支持使用阶段一保存的最终状态分布与最低正常训练对数似然，只用于检查输入是否正常，不等同于已经标定 `theta_local`。
- 关系“可观测”与“稳定判定”分开统计：有真值位姿不代表当前运动激励足以把在线关系从 `Unknown` 判为 linked/external。
- `H 输入` 要求每条示范至少有配置数量的连续本地就绪样本；它不直接给出控制周期数 `H`。
