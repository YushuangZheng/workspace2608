# 阶段四边界参数标定结果

- 控制周期：0.050 秒（`RLBench/rlbench/task_environment.py::_DT`）
- 正常示范：每任务 5 条
- 已标定边界：40/40
- 无需连续确认消歧即可分离的边界：28/40
- 正常运行配置放行复核：200/200 条边界×示范
- 真实模型联合事务：1 组，正常联合就绪复核 5/5 组×示范
- 故障数据使用：0

| 任务/机械臂 | 边界 | 正常末端下界 | 边界前上界 | theta_local | H | 秒 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| bimanual_handover_item/left | `left:0->1` | 0.00411186 | 0.0406199 | 0.00328949 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:1->2` | 0.00997726 | 0.0297596 | 0.0079818 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:2->3` | 0.0014876 | 0.0716304 | 0.00119008 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:3->4` | 0.0388614 | 2.15142e-10 | 0.0194307 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:4->5` | 0.028778 | 0.0744018 | 0.0230224 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:5->6` | 0.265305 | 0.585067 | 0.212244 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/right | `right:0->1` | 0.00343392 | 0.247594 | 0.00274713 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/right | `right:1->2` | 0.131649 | 0.145883 | 0.105319 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/right | `right:2->3` | 0.998003 | 0.154808 | 0.576405 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/right | `right:3->4` | 0.998481 | 0.180572 | 0.589527 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/right | `right:4->5` | 0.997833 | 0.180426 | 0.58913 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/right | `right:5->6` | 0.294903 | 0.0447008 | 0.169802 | 2 | 0.10 | `calibrated` |
| bimanual_lift_tray/left | `left:0->1` | 0.00656496 | 0.0427082 | 0.00525197 | 2 | 0.10 | `calibrated` |
| bimanual_lift_tray/left | `left:1->2` | 0.00199782 | 0.0217458 | 0.00159825 | 2 | 0.10 | `calibrated` |
| bimanual_lift_tray/right | `right:0->1` | 0.00783794 | 0.0473218 | 0.00627035 | 2 | 0.10 | `calibrated` |
| bimanual_lift_tray/right | `right:1->2` | 0.00146531 | 0.00202034 | 0.00117224 | 2 | 0.10 | `calibrated` |
| bimanual_put_bottle_in_fridge/left | `left:0->1` | 0.0970226 | 4.00258e-05 | 0.0485313 | 2 | 0.10 | `calibrated` |
| bimanual_put_bottle_in_fridge/left | `left:1->2` | 0.0592898 | 7.78198e-08 | 0.0296449 | 2 | 0.10 | `calibrated` |
| bimanual_put_bottle_in_fridge/right | `right:0->1` | 0.991093 | 1.86799e-55 | 0.495547 | 2 | 0.10 | `calibrated` |
| bimanual_put_bottle_in_fridge/right | `right:1->2` | 0.899392 | 2.17048e-25 | 0.449696 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:0->1` | 0.0694609 | 5.74698e-10 | 0.0347304 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:1->2` | 0.994343 | 1.13425e-131 | 0.497171 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:2->3` | 0.997215 | 0.154429 | 0.575822 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:3->4` | 0.499884 | 0.0145099 | 0.257197 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:0->1` | 0.00306071 | 5.56469e-21 | 0.00153036 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:1->2` | 0.00132545 | 8.0289e-13 | 0.000662726 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:2->3` | 0.0282061 | 2.80323e-09 | 0.0141031 | 2 | 0.10 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:3->4` | 0.617371 | 0.159247 | 0.388309 | 2 | 0.10 | `calibrated` |
| open_microwave/single | `single:0->1` | 0.0166931 | 1.35024e-10 | 0.00834653 | 2 | 0.10 | `calibrated` |
| open_microwave/single | `single:1->2` | 0.0129683 | 4.06346e-47 | 0.00648417 | 2 | 0.10 | `calibrated` |
| place_cups/single | `single:0->1` | 0.159132 | 1.67698e-27 | 0.0795661 | 2 | 0.10 | `calibrated` |
| place_cups/single | `single:1->2` | 0.713832 | 7.2826e-13 | 0.356916 | 2 | 0.10 | `calibrated` |
| place_cups/single | `single:2->3` | 0.147994 | 0.033372 | 0.090683 | 2 | 0.10 | `calibrated` |
| place_cups/single | `single:3->4` | 0.361466 | 0.00112867 | 0.181297 | 2 | 0.10 | `calibrated` |
| stack_wine/single | `single:0->1` | 0.11077 | 8.07217e-48 | 0.0553849 | 2 | 0.10 | `calibrated` |
| stack_wine/single | `single:1->2` | 0.803025 | 5.99693e-28 | 0.401512 | 2 | 0.10 | `calibrated` |
| stack_wine/single | `single:2->3` | 0.0244959 | 0.000649513 | 0.0125727 | 2 | 0.10 | `calibrated` |
| wipe_desk/single | `single:0->1` | 0.203305 | 4.39141e-08 | 0.101653 | 2 | 0.10 | `calibrated` |
| wipe_desk/single | `single:1->2` | 0.0635964 | 0.329764 | 0.0508771 | 4 | 0.20 | `calibrated` |
| wipe_desk/single | `single:2->3` | 0.392668 | 1.25639e-27 | 0.196334 | 2 | 0.10 | `calibrated` |

## 联合事务元数据

| 任务 | 事务组 | 成员边界 |
|---|---|---|
| bimanual_lift_tray | `joint:left-k1-to-2:right-k1-to-2:item-tray` | `left:1->2`, `right:1->2` |

## 口径

`theta_local` 与 `H` 只由正常回放确定。若边界前与末端分数重叠，不增加新的特殊门控，而是由既有连续确认机制抑制短暂提前脉冲。末端保持分支使用相同的在线关系与进度更新链，并以真实 0.05 秒控制周期重复当前正常末端观测。
