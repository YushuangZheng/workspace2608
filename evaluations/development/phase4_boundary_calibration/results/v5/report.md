# 阶段四边界参数标定结果

- 控制周期：0.050 秒（`RLBench/rlbench/task_environment.py::_DT`）
- 正常示范：每任务 5 条
- 已标定边界：40/40
- 无需连续确认消歧即可分离的边界：30/40
- 正常运行配置决策复核：200/200 条边界×示范
- 真实模型联合事务：1 组，正常联合就绪复核 5/5 组×示范
- 故障数据使用：0

| 任务/机械臂 | 边界 | 正常末端下界 | 边界前上界 | theta_local | H | 秒 | 结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| bimanual_handover_item/left | `left:0->1` | 1.20206e-07 | 1.62518e-05 | 9.6165e-08 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:1->2` | 0.0104907 | 0.0297214 | 0.00839258 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:2->3` | 0.00178554 | 0.0756914 | 0.00142843 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:3->4` | 0.0151563 | 1.75895e-16 | 0.00757814 | 1 | 0.05 | `calibrated` |
| bimanual_handover_item/left | `left:4->5` | 0.000121246 | 0.00122969 | 9.69968e-05 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/left | `left:5->6` | 0.831906 | 0.163181 | 0.497543 | 1 | 0.05 | `calibrated` |
| bimanual_handover_item/right | `right:0->1` | 0.00144935 | 0.230132 | 0.00115948 | 2 | 0.10 | `calibrated` |
| bimanual_handover_item/right | `right:1->2` | 0.536393 | 0.147854 | 0.342124 | 1 | 0.05 | `calibrated` |
| bimanual_handover_item/right | `right:2->3` | 0.998003 | 0.154357 | 0.57618 | 1 | 0.05 | `calibrated` |
| bimanual_handover_item/right | `right:3->4` | 0.998481 | 0.154788 | 0.576634 | 1 | 0.05 | `calibrated` |
| bimanual_handover_item/right | `right:4->5` | 0.997833 | 0.154883 | 0.576358 | 1 | 0.05 | `calibrated` |
| bimanual_handover_item/right | `right:5->6` | 0.295105 | 0.0447815 | 0.169943 | 1 | 0.05 | `calibrated` |
| bimanual_lift_tray/left | `left:0->1` | 0.000297201 | 0.234723 | 0.00023776 | 2 | 0.10 | `calibrated` |
| bimanual_lift_tray/left | `left:1->2` | 0.00182793 | 0.0126642 | 0.00146234 | 2 | 0.10 | `calibrated` |
| bimanual_lift_tray/right | `right:0->1` | 0.000445322 | 0.0293371 | 0.000356257 | 2 | 0.10 | `calibrated` |
| bimanual_lift_tray/right | `right:1->2` | 0.00119239 | 0.00260936 | 0.000953913 | 2 | 0.10 | `calibrated` |
| bimanual_put_bottle_in_fridge/left | `left:0->1` | 0.698092 | 1.16716e-06 | 0.349047 | 1 | 0.05 | `calibrated` |
| bimanual_put_bottle_in_fridge/left | `left:1->2` | 0.228956 | 3.63487e-11 | 0.114478 | 1 | 0.05 | `calibrated` |
| bimanual_put_bottle_in_fridge/right | `right:0->1` | 0.982301 | 4.24059e-110 | 0.491151 | 1 | 0.05 | `calibrated` |
| bimanual_put_bottle_in_fridge/right | `right:1->2` | 0.826252 | 6.92024e-38 | 0.413126 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:0->1` | 0.975326 | 2.27334e-11 | 0.487663 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:1->2` | 0.99434 | 1.21325e-132 | 0.49717 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:2->3` | 0.997214 | 0.154429 | 0.575822 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/left | `left:3->4` | 0.499881 | 0.014503 | 0.257192 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:0->1` | 2.23866e-09 | 3.5559e-41 | 1.11933e-09 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:1->2` | 0.00119 | 4.5159e-11 | 0.000594998 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:2->3` | 0.0281167 | 2.88178e-09 | 0.0140583 | 1 | 0.05 | `calibrated` |
| bimanual_sweep_to_dustpan/right | `right:3->4` | 0.617543 | 0.159215 | 0.388379 | 1 | 0.05 | `calibrated` |
| open_microwave/single | `single:0->1` | 0.00996768 | 1.61652e-14 | 0.00498384 | 1 | 0.05 | `calibrated` |
| open_microwave/single | `single:1->2` | 0.0132333 | 3.93261e-47 | 0.00661663 | 1 | 0.05 | `calibrated` |
| place_cups/single | `single:0->1` | 2.90269e-05 | 3.09466e-91 | 1.45135e-05 | 1 | 0.05 | `calibrated` |
| place_cups/single | `single:1->2` | 0.712713 | 8.00917e-13 | 0.356357 | 1 | 0.05 | `calibrated` |
| place_cups/single | `single:2->3` | 0.0370551 | 0.00259619 | 0.0198256 | 1 | 0.05 | `calibrated` |
| place_cups/single | `single:3->4` | 0.333938 | 0.00121189 | 0.167575 | 1 | 0.05 | `calibrated` |
| stack_wine/single | `single:0->1` | 0.187785 | 1.1006e-68 | 0.0938925 | 1 | 0.05 | `calibrated` |
| stack_wine/single | `single:1->2` | 0.805291 | 2.33799e-28 | 0.402646 | 1 | 0.05 | `calibrated` |
| stack_wine/single | `single:2->3` | 0.0288478 | 0.00189952 | 0.0153737 | 1 | 0.05 | `calibrated` |
| wipe_desk/single | `single:0->1` | 0.760199 | 6.10816e-14 | 0.3801 | 1 | 0.05 | `calibrated` |
| wipe_desk/single | `single:1->2` | 0.0442027 | 0.21383 | 0.0353622 | 3 | 0.15 | `calibrated` |
| wipe_desk/single | `single:2->3` | 0.287436 | 3.26866e-68 | 0.143718 | 1 | 0.05 | `calibrated` |

## 联合事务元数据

| 任务 | 事务组 | 成员边界 |
|---|---|---|
| bimanual_lift_tray | `joint:left-k1-to-2:right-k1-to-2:item-tray` | `left:1->2`, `right:1->2` |

## 口径

`theta_local` 与 `H` 只由正常回放中的本地完成度确定，不以关系/场景守卫是否已经放行为正样本筛选条件。若边界前与末端分数重叠，不增加新的特殊门控，而是由既有连续确认机制抑制短暂提前脉冲。末端保持分支使用相同的在线关系与进度更新链，并以真实 0.05 秒控制周期重复当前正常末端观测；单向跨臂条件尚未满足时，`LocalDone=True` 且继续等待也是正确决策。
