# 阶段六正式评测结果

本文件由冻结协议结果自动生成。所有成功率均以预注册 episode 为分母。

## 单元结果

| 实验 | 故障 | 任务 | 方法 | 成功 | 成功率 | 95% CI | 触发 |
|---|---|---|---|---:|---:|---:|---:|
| normal | - | bimanual_handover_item | dynamac_v4 | 175/200 | 0.875 | [0.822, 0.914] | - |
| normal | - | bimanual_handover_item | progress_only | 149/200 | 0.745 | [0.680, 0.800] | - |
| normal | - | bimanual_handover_item | progress_dynamic_roles | 139/200 | 0.695 | [0.628, 0.755] | - |
| normal | - | bimanual_handover_item | full | 187/200 | 0.935 | [0.892, 0.962] | - |
| normal | - | bimanual_lift_tray | dynamac_v4 | 176/200 | 0.880 | [0.828, 0.918] | - |
| normal | - | bimanual_lift_tray | progress_only | 186/200 | 0.930 | [0.886, 0.958] | - |
| normal | - | bimanual_lift_tray | progress_dynamic_roles | 188/200 | 0.940 | [0.898, 0.965] | - |
| normal | - | bimanual_lift_tray | full | 194/200 | 0.970 | [0.936, 0.986] | - |
| normal | - | bimanual_sweep_to_dustpan | dynamac_v4 | 194/200 | 0.970 | [0.936, 0.986] | - |
| normal | - | bimanual_sweep_to_dustpan | progress_only | 199/200 | 0.995 | [0.972, 0.999] | - |
| normal | - | bimanual_sweep_to_dustpan | progress_dynamic_roles | 195/200 | 0.975 | [0.943, 0.989] | - |
| normal | - | bimanual_sweep_to_dustpan | full | 196/200 | 0.980 | [0.950, 0.992] | - |
| normal | - | bimanual_put_bottle_in_fridge | dynamac_v4 | 164/200 | 0.820 | [0.761, 0.867] | - |
| normal | - | bimanual_put_bottle_in_fridge | progress_only | 154/200 | 0.770 | [0.707, 0.823] | - |
| normal | - | bimanual_put_bottle_in_fridge | progress_dynamic_roles | 167/200 | 0.835 | [0.777, 0.880] | - |
| normal | - | bimanual_put_bottle_in_fridge | full | 166/200 | 0.830 | [0.772, 0.876] | - |
| normal | - | place_cups | dynamac_v4 | 193/200 | 0.965 | [0.930, 0.983] | - |
| normal | - | place_cups | progress_only | 125/200 | 0.625 | [0.556, 0.689] | - |
| normal | - | place_cups | progress_dynamic_roles | 123/200 | 0.615 | [0.546, 0.680] | - |
| normal | - | place_cups | full | 196/200 | 0.980 | [0.950, 0.992] | - |
| normal | - | open_microwave | dynamac_v4 | 197/200 | 0.985 | [0.957, 0.995] | - |
| normal | - | open_microwave | progress_only | 197/200 | 0.985 | [0.957, 0.995] | - |
| normal | - | open_microwave | progress_dynamic_roles | 181/200 | 0.905 | [0.856, 0.938] | - |
| normal | - | open_microwave | full | 195/200 | 0.975 | [0.943, 0.989] | - |
| normal | - | wipe_desk | dynamac_v4 | 137/200 | 0.685 | [0.618, 0.745] | - |
| normal | - | wipe_desk | progress_only | 140/200 | 0.700 | [0.633, 0.759] | - |
| normal | - | wipe_desk | progress_dynamic_roles | 144/200 | 0.720 | [0.654, 0.778] | - |
| normal | - | wipe_desk | full | 145/200 | 0.725 | [0.659, 0.782] | - |
| normal | - | stack_wine | dynamac_v4 | 200/200 | 1.000 | [0.981, 1.000] | - |
| normal | - | stack_wine | progress_only | 200/200 | 1.000 | [0.981, 1.000] | - |
| normal | - | stack_wine | progress_dynamic_roles | 200/200 | 1.000 | [0.981, 1.000] | - |
| normal | - | stack_wine | full | 200/200 | 1.000 | [0.981, 1.000] | - |

## 跨任务宏平均差值

| 实验 | 故障 | 方法 - 参照 | 差值 | 成对 bootstrap 95% CI |
|---|---|---|---:|---:|
| normal | - | progress_only - dynamac_v4 | -0.054 | [-0.147, 0.015] |
| normal | - | progress_dynamic_roles - progress_only | -0.008 | [-0.040, 0.024] |
| normal | - | full - progress_dynamic_roles | 0.089 | [0.010, 0.188] |
| normal | - | full - dynamac_v4 | 0.027 | [0.004, 0.054] |
