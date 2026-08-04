# Oracle relation 恢复消融

## 消融问题

Phase 2 比较：

| 方法 | 关系输入 | 恢复图 |
| --- | --- | --- |
| Legacy Full | 旧运动学检测与永久锁存 | 无 |
| RelationDynaMAC | 双向在线估计 | 无 |
| RelationDynaMAC-Recovery | 双向在线估计 | 有 |
| OracleRelation-Recovery | 当前仿真真值谓词 | 有 |

Oracle 的用途不是提高动作质量，而是区分“关系判断错误”和“恢复控制错误”。四种方法使用相同冻结示范、Gaussian 动作模型、任务环境和成功标准。

## Oracle 的精确定义

Isaac Sim 4.5 在当前 Franka articulation 上初始化 filtered ContactSensor 时会直接关闭 SimulationApp，无法提供可审计的 Python 异常。该传感器方案已撤销，没有进入任务环境。

当前单臂 Oracle 使用瞬时 privileged grasp predicate：

```text
0.020 m ≤ actual gripper opening ≤ 0.063 m
且
|| actual EE position - actual object position ||₂ ≤ 0.040 m
```

开度下界排除空夹爪完全闭合，距离条件排除掉落后夹爪仍保持占用开度的情况。输入全部来自当前仿真 step，不读取 phase、未来轨迹或扰动控制器，也不提供目标动作。它是已知 Franka—方块几何下的单臂真值谓词，不应写成通用接触力 Oracle。

Oracle 可在 grasp phase 3 建立关系，而旧指标把 phase 4–6 硬编码为“期望连接”。因此 Oracle 的 `mask_false_positive_rate` 和负 onset delay 不能解释为 Oracle 错误；本消融使用最终恢复、触发、恢复时间和动作安全指标。以后加入稳定接触传感器后，应以双指接触与相对运动的一致性替换该谓词。

## 单 seed 集成结果

development seed 6400 上：

| 条件 | 最终结果 | 恢复时间 | 关系丢失延迟 |
| --- | --- | ---: | ---: |
| static | 成功，无恢复触发 | 不适用 | 不适用 |
| drop_after_grasp | 成功 | 1.700 s | 0 step |
| close_without_grasp | 成功 | 1.280 s | 不适用 |

同一 seed 的在线 RelationDynaMAC-Recovery 也全部成功，drop 关系丢失延迟为 40 ms，恢复时间分别为 1.920 s 和 1.373 s。这个小样本说明恢复图本身可以工作，Oracle 主要减少检测等待；它还不足以估计成功率差异。

## 三 seed 统一 development 消融

在 seeds 6400–6402 上得到 36 个完整 trial：

| 方法 | static | drop | miss | drop 检测延迟 | drop 恢复时间 | miss 恢复时间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy Full | 3/3 | 0/3 | 0/3 | 0.840 s* | 不适用 | 不适用 |
| RelationDynaMAC | 3/3 | 0/3 | 0/3 | 0.040 s | 不适用 | 不适用 |
| RelationDynaMAC-Recovery | 3/3 | 3/3 | 3/3 | 0.040 s | 1.920 s | 1.373 s |
| OracleRelation-Recovery | 3/3 | 3/3 | 3/3 | 0.000 s | 1.713 s | 1.240 s |

`*` Legacy 的 0.840 s 是后续夹爪打开导致旧锁存清除，不是掉落检测或恢复触发。

两种恢复方法的 static 恢复误触发都是 0/3、平均重抓都是 1 次、恢复失败都是 0/3，限幅后最大动作跳变均为 0.020 m。Oracle 将 drop 检测提前 40 ms，并缩短平均恢复时间约 0.207 s，但没有进一步提高本 development 集的最终成功率。这支持“恢复图有效，检测等待影响恢复速度”；3 seeds 不支持宣称两种恢复方法成功率等价。

完整比较写入新的 `outputs/recovery_scientific/phase2_dev_v1`，36 个 JSON/NPZ 均带完整 fingerprint。不得用其结果调整后续 held-out 阈值。
