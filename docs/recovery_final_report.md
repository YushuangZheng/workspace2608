# 单臂关系触发式恢复正式报告

## 结论先行

预注册的 560 次 held-out 试验已经完整结束。最清楚的结果是：关系检测只有进入独立恢复图后，才转化为任务能力。六个扰动条件合计，`relation_dynamac` 成功 23/120（19.2%），加入同一在线关系估计驱动的恢复图后为 105/120（87.5%），绝对提高 68.3 个百分点。三个 drop 条件中，两个无恢复方法合计只有 1/120 成功，两种恢复方法均为 120/120。

结果也暴露了两个不能回避的问题：`miss_small_shift` 仍是主要弱点；`relation_dynamac_recovery` 在正常条件有 5/20 次短暂误触发，而 Oracle 版本为 0/20。它们已作为正式反例保留，没有根据 test seeds 修改阈值或状态机。

## 不可变性与完整性

- 预注册 tag：`recovery-protocol-v1`；指向提交 `bf53974f5010c17caae09a7ee9903774edc3c3ef`；
- 结果目录：`outputs/recovery_scientific/v1`，未覆盖冻结的 `outputs/single_arm_scientific/v1`；
- 试验规模：4 个方法 × 7 个条件 × 20 个 seeds＝560；JSON 与 NPZ 各 560 个；
- 唯一 `(method, condition, seed)` 组合 560 个，唯一实验指纹 560 个；
- 每方法 140 次、每条件 80 次、每 seed 28 次；
- 所有试验源码提交均为 `bf53974f5010c17caae09a7ee9903774edc3c3ef`；
- 源码内容哈希：`2f041ac992cb8198b9360a67429c9a4d1f15d65bf6c3e33f865ba1634eb1eba7`；
- 冻结数据哈希：`8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8`；
- 最终 `summary.json` 哈希：`2f58eef7a7493c2871854fa0dc9a7c060f43fb3ee7e14f7e49ab8d0acd69914f`；
- schema 7、1000 steps、10 mm 主成功阈值以及具体 drop 距离、方向、夹爪行为均与预注册配置一致；
- 每个 NPZ 均含逐 step 关系/恢复状态、active frames、动作前后目标和独立终端快照；逐 step 长度及 JSON—NPZ 终端位置一致。

这些检查已固化为：

```bash
conda run -n env_isaaclab python scripts/audit_recovery_results.py
```

审计失败时脚本返回非零状态，不允许静默删除不完整试验。

## 主结果

表中均为严格语义成功数，分母为 20。成功要求策略完成、环境未异常终止、夹爪释放、物体在支撑面稳定且最终 XY 误差小于 10 mm。

| 条件 | Legacy Full | Relation | Relation＋Recovery | Oracle＋Recovery |
| --- | ---: | ---: | ---: | ---: |
| drop_lift_early | 1/20 | 1/20 | **20/20** | **20/20** |
| drop_transport_middle | 0/20 | 0/20 | **20/20** | **20/20** |
| drop_before_lower | 0/20 | 0/20 | **20/20** | **20/20** |
| miss_small_shift | 6/20 | 7/20 | **10/20** | 7/20 |
| miss_large_shift | 0/20 | 0/20 | **18/20** | **18/20** |
| edge_grasp | 10/20 | 15/20 | **17/20** | 16/20 |
| normal_no_failure | 18/20 | 18/20 | 18/20 | **19/20** |
| 六个扰动条件合计 | 17/120 | 23/120 | **105/120** | 101/120 |
| 全部条件合计 | 35/140 | 41/140 | **123/140** | 120/140 |

按 seed 平衡的全部条件平均成功率为：

| 方法 | 成功率 | 20-seed bootstrap 95% 区间 |
| --- | ---: | ---: |
| Legacy Full | 25.0% | [19.3%, 30.7%] |
| RelationDynaMAC | 29.3% | [24.3%, 34.3%] |
| RelationDynaMAC-Recovery | **87.9%** | [83.6%, 92.1%] |
| OracleRelation-Recovery | 85.7% | [79.3%, 91.4%] |

该表的主因果比较是同一在线检测器的 `RelationDynaMAC` 与 `RelationDynaMAC-Recovery`，不是把四种方法误写成独立训练样本。恢复图贡献为扰动条件成功数从 23/120 增至 105/120。

## Drop 的检测与恢复

| 条件 | Relation 检测延迟 | Relation 恢复时间 | Oracle 检测延迟 | Oracle 恢复时间 |
| --- | ---: | ---: | ---: | ---: |
| lift 早期 | 40 ms | 1.900 s | 0 ms | 1.734 s |
| transport 中期 | 40 ms | 2.003 s | 0 ms | 1.644 s |
| lower 前 | 40 ms | 2.048 s | 0 ms | 1.660 s |

Legacy Full 的对应关系解除延迟分别为 1.357 s、0.817 s 和 0.619 s；在线关系估计在三个时点均稳定为 40 ms。两种恢复方法在所有距离、方向和夹爪行为子组均成功：

- 距离 5/10/15/20 cm：每种方法各 15/15；
- 方向 back/front/left/right：每种方法各 24/24、12/12、12/12、12/12；
- 夹爪保持原命令/强制张开 3 steps：每种方法各 30/30。

在线检测多出的 40 ms 没有降低 drop 最终成功率，但平均恢复时间比 Oracle 多 0.166–0.388 s。它反映检测确认和恢复状态推进的联合开销。

## Miss、边缘抓持与误触发

| 方法与条件 | 恢复触发率 | 最终成功率 | 平均恢复时间 | 平均额外路径 |
| --- | ---: | ---: | ---: | ---: |
| Relation＋Recovery，miss_small | 70% | 50% | 1.665 s | 0.094 m |
| Oracle＋Recovery，miss_small | 0% | 35% | — | 0.005 m |
| Relation＋Recovery，miss_large | 100% | 90% | 1.399 s | 0.351 m |
| Oracle＋Recovery，miss_large | 100% | 90% | 1.261 s | 0.184 m |
| Relation＋Recovery，edge_grasp | 40% | 85% | 1.547 s | 0.075 m |
| Oracle＋Recovery，edge_grasp | 0% | 80% | — | 约 0 m |
| Relation＋Recovery，正常 | 25%（误触发） | 90% | — | 0 m |
| Oracle＋Recovery，正常 | 0% | 95% | — | 0 m |

正常条件的五次在线误触发都是持续一个 step 的 `LOSS_DETECTED → NORMAL`，没有重抓、没有记录到完整恢复时间，也没有造成额外路径；协议仍将其严格计为误触发，不能因“没有明显动作后果”而排除。

`miss_small_shift` 中 Oracle 没有触发恢复，因为当前 Oracle 是“夹爪占用开度＋末端到物体距离”的瞬时 privileged 几何谓词，不是指尖接触力真值。它会把部分弱抓或擦碰判断为已连接。在线估计器的短暂 loss 反而救回了一部分 trial，但同时造成正常条件误触发。因此，87.9% 高于 85.7% 不能解释为在线估计优于真正物理 Oracle；它首先揭示了当前几何 Oracle 的适用边界，也明确指出下一阶段需要接触真值。

## 安全性与失败分类

四种方法全部试验的限幅后策略动作跳变最大值均为 20 mm/step（浮点误差量级除外）。最大实测末端速度分别为 0.860、0.839、0.867 和 0.920 m/s；恢复层没有绕过原动作限幅。

| 方法 | success | placement_xy_above_threshold | recovery_failed | 其他失败 |
| --- | ---: | ---: | ---: | ---: |
| Legacy Full | 35 | 105 | 0 | 0 |
| RelationDynaMAC | 41 | 99 | 0 | 0 |
| Relation＋Recovery | 123 | 15 | 2 | 0 |
| Oracle＋Recovery | 120 | 18 | 2 | 0 |

四次 `recovery_failed` 全部来自 `miss_large_shift`，两种恢复方法各 2/20；其余失败都是最终 XY 超过 10 mm，而不是环境异常、未释放、支撑失败或稳定性失败。结果说明当前主要瓶颈已经从“掉落后继续盲走”转移到小位移弱抓判定、偶发正常 loss 和大位移重抓可达性。

## 科研声明边界

本轮结果支持：

1. 在线双向关系估计能把 drop 后关系解除延迟从 0.619–1.357 s 降到 40 ms；
2. 恢复图而非检测本身，带来 held-out 扰动成功率的主要提升；
3. 同一恢复图在三个 drop 时点、四种距离、四个方向和两种夹爪行为上均完成恢复；
4. 当前 Oracle 与在线版本的差异可以定位检测确认、Oracle 定义和恢复控制各自的限制。

本轮结果不支持：

- RelationDynaMAC 在无恢复时具有显著普通任务成功率优势；
- 在线估计器优于接触力物理真值；当前 Oracle 只是已知单臂方块几何下的 privileged 谓词；
- `miss_small_shift` 已解决，或正常条件误触发已经可忽略；
- 当前几何双臂交接已经升级为真实物理交接；
- 已复现完整 TAPAS、MiDiGaP 或论文中的黎曼策略。

正式 v1 不再修改。若优化正常误触发或弱抓检测，必须建立新协议、新结果目录和新的 test seeds，不能覆盖本报告。下一主线转入独立的物理 handover 任务与真实双臂关系生命周期；在物理脚本专家通过前不训练完整双臂策略。

## 复现入口

预注册文档和机器配置分别为 `docs/recovery_protocol.md` 与 `configs/experiments/recovery_protocol_v1.json`。正式执行命令见预注册文档；已有结果可用默认 `--resume` 做全量指纹验证。最低复核为：

```bash
conda run -n env_isaaclab python scripts/audit_recovery_results.py
conda run -n env_isaaclab python -m pytest -q
git diff --check
```
