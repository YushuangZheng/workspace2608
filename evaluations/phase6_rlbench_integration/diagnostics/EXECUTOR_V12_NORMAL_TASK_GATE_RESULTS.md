# 阶段六执行器 v12 正常任务门控

日期：2026-08-29

## 目的

在扩大正常样本矩阵前，用当前正式 sidecar 和统一 RLBench 执行器覆盖全部八类任务各一个封存正常样本，检查：

- 正常任务能否完成且不产生 `InvalidAction`；
- 同目标物理停滞后的求解器升级是否会选择不连续关节分支；
- 正常执行是否误触发破坏性恢复；
- Handover 主动验证和 LiftTray 联合事务是否仍走通真实控制链。

本轮不是八任务完整成功率实验，也不是故障/扰动 A/B。

## 发现与通用修复

执行器 v11 在 StoreBottle seed `2608000000` 的 tick 102→103 选择了一个末端位姿近似相同、但关节构型不连续的采样 IK 分支：关节向量 L2 跳变 4.178553 rad，单关节最大跳变 2.887153 rad。该动作撞击并移动瓶子，最终在161周期以 `no_legal_reentry_state` 结束。旧 v10 同 seed 成功，且 v10/v11 在 tick 102 以前的动作一致，因此该失败可定位到物理停滞升级中新引入的采样分支，而不是任务模型或闭环进度差异。

v12 保留“所有局部 IK 初始不可行时”的普通全局采样后备；只对**同一策略目标已经发生物理停滞后的升级采样**施加与 bounded TRAC-IK 相同的关节连续性约束：单关节绝对变化不超过0.35 rad，关节向量 L2 不超过0.50 rad。不连续候选被拒绝，随后仍可进入连续路径后备。该规则不读取任务名、seed、StateId 或目标物体。

同一 StoreBottle seed 在 v12 中274周期成功、0 InvalidAction；最大关节 L2/单关节变化分别为0.499987/0.347337 rad。

## 八任务正常门控结果

| 任务 | 样本索引 | 结果 | 策略周期 | InvalidAction | 局部 IK 族耗尽 | 物理停滞升级 | 最大关节 L2 | 最大单关节变化 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| OpenMicrowave | 0 | 成功 | 163 | 0 | 1 | 7 | 0.463053 | 0.311706 |
| PlaceCups | 0 | 成功 | 187 | 0 | 2 | 30 | 0.471445 | 0.350000 |
| StackWine | 0 | 成功 | 167 | 0 | 1 | 4 | 0.088280 | 0.074454 |
| WipeDesk | 199 | 成功 | 240 | 0 | 0 | 0 | 0.384629 | 0.287125 |
| BimanualHandoverItem | 0 | 成功 | 782 | 0 | 436 | 1580 | 0.491743 | 0.350000 |
| BimanualLiftTray | 1 | 成功 | 106 | 0 | 2 | 16 | 0.451265 | 0.281386 |
| BimanualPutBottleInFridge | 0 | 成功 | 274 | 0 | 86 | 218 | 0.499987 | 0.347337 |
| BimanualSweepToDustpan | 0 | 成功 | 141 | 0 | 19 | 110 | 0.491582 | 0.349727 |

表中的“局部 IK 族耗尽”表示伪逆、bounded TRAC-IK 和采样在某次局部求解中均未直接给出候选，随后路径层仍可能成功；它不等于环境拒绝动作。真实命令失败以 `InvalidAction` 为准，本轮为0。

## 闭环行为复核

- 八个样本均未触发 `recovery_trigger`，没有以破坏性 LINK/UNLINK 恢复换取正常成功。
- Handover 右臂进入17个 `VERIFY_LINK` 周期，其中16个周期处于 `mode_before=verify_link`；窗口累计16个真实响应，末次共动残差比为0.216902，响应方向为 linked。benchmark 在探测期间已判定成功，因此该样本证明没有发生原先的单周期 external→错误恢复，但不能作为“验证稳定结束并完成原路返回”的证据；返回与重复抑制仍由阶段五专项测试验收。
- SweepDust 左臂进入7个 `VERIFY_LINK` 周期，未进入恢复并正常完成。
- LiftTray 的真实联合事务产生17次等待记录和4次边界提交，最终正常完成；其余任务按独立或单向边界语义推进。
- 八个样本所有已选关节候选都满足 v12 的连续性上限；没有再次出现 v11 StoreBottle 的构型跳变。

## 自动化回归

- v12 执行器定向测试：37/37通过，包含停滞升级采样接受与不连续分支否决的正反例；
- Black、Python 编译和 `git diff --check` 通过；
- 阶段一至六、RLBench 集成与冻结基线全仓回归：601项通过、4项跳过、0项失败，耗时320.28秒。

## 原始结果与 SHA256

```text
7ad1cd44d3e507c6389f5057b9d5d33f4f475580e6cc07c683466e635d7a3a33  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_bimanual_handover_item_i000_executor_v12_continuity.json
7e28433b2f06b68767d5b60faa01d1d66ca66e16a258ac99478c438357d98615  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_bimanual_lift_tray_i001_executor_v12_continuity.json
190293b9bbb4f200912049cb78a3444e81b8f9a721f2cef050db98430492acc2  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_bimanual_put_bottle_in_fridge_i000_executor_v12_continuity.json
05119e11a9f4d9a62f0de76ae25e318a7ec756994db734c92bf46c60de9ea9c3  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_bimanual_sweep_to_dustpan_i000_executor_v12_continuity.json
ed0eda441bd96a019d73e52d9193f4726367d1c3396d5e8649cf5761473bbec1  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_open_microwave_i000_executor_v12_continuity.json
96641441a293da0f48e86fd89fa4a65e484ef2b0e3f671c39214c6ad17ac48de  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_place_cups_i000_executor_v12_continuity.json
b488deb7ce223ac281c8a65b02798998ead724b70e59be8e48827e68d235f2cc  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_stack_wine_i000_executor_v12_continuity.json
1f5a79a129c7cd47616edef61f89dcefb27bba3d3799c562d9fd20cccc29954b  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_wipe_desk_i199_executor_v12_continuity.json
```

逐周期记录位于各 JSON 同名目录。v11 StoreBottle 失败原始摘要为：

```text
a5b32906857253cfd719d331f7eafe10e3bd6bf5d7db27ee9cde7e4102b2bdfa  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_bimanual_put_bottle_in_fridge_i000_final_bundle_v3.json
```

## 结论边界

当前 v12 已通过八类任务各一个正常样本的首轮门控，且修复了 v11 引入的通用关节连续性偏差。该结果支持继续扩大正常样本矩阵，但不能替代多 seed 正常成功率、DynaMAC V4 同集对照或后续故障消融；阶段六仍处于开发验收中。
