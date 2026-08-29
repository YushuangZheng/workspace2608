# 阶段六联合目标 PoE、执行器 v11 与主动验证窗口复核

日期：2026-08-29

## 本轮变更

1. 阶段四 `L_goal` 改为：将当前 mode 的全部最终目标流变换到世界坐标系，以动作生成相同的精度加权 PoE 构造一个联合末端目标，再对当前真实末端位姿做一次峰值归一评分。
2. RLBench 通用执行器升级为 `stage6_hybrid_cartesian_executor_v11` / `rlbench-stage6-hybrid-cartesian-continuation-v13`。合法关节目标发生真实物理停滞时，对同一个策略目标执行一次有界求解器升级；产生进展后立即返回策略重新观测。
3. `VERIFY_LINK` 排除进入模式前的 TASK 运动，只累计验证动作的实际响应；至少3个可靠响应后，以0.85的窗口共动残差比阈值给出 linked/external 响应方向，并要求它与原二元关系滤波器的有效非 Unknown 决策一致。

以上均为统一算法或 benchmark 执行层修改，不读取任务名、seed 或固定 StateId。

## 自动化回归

- 阶段五主动验证专项：17/17 通过；
- 阶段二至六、RLBench 执行器与正常诊断入口的跨阶段定向回归：146/146 通过，耗时132.93秒；
- 当前全仓回归：600项通过、4项跳过、0项失败，耗时316.59秒；
- 覆盖单帧 external 不得提前终止真实共动验证、静止参考系在末端运动时仍判 external、超时返回、原路返回、重复触发抑制、入口联合目标 PoE、边界事务与执行器停滞升级。

## 真实 RLBench 正常子集

| 任务 | seed | 结果 | 策略周期 | InvalidAction | 全 IK 链耗尽 | 控制预算耗尽 |
|---|---:|---|---:|---:|---:|---:|
| BimanualHandoverItem | 2608000000 | 成功 | 301 | 0 | 0 | 0 |
| WipeDesk | 2608000199 | 成功 | 240 | 0 | 0 | 0 |
| BimanualLiftTray | 2608000001 | 成功 | 106 | 0 | 0 | 0 |

Handover 在第一次验证中没有因早期 external 瞬态立即进入错误恢复：共动窗口在第3个真实验证响应后开始提供方向，探测持续到上限并完成返回；返回期间关系状态发生变化，因而按既有“关系状态、任务状态或抓取事件变化才允许重试”规则重新解锁了一次短验证。第二次验证后 benchmark 正常成功。该运行证明本轮发现的**单周期假 external → 错误 LINK 恢复**链已被阻断，但关系滤波与窗口方向仍发生过切换，不能把它表述为所有在线关系扰动都已完成验收。

## 原始结果与校验

原始逐周期结果属于本地/Release 评测产物，不纳入普通 Git 源码历史；本报告固定其路径和摘要：

```text
14a8f798210eb4fe1320cc0fddad6cbabd242eb7a1570665ab10628d2919ad06  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_bimanual_handover_item_i000_final_bundle_v3.json
38c4edcd71057f2493c2c74567b3dc3e3d10c9607546b84491f14201bf94708c  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_wipe_desk_i199_goal_poe_executor_v11_verify_window.json
d7cae1f9d926bbe801a9fa485f07b129e2395fe326f49357071092784c55bd0a  integrations/rlbench/results/diagnostics/phase6_normal_smoke_20260829_bimanual_lift_tray_i001_goal_poe_executor_v11_verify_window.json
```

三个同名无 `.json` 目录保存逐周期 `episode_0000.jsonl`。

## 结论边界

本轮确认联合目标评分、执行器停滞升级和主动验证窗口已经接入真实控制链，并在三个封存正常样本中没有引入 InvalidAction 或任务失败。它不是正常全任务矩阵，也不是故障/扰动 A/B；阶段六仍需完成完整正常回归和随后冻结的故障消融，才能最终验收。
