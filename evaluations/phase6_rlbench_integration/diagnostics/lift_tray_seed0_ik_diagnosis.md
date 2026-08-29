# LiftTray seed 2608000000 IK 诊断

> 后续状态（2026-08-28）：该报告对应旧执行器。通用混合执行器复跑同一 seed
> 后，IK 无效动作已降为0；剩余任务失败发生在抓取/关系后的状态重入。最新结果
> 见同目录 `HYBRID_EXECUTOR_RESULTS.md`，本文件只作为改进前根因证据保留。

## 运行范围

- 任务：`bimanual_lift_tray`
- 条件：正常 static，variation 0，seed `2608000000`
- 策略：`closed_loop_multistream`
- 执行器：`stage6_converged_cartesian_feedback_v4`
- 上限：400个策略控制周期
- 数值模型：正式 V4 DynaMAC 模型与当前 `closed_loop_v1` 侧车模型
- 评测样本：从封存 `rlbench_eval_v2` 只读选取第1个样本的诊断子集运行；不是200样本正式结果单元。

Python 3.8 仿真进程在启动前通过固定 bounded TRAC-IK 的版本、ABI、API和本地扩展哈希检查。本次 `trac_ik_distance_factory_failures=0`，不再存在旧诊断遗漏 Python 扩展路径的问题。

## 结果

- 任务在400周期上限时未成功；
- 完整执行23个策略动作，377个周期因关节目标不可用而提交原关节保持；
- TRAC-IK 共800次目标请求，422次成功、378次求解耗尽，求解器创建失败0次；
- 连续伪逆备选378次全部失败；碰撞感知采样备选只成功1次，失败377次；
- 两臂目标按 `right → left` 顺序在任何物理步前准备。400个周期均进入了左臂求解，说明右臂每周期都至少由某一级求解器获得候选；377个无效周期都在随后的左臂求解中耗尽。聚合计数器可以确定失败臂，但现有结果没有按臂拆分那1次采样成功，因此不对单臂 TRAC 成功数作过度精确归因；
- 左臂从 `k0:t20` 开始间歇求解失败，在 `k0:t25` 上自第152周期起持续失败直到结束。第152周期的位置/旋转残差约为 `0.02255 m / 0.15099 rad`；最终由于不执行和物理漂移扩大为 `0.07085 m / 0.19708 rad`；
- 右臂的求解本身未耗尽，但双臂动作采用执行前原子准备：左臂无解时两臂都不执行。长期静止后右臂从第322周期开始进入 `NO_PLAUSIBLE_STATE`，这是执行停滞的下游结果。

## 与冻结 V4 对照

冻结 DynaMAC V4 在同一 seed 上也未成功：116个策略状态后结束，其中40次无效动作。V4 的200个 LiftTray static 样本中190个成功、10个失败，10个失败样本均有32至40次无效动作。这说明该 seed 属于基线共享的困难初始布局，但当前闭环执行器在同一局部目标上的重复耗尽仍是需要改进的阶段六执行问题。

## 产物

- 汇总：`integrations/rlbench/results/diagnostics/phase6_normal_lift_seed0_correct_trac_v5.json`
- 逐周期轨迹：`integrations/rlbench/results/diagnostics/phase6_normal_lift_seed0_correct_trac_trace_v5/bimanual_lift_tray/episode_0000.jsonl`
- 汇总 SHA256：`3a3e2d67ca6e8a29eb3348a4fa7494f8dc9a813e8996f4ad393175f6b396f95d`
- 轨迹 SHA256：`83b8a45d6dd7f381b6d3b64bb016c3318d3d4772bc70c5b5cdba297651df52d9`

## 解释边界

TRAC-IK 本身只做运动学求解，本轮在不依赖碰撞规划的 TRAC 层已经出现左臂耗尽，因此不能把失败归因为“开启了碰撞检测”。关闭采样或路径层的碰撞检测可能扩大候选集，但无法消除工作空间、姿态、关节限位、奇异位形和局部运动学分支问题，而且会生成实际不安全的动作。
