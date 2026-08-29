# 阶段六通用混合末端执行器：调研、实现与定向验证

## 1. 结论

当前 RLBench 最合适的执行器不是单独更换为某一个 IK 求解器，而是保留策略输出
的绝对末端目标，在 benchmark 适配层采用“终点 IK + 小步笛卡尔闭环 + 有界路径
后备”的混合执行器。正式配置标识为：

```text
profile = stage6_hybrid_cartesian_executor_v5
protocol = rlbench-stage6-hybrid-cartesian-continuation-v7
```

该实现不读取任务名或固定 StateId，不改变 DynaMAC/闭环策略的目标、关系、进度、
边界或恢复机制。困难 LiftTray 样本中，旧执行器的大量 IK 无效动作已降为 0；
剩余失败明确发生在抓取/关系后的策略重入，而不是端点 IK 耗尽。因此执行器的
致命阻断已解除，但不能据此宣称该 LiftTray 样本已经成功。

## 2. 执行器方法调研

### RLBench 与现有策略常用路径

RLBench 官方 `EndEffectorPoseViaPlanning` 以末端位姿为目标，先尝试线性 IK 路径，
再使用采样式路径规划；ARM 和其他 RLBench 方法也普遍复用该规划入口。这类方案
易于接入 benchmark，但对精确接触目标、局部不可解和规划预算耗尽仍可能脆弱。

- RLBench 官方仓库：<https://github.com/stepjam/RLBench>
- ARM 官方实现：<https://github.com/stepjam/ARM>

### TRAC-IK

TRAC-IK 并发运行改进的 Newton/KDL 搜索与 SQP 优化，对关节限位和局部极小值
通常比单一伪逆稳定，适合作为当前实时运动链上的终点 IK 主求解器。官方同时
明确指出它不使用 mesh 判断自碰撞，因此它不能单独替代路径与碰撞处理。

- TRAC-IK 官方仓库：<https://github.com/HIRO-group/trac_ik>

### MoveIt Servo

MoveIt Servo 采用连续笛卡尔/关节命令和实时状态反馈，并提供奇异性、关节限位、
碰撞减速和信号平滑。它的闭环小步思想最适合当前问题；但完整引入需要 ROS、
URDF/SRDF、planning scene 和控制器桥接，与现有 CoppeliaSim/RLBench 进程重复。
因此本次复用其通用控制思想，而不是迁移整个 MoveIt 栈。

- MoveIt Servo 官方文档：<https://moveit.picknik.ai/main/doc/examples/realtime_servo/realtime_servo_tutorial.html>

### cuRobo

cuRobo 提供 GPU 批量碰撞 IK、几何规划和轨迹优化，适合未来替换为高吞吐执行
后端。但当前 RLBench 仿真进程尚无 cuRobo、Torch 或同步的 GPU 碰撞世界模型；
立即迁移需要先建立 CoppeliaSim 几何、关节状态和碰撞对象的持续同步，已经超出
本轮“修复通用执行器”的范围。

- cuRobo 官方文档：<https://curobo.org/>

## 3. 最终实现

每个原始策略目标按以下顺序执行：

1. 在实时提取的 Panda 运动链上运行 current-seeded bounded TRAC-IK Distance；
2. 失败后运行带关节连续性检查的当前种子伪逆；
3. 全目标仍不可解时，以 5 mm / 2° 生成同一目标的局部 SE(3) 子目标，并按
   `1, 0.5, 0.25, 0.125` 回退；
4. 每次物理执行后重新观测末端、重新提取当前链并重新求解，最多 96 段；
5. 局部链耗尽后先尝试碰撞感知采样，再尝试碰撞放宽采样；
6. 路径后备先尝试碰撞感知线性路径，再尝试碰撞放宽线性路径；只有平移距离
   大于 10 cm 的远目标才进入碰撞感知 RRTConnect，避免近距离接触目标在非线性
   规划器中长时间阻塞。

数值 IK 与执行完成使用同一 2 mm / 1° 控制接受范围；0.5 mm / 0.25° 只作为
更严格的物理完成审计。内部子目标不会作为策略动作提交：执行器只有在原始目标
到达时返回 `reached`，有实际改善但未到达时返回 `progressed`，合法命令不再改善
时返回 `stopped`。

双臂先完成全部候选准备再推动共享物理时钟。物理进展采用 Pareto 口径：至少
一臂改善且没有任何一臂退化，避免一臂的大幅改善掩盖另一臂的退化，也不要求
两臂在每个原始物理步中同步改善。

## 4. 真实 RLBench 定向结果

| 运行 | 结果 | 策略周期 | IK 无效动作 | 关键解释 |
| --- | ---: | ---: | ---: | --- |
| LiftTray seed `2608000000`，旧执行器 | 失败 | 400 | 377 | 左臂局部终点反复求解耗尽 |
| 同一 seed，混合执行器 | 失败 | 131 | **0** | 推进至后续抓取/关系状态，最终为左臂 `no_legal_reentry_state` |
| LiftTray seed `2608000001`，混合执行器 | 成功 | 108 | **0** | 正常成功样本未出现执行器回归 |

困难 seed 的混合执行器共调用 TRAC-IK 895 次，352 次直接成功；全目标 TRAC
失败后由伪逆成功 124 次，小步延续成功准备 190 段，最终没有进入采样/路径的
全链耗尽。正常成功 seed 调用 TRAC-IK 368 次、成功 211 次，其余 157 次全部由
伪逆接续，也没有无效动作。

该结果证明：

- `TRAC-IK failure` 不再等于策略动作失败，局部闭环可以继续执行同一原始目标；
- 已知困难 seed 中“数值 IK 无解导致数百次关节保持”的主瓶颈已经消除；
- 该困难 seed 的任务失败已经转移到抓取/关系和重入语义，必须单独诊断，不能
  再靠放宽 IK 或碰撞门限掩盖；
- 当前结果是两个封存样本的定向执行器验证，不是完整成功率或论文正式评测。

原始结果：

- `integrations/rlbench/results/diagnostics/phase6_normal_lift_seed0_hybrid_executor_v7_h140.json`
- `integrations/rlbench/results/diagnostics/phase6_normal_lift_seed0_hybrid_executor_v7_h140_trace/`
- `integrations/rlbench/results/diagnostics/phase6_normal_lift_seed1_hybrid_executor_v7.json`
- `integrations/rlbench/results/diagnostics/phase6_normal_lift_seed1_hybrid_executor_v7_trace/`

上述运行目录按项目规则不进入 Git。可随代码提交的最终副本位于：

- `evaluations/phase6_rlbench_integration/diagnostics/results/hybrid_executor_v7/seed0_difficult.json`
- `evaluations/phase6_rlbench_integration/diagnostics/results/hybrid_executor_v7/seed0_difficult_trace.jsonl.gz`
- `evaluations/phase6_rlbench_integration/diagnostics/results/hybrid_executor_v7/seed1_normal.json`
- `evaluations/phase6_rlbench_integration/diagnostics/results/hybrid_executor_v7/seed1_normal_trace.jsonl.gz`

## 5. 验收边界

- 单元测试覆盖全目标 IK、小步延续、回退、逐段重观测、碰撞感知到放宽采样、
  线性路径顺序、近目标跳过非线性规划和元数据完整性；
- 混合执行器没有任务名、seed、技能编号或固定状态阈值分支；
- 执行器仍是 RLBench benchmark 适配层，不进入阶段一至五的核心算法；
- 当前 Python/CoppeliaSim 运行中，困难 seed 的 TRAC 求解累计约 45.6 秒，正常
  seed 累计约 18.8 秒，满足离线仿真评测，但不等同于真实机器人实时频率验收；
- 阶段六仍需完成正常全任务回归和正式故障/扰动消融，才能验收完整技术路线。
