# 单臂 DynaMAC 最小闭环

本阶段刻意限制在现有 Franka 抓取放置任务中。在引入双臂环境前，先验证数据集、
相对几何现象、高斯基线、动态屏蔽、虚拟末端参考系和受控测试时扰动。

## 冻结数据集

`data/pick_place_static/v1` 是不可变输入。manifest 记录逐文件 SHA-256、seed、
初始位姿、状态覆盖、步数、最终误差、复位跳变诊断、连接稳定性和来源 Git 提交。
`scripts/collect_demos.py` 拒绝把带 `FROZEN` 的目录作为输出。

验证命令：

```bash
python scripts/audit_dataset.py --data_dir data/pick_place_static/v1
```

## 复现最小闭环

```bash
python scripts/analyze_relative_frames.py
python scripts/train_single_arm.py
python scripts/eval_single_arm.py --headless --seeds 6200 6201 6202 \
  --output_dir outputs/single_arm_strict/v2
```

每条评测 rollout 都使用新的 Isaac Lab 进程。试验包含：

- World Gaussian；
- 静态物体/目标 Gaussian Product-of-Experts；
- Mask-only 多流策略；
- 带连接检测和虚拟末端参考系的 Full DynaMAC；
- 静态、物体平滑/突发移动、目标平滑/突发移动和机械臂命令偏置条件。

恢复运行采用内容寻址。只有指纹中的 Git 提交、冻结数据、相关源码、方法、扰动、
seed、rollout 设置、成功标准和生成模型 checkpoint 全部匹配时，才复用缓存。

## 语义成功与命令连续性

旧 60 mm 三维半径仅作为 `legacy_success_3d` 保留。它不能作为主要成功标准，因为
目标命令使用 `z = 0.08 m`，而释放后的方块中心稳定在约 `z = 0.021 m`，正确放置
天然带有约 59 mm 竖直残差。新的主要成功定义要求：

- 物体—目标 XY 误差低于 10 mm；
- 物体高度位于冻结演示测得支撑高度的 10 mm 内；
- 夹爪张开；
- 最后 25 个控制步的位移小于 5 mm，最大速度低于 0.05 m/s。

评测器还报告 5、10、20 mm XY 阈值下的组合成功。共享笛卡尔限速器把策略命令
限制为每控制步 20 mm；原始策略意图、限速命令、受控扰动后命令和参考系切换诊断
分别保存。

[单臂科学审计](single_arm_scientific_audit.md) 的可加阶段归因表明，Full 的汇总
路径减少与阶段 4/5 更短的执行耦合。Full 只在 10/18 对试验中更短，并在 seed
6202 的每个条件都更长，因此虚拟参考系不支持与 seed 无关的效率主张。

## 三 seed 受控结果

seed `6200`、`6201`、`6202` 共覆盖 72 条独立仿真 rollout（四方法 × 六条件 ×
三 seed）。表中先在每个 seed 内对条件取平均，再统计跨 seed 数值。

| 方法 | 平均成功 | 平均恢复 | XY 误差均值 [95% bootstrap CI] | 路径 | 策略计算 |
|---|---:|---:|---:|---:|---:|
| World Gaussian | 0.0% | 0.0% | 200.48 [53.11, 340.75] mm | 1.128 m | 0.039 ms |
| Static Multi-stream | 0.0% | 0.0% | 42.46 [29.96, 59.25] mm | 1.219 m | 0.233 ms |
| Mask-only | 100.0% | 100.0% | 3.01 [1.91, 3.57] mm | 1.226 m | 0.259 ms |
| Full DynaMAC | 100.0% | 100.0% | 2.98 [1.99, 3.73] mm | 1.122 m | 0.262 ms |

搬运阶段中，静态 PoE 给物体流分配约 79% 位置精度，尽管此时物体已经是机器人动作
的结果。动态屏蔽移除了该内生流。三 seed 汇总中，Full 保持 Mask-only 的全部成功，
并把平均路径缩短约 8.5%；但 seed 6202 上该效应反向约 3.6%。

Mask-only 和 Full 在更严的 5 mm XY 阈值下均为 17/18，在 10 和 20 mm 下均为
18/18，因此成功不是阈值边缘伪影。Full/seed 6202 的最大原始参考系切换请求仍约
406 mm，但限速后策略跳变为 20 mm；全部试验最大实测末端速度约 1.01 m/s。
`arm_offset` 在限速后刻意注入 60 mm 测试扰动，因此最大扰动后跳变约 80 mm。

三个 seed 是受控试验证据，不是论文规模样本。它们改变评测初始化，但所有策略都拟合
同一冻结五演示数据，因而测量该数据集上的测试时稳健性，不是多个五演示训练集方差，
也不是 5-shot 样本效率。summary JSON 还包含逐条件 Wilson 区间、确定性 bootstrap
区间、语义失败原因和单条 rollout 记录。

生成 seed 6202 转移审计和图：

```bash
python scripts/analyze_action_transitions.py \
  --result_dir outputs/single_arm_strict/v2
```

## 条件扩散基线

后续生成式比较使用低维条件 DDPM：八动作 chunk、32 个去噪步、177,984 个参数。
它不是官方图像 Diffusion Policy 架构的复现。

```bash
python scripts/train_diffusion.py
python scripts/eval_single_arm.py --headless \
  --methods diffusion_policy --seeds 6200 \
  --output_dir outputs/diffusion/v2/strict_eval
```

严格单 seed 试验中，它完成四个条件，但 XY 误差为 42.4–58.9 mm，两个移动目标
条件未完成，总成功与恢复为 0/6。该固定数据负结果只支持一个窄结论：此小型 DDPM
在这五条演示上没有追平显式几何方法；它不是样本效率估计，也不能作为反对完整视觉
Diffusion Policy 的证据。
