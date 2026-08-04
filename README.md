# DynaMAC 论文算法复现

这个仓库现在只做一件事：复现论文 *One Hand Watches The Other: Dynamic Multi-Agent
Cooperation for Sample-Efficient Bimanual Manipulation in Dynamic Environments* 中公开的
DynaMAC 算法。过去的在线关系检测、故障恢复状态机和大量 Isaac Lab 验证脚本已经退出
主代码；历史结论压缩在一份中文日志中，完整旧文件仍可从 Git 历史恢复。

## 只需要看的文件

1. `source/essay2608/essay2608/policy/dynamac.py`：DynaMAC 全部核心；
2. `scripts/run.py`：唯一运行入口；
3. `configs/dynamac.json`：全部阈值和数值选择；
4. `tests/test_dynamac_reproduction.py`：公式与 Algorithm 1 验证；
5. `logs/research_log.md`：论文理解、实验记录、已知边界。

`policy/` 只放策略。目前有完整 DynaMAC 和一个明确标注为低维工程对照的 Diffusion
Policy。没有用空壳文件假装已经实现 ACT；以后新增 ACT、DP 官方版等策略，也只能放在
这个目录，并必须独立标注复现范围。

## 实现覆盖

`DynaMAC` 逐项实现：

- 公式 (1)：`R3 × S3` 位姿到局部任务参数坐标系；
- 公式 (2)：黎曼高斯 marginal 随当前任务参数位姿变回世界系；
- 公式 (3)：在共同世界切空间通过 Log/Exp 迭代完成 Product-of-Experts；
- 公式 (5)：六维协方差的 geometric mean standard deviation；
- 公式 (6)：逐时刻相对精度归一化并取时间最大值；
- Algorithm 1：离线技能序列、链接过滤、累积虚拟末端帧、任务参数选择、逐技能
  DiGaP/MiDiGaP 拟合与顺序执行；
- MiDiGaP：真正的 Karcher/Fréchet 均值、对角切空间协方差、`M^T` 上的 Riemannian
  k-means+BIC，以及按演示集合交集估计的跨技能模态转移与整条模态路径选择；
- 双臂：左右两套独立 DynaMAC，并将对侧末端加入候选任务参数，不使用联合动作策略、
  固定 leader 或额外协调器；
- 单文件、无 pickle checkpoint 与内容指纹。

这里严格遵守论文的真实推理语义：链接与有效流集合在训练时按技能固定；推理时只更新
已保留参考系的当前位姿，并按离散时间索引切技能。它不会在线确认抓取、滑移或接触，也
不会自动恢复失败。此前的在线扩展不能再使用 `DynaMAC` 名称。

运动学链接由实际观测到的末端轨迹与参考帧的相对位姿判定；控制目标轨迹只用于拟合策略
分布，两者不会混用。新技能的虚拟末端帧在该技能第一条观测到达时冻结。

## 运行

```bash
python -m pip install -e '.[test]'

# 不写模型，只验证随附五条单臂和五条双臂演示能走完整训练链
python scripts/run.py verify

# 保存单臂 checkpoint
python scripts/run.py fit --task single --output outputs/single_dynamac.npz

# 保存左右两套双臂 checkpoint
python scripts/run.py fit --task bimanual --output outputs/bimanual_dynamac

# 检查 checkpoint 内的链接、流选择和模态
python scripts/run.py inspect outputs/single_dynamac.npz

pytest -q
```

随附 `data/dynamac_demos.npz` 把原来分散的文件压为一个无 pickle 包，包含五条单臂和
五条真实接触双臂交接。它只用于接口与算法结构验证：技能标签由旧脚本状态合并而来，
不是 TAPAS 输出，任务参数变化也不足以复现论文 DynaBench 的成功率。

## “完整复现”的边界

这里的“完整”指论文正文公开的 DynaMAC 算法和其 MiDiGaP 数学实例，而不是声称已经
重跑论文的 RLBench2/DynaBench、真机视觉或表 I–IV。论文官网指向的官方 DynaMAC
GitHub 仓库截至 2026-08-04 仍只有 `Coming soon`，MiDiGaP 官网同样尚未发布代码；
TAPAS 分段、DINO 位姿感知、RLBench2 任务和真机控制是外部依赖，不属于 Algorithm 1
本体。所有无法从论文唯一确定的数值选择都显式放在配置和研究日志中。
