# DynaMAC 夜间研发最终报告

## 总体结果

六阶段任务已在分支 `codex/overnight-dynamac-audit` 完成。本轮把已有单臂工程原型
加固为可审计科学基线，明确区分论文来源与项目新增机制，完成六方法、480 次隔离仿真
评测，并补齐双臂交接最小环境/数据链路；没有把它包装成完整双臂 DynaMAC 结果。

最强的新结果是机制证据，而不是端到端成功率优势：RelationDynaMAC 在 40 ms 内
撤销全部强制掉落关系，并拒绝全部空抓闭合；两个旧在线变体则一直保留关系到预定张开，
且每次空抓都误连接。但是，三种方法在六个普通条件上都为 51/60，在明确掉落/空抓
恢复条件上全部失败。因此，在线关系估计被验证为检测器，而不是恢复策略。

## 完整性与受保护资产

- `DynaMAC.pdf` 未修改，SHA-256 仍为
  `3fdf0a6ac46bced00885ea01e2a21d918ce12f4fd832a3e0b2d97ed34af10431`。
- 单臂冻结数据从未重写。所有方法使用同一五演示 `data/pick_place_static/v1`，
  数据集 SHA-256 为
  `8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8`。
- 测试 seed 6300–6309 在实现和阈值冻结后保留；它们是仿真实例，不是独立训练集。
- 旧输出全部保留；新科学输出使用新目录，并记录源码、配置和数据指纹。
- 旧 `data/handover_static/v1` 保持不变；修正后的四标签数据是独立冻结 v2。

## 阶段结果

### 1. 单臂指标与旧结果审计

语义成功现在同时要求 XY 放置、支撑高度、夹爪释放和释放后稳定。旧三维误差只作
诊断。审计重算 72 条旧试验，发现目标参考点与支撑几何存在固定 59 mm Z 偏差，
因此旧三维阈值具有误导性。

18 对 Full/Mask 试验中，Full 平均路径短 104.6 mm，主要来自抬升 77.8 mm 和
搬运 25.0 mm，但只在 10/18 对中更短。路径差与时长差相关系数为 0.86。这支持
耦合参考系/计时效应，不支持全局路径最优。

详细报告：[单臂科学审计](single_arm_scientific_audit.md)。

### 2. 论文忠实的简化 SkillDynaMAC

`SkillDynaMACPolicy` 实现固定人工技能、六维位姿协方差、简化 Eq. (5) 连接选择、
简化 Eq. (6) 参考系选择和只由五条演示拟合的技能局部虚拟参考系；它与旧在线原型
明确分离。

90 条干净试验中，SkillDynaMAC 在静态/机械臂偏置相关试验成功 10/18，在 15 条
合格扰动中恢复 8 次，但目标运动条件为 0/18。简化协方差规则会过度选择静态参考系；
这是有价值的忠实基线负结果。

详细报告：[方法来源与实现边界](method_provenance.md)；输出目录
`outputs/single_arm_scientific/skill_baseline_v1`。

### 3. 速度分段诊断

数据驱动诊断在每条冻结演示中都恢复五个可重复速度片段和四个边界簇，全部边界簇都有
五条演示支持。对齐边界时间标准差平均 39 ms，但离最近人工边界平均 211 ms。发现
的是粗粒度“接近/抓取、搬运/放置、撤离”结构，不是语义 TAPAS 技能。

详细报告：[速度技能分段](segmentation_analysis.md)；输出
`outputs/single_arm_scientific/segmentation_v1_clean/analysis.json`。

### 4. 双向在线关系估计器

新估计器使用四个滞回状态（`DISCONNECTED`、`CANDIDATE_CONNECTED`、
`CONNECTED`、`CANDIDATE_LOST`），读取实测夹爪开度/速度、六维相对运动、窗口
稳定性、共同运动相关性、可选接触和连续置信度。全部阈值只从冻结演示标定。

确定性测试覆盖空抓、成功搬运、闭合夹爪掉落和外部未抓取物体运动。干净仿真冒烟中，
六个普通条件全部成功；强制掉落 40 ms 内断开，空抓从不连接。两个反例都未恢复，
因为没有重抓或阶段重规划图。

详细报告：[双向在线关系估计器](online_relation_estimator.md)；输出
`outputs/single_arm_scientific/relation_calibration_v1_clean/analysis.json` 和
`outputs/single_arm_scientific/relation_smoke_v1_clean`。

### 5. 留出 seed 单臂扩展评测

验收矩阵包含六方法、八条件、十 seed：480 个独立 Isaac worker、480 对 JSON/NPZ、
480 个唯一组合和指纹、统一 schema/源码/数据身份、零缺失指标，阶段路径分区最大
残差为 `6.67e-16 m`。

六个普通条件的稳定放置结果：

| 方法 | 成功 | 平均 XY 误差 | 平均路径 | 平均计算时间 |
|---|---:|---:|---:|---:|
| World Gaussian | 0/60 | 164.91 mm | 1.129 m | 0.035 ms |
| Static Multi-stream | 9/60 | 34.28 mm | 1.164 m | 0.219 ms |
| SkillDynaMAC | 38/60 | 25.04 mm | 1.113 m | 0.296 ms |
| Mask-only | 51/60 | 4.76 mm | 1.222 m | 0.244 ms |
| 旧 Full | 51/60 | 4.74 mm | 1.084 m | 0.249 ms |
| RelationDynaMAC | 51/60 | 5.04 mm | 1.117 m | 0.899 ms |

反例机制结果：

| 方法 | 掉落丢失延迟 | 空抓误连接 | 掉落恢复 | 空抓恢复 |
|---|---:|---:|---:|---:|
| Mask-only | 0.910 s | 10/10 | 0/10 | 0/10 |
| 旧 Full | 0.880 s | 10/10 | 0/10 | 0/10 |
| RelationDynaMAC | 0.040 s | 0/10 | 0/10 | 0/10 |

Wilson 区间、seed 平衡 bootstrap、恢复、建立/释放、阶段路径、跳变、最大速度、失败
分类和精确复现见 [单臂扩展评测](single_arm_final_report.md)。验收输出为
`outputs/single_arm_scientific/v1/summary.json`。

### 6. 最小双臂交接链路

`Essay2608-Bimanual-Handover-v0` 现在暴露两套独立绝对位姿 Franka IK、两个独立
夹爪，以及左右末端、物体、目标和实测双指观测。13 状态专家记录完整的
`none → left_only → both → right_only → none` 监督序列。

独立 seed 7300 冒烟在 575 步成功，最终误差 10.62 mm。正式 v2 采集从八个隔离
worker 中接受五条完整 episode，每条都有 15 步/0.30 s 共同持物。冻结 v2 哈希为
`91706df18abfea606c9e6836f1864e675610633ce5cb0c3c23846a1ea4f5fe18`；
最大最终误差 11.04 mm、最大单步笛卡尔位移 29.21 mm、初始物体最小间距
13.91 mm。重复冻结和重复采集都会失败关闭，manifest 不变。

详细报告：[双臂交接环境与数据](bimanual_handover_setup.md)；冻结数据位于
`data/handover_static/v2`。

## 提交记录

| 提交 | 交付内容 |
|---|---|
| `4d0b806` | 审计单臂成功指标和实验有效性 |
| `c979a94` | 增加论文忠实的技能级 DynaMAC 基线 |
| `48346b7` | 记录 SkillDynaMAC 基线评测 |
| `7bfdfc4` | 增加速度技能分段诊断 |
| `0620b7b` | 记录干净分段分析指纹 |
| `1143f17` | 原型化双向在线关系估计 |
| `7362714` | 记录干净在线关系评测 |
| `3673dd2` | 准备扩展单臂评测指标 |
| `e8df22c` | 运行扩展单臂 DynaMAC 评测 |
| `5c4921e` | 准备可审计双臂交接 v2 数据 |
| `7489495` | 增加双臂交接环境和脚本演示 |
| `ab1d4ef` | 汇总夜间 DynaMAC 研究结果 |

## 验证与复现命令

最终纯测试和语法检查：

```bash
conda run -n env_isaaclab python -m pytest -q
conda run -n env_isaaclab python -m compileall -q \
  source/essay2608/essay2608 scripts tests
git diff --check
```

结果为 20 项测试通过。环境当时没有可选 `ruff` 可执行文件，因此未把它作为验收信号。

单臂完整集成命令：

```bash
conda run -n env_isaaclab python scripts/eval_single_arm.py --headless \
  --methods world_gaussian static_multistream skill_dynamac mask_only \
  full_dynamac relation_dynamac \
  --conditions static smooth_object sudden_object smooth_target sudden_target \
  arm_offset drop_after_grasp close_without_grasp \
  --seeds 6300 6301 6302 6303 6304 6305 6306 6307 6308 6309 \
  --output_dir outputs/single_arm_scientific/v1
```

双臂采集和审计命令：

```bash
conda run -n env_isaaclab python scripts/collect_handover.py --headless \
  --num_demos 5 --max_attempts 10 --seed 7400 \
  --output_dir data/handover_static/v2
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir data/handover_static/v2
```

## 已解决与未解决问题

已解决：

- 成功定义改为语义稳定放置，不再依赖宽松三维位置；
- 全部单臂方法使用同一不可变五演示训练集；
- 论文忠实简化 SkillDynaMAC 与在线项目代码明确分离；
- 速度分段的可重复性与语义不匹配已量化；
- 在线关系建立和丢失有机制反例覆盖；
- 检测正确性与端到端恢复明确分开；
- 路径变化可按目的阶段归因；
- 双臂交接具备显式观测/标签/数据契约和不可变审计 v2。

未解决：

- 没有策略在 `CANDIDATE_LOST` 或空抓后返回接近/重抓；
- 关系检测尚未提高普通条件任务成功率；
- SkillDynaMAC 简化 Eq. (5–6) 对极小朝向协方差和目标动态敏感；
- 速度片段不是学习到的语义技能；
- 单臂证据仅来自一个自定义任务，同一 seed 内条件切片相关，不能代表广泛基准；
- 双臂物体禁用重力并运动学附着到单一载体，脚本 `both` 不是实测接触真值；
- 旧双臂策略试验不属于本次验收骨架，不应引用为完整 DynaMAC 比较。

## 论文主张与项目方向

证据支持一条窄论文叙事：关系改变后，静态任务参数专家乘积可能产生因果有害反馈；
关系感知屏蔽和虚拟参考系能改善单臂动态行为；读取实际状态的双向关系估计器能修复
阶段/夹爪命令锁存无法识别的强制丢失和空抓错误。

证据不支持等价于 TAPAS、MiDiGaP、黎曼 DynaMAC 或 DynaBench，不支持
RelationDynaMAC 普通任务成功率更优，不支持丢失后恢复、接触丰富双臂交接或广泛泛化。

最清晰的项目自研方向是：把已经验证的四状态关系生命周期接入恢复图，再用实测双臂
接触和相对运动证据替换脚本关系标签。只有这些机制通过因果反例后，才应拟合和比较
并发双臂 DynaMAC。

## 次日三项优先检查

1. 检查十条代表性视频/轨迹：SkillDynaMAC、Full、RelationDynaMAC 各一成功一失败，
   再加掉落与空抓反例，确认数值失败分类和物理现象一致。
2. 预注册恢复状态协议（`LOST → retreat → re-approach → regrasp`）及成功/恢复标准，
   然后再运行新测试 seed。
3. 在“接触丰富双臂物理”和“改进论文忠实参考系/连接估计”之间明确下一项投入；不要
   用当前几何骨架声称双臂学习结果。
