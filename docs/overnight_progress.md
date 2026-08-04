# DynaMAC 夜间研发过程记录

开始时间：2026-08-04（Asia/Shanghai）

## 基线与约束

- 工作分支：`codex/overnight-dynamac-audit`
- 起始提交/检查点：`57e01c4fef0cacda9a37e7313afcff40c0b1b496`
  （标签 `single-arm-v1`）
- 起始工作树干净，因此没有制造额外检查点提交。
- Python：3.10.19
- Isaac Lab：0.54.0
- Isaac Sim：4.5.0.0
- PyTorch：2.7.0+cu128
- NumPy：1.26.4
- GPU：NVIDIA GeForce RTX 4090，驱动 550.90.07
- 单臂冻结数据：`pick_place_static/v1`，五条演示，SHA-256
  `8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8`

冻结数据及 manifest、`FROZEN`、旧输出目录和论文 PDF 均作为只读输入。测试 seed
只用于评测，不参与阈值调整。

## 执行计划

1. 重新审计单臂严格修正，补充阶段级因果诊断且不覆盖旧摘要。
2. 区分方法来源，实现论文忠实的简化技能级基线，并保留兼容名称。
3. 增加只用于诊断的速度技能分段。
4. 增加双向在线关系估计器，用训练/标定数据阈值运行合成与仿真反例。
5. 在至少十个留出 seed 上评测稳定方法，把逐试验证据保存到新指纹目录。
6. 审计现有双臂交接环境、专家、冻结数据和冒烟链路，只补缺口，不重建系统。

## 阶段日志

### 阶段 1：科学审计

状态：完成。

起始检查点已经修正固定 59 mm 竖直残差、语义放置成功、阈值灵敏度、动作限速、
终止后场景读取和实验指纹。本阶段继续量化 Mask-only 与 Full 的逐阶段差异。

完成内容：

- 增加旧三维、XY 和组合稳定放置指标；
- 增加可加阶段归因模块和轻量分析入口；
- 不重跑、不覆盖地重算全部 72 条严格轨迹；
- 阶段分区以 `2.3e-16 m` 内误差重构每条保存路径；
- Full 只在 10/18 对中更短；Full 减 Mask 的抬升路径均值为 -77.80 mm，移至目标
  上方为 -24.99 mm，其余阶段合计约 -1.81 mm；
- 时长/路径关联 `r = 0.86`，强制转移为零，因此结论限制为耦合的虚拟参考系/计时效应；
- 新增 [单臂科学审计](single_arm_scientific_audit.md)。

验证：

- `python -m compileall -q source/essay2608/essay2608 scripts tests`
- `python -m pytest -q`：5 项通过
- `python scripts/analyze_phase_diagnostics.py ...`：36 条 Mask/Full 轨迹、18 个精确配对
- `git diff --check`

提交：`4d0b806 Audit single-arm success metrics and experiment validity`。

### 阶段 2：方法来源与技能级基线

状态：完成。

实现内容：

- 把运行时工程控制器明确命名为 `OnlineDynaMACPrototype`，保留
  `DynaMACPolicy` 兼容别名和 `full_dynamac` 结果标签；
- 增加 `SkillDynaMACPolicy`，使用项目阶段标签、高斯拟合和平移专家乘积实现
  Algorithm 1 与 Eq. (5–6) 的简化版本；
- 增加六维位置/旋转向量协方差，同时保留用于动作融合的原三维协方差；
- 增加十个技能起点虚拟参考系、固定逐技能选择和可序列化训练诊断；
- 把显式策略/扰动配置写入评测指纹，并提升 schema；
- 在 [方法来源](method_provenance.md) 中记录精确来源和缺失组件。

训练集诊断表明，不加权六维行列式被本数据集中极小旋转方差主导，把阶段 0 和 2–9
的物体都判断为连接；Eq. (6) 在阶段 5 也过度选择历史虚拟参考系。这些被保留为限制，
没有用评测 seed 调权隐藏。

预提交验证：

- `python -m pytest -q`：8 项通过
- 静态 seed 6200：稳定成功，XY 误差 4.65 mm，三维误差 59.18 mm，322 步，
  路径 1.037 m，无强制转移
- 六条件 seed 6200：4/6 成功；两个 10 cm 目标移动均以 67.83 mm XY 误差失败

提交 `c979a94` 的干净评测：

- 90/90 个唯一组合、90 个 JSON 和 90 个 NPZ、指标齐全、schema 3、统一源码/数据哈希；
- World 与 Static 各 0/18；
- SkillDynaMAC 成功 10/18、恢复 8/15，物体移动全成功，六个目标移动全失败，
  条件平衡 XY 误差 25.95 mm、路径 1.152 m；
- 固定连接标签相对脚本连接阶段的平均假阳性比例为 0.621、假阴性为零；平均原始
  参考系切换 0.259 m，经限速为 0.020 m；
- Mask-only 与在线旧原型各 18/18 成功、15/15 恢复，但它们不是论文忠实方法。

实现提交：`c979a94 Add paper-faithful skill-level DynaMAC baseline`。
评测记录提交：`48346b7 Record SkillDynaMAC baseline evaluation`。

### 阶段 3：自动技能分段诊断

状态：完成。

- 使用末端线/角速度、共同训练分位数标定、持续低速区间、短段删除、近段合并、
  端点感知候选提取和无参考跨演示对齐；
- 增加轻量分析/可视化入口、数据/源码/配置指纹和两张图；
- 五条演示都产生五个自动片段和四个 5/5 支持边界簇；边界时间标准差平均 39 ms；
- 候选点离最近人工转移平均 211 ms，因为抓取/释放停留中心是事件，而人工状态标边缘；
- 支持粗粒度“接近抓取、搬运放置、撤离”，但速度无法单独恢复夹爪语义，也不是 TAPAS。

验证：

- `conda run -n env_isaaclab python -m pytest -q`：11 项通过
- 五条演示片段数 `[5, 5, 5, 5, 5]`，四个全支持簇
- 检查两张图的轨迹、阈值、人工转移、候选点和对齐
- compileall 与 `git diff --check` 通过

干净输出：`outputs/single_arm_scientific/segmentation_v1_clean`；实现提交
`7bfdfc4`，源码哈希
`30cded39e7941bf39070079771db321fcbb8effb311094fec530fa6b38d348c4`，
分析指纹
`867c512ca7a7ecee6a6905cd71303d9ed749534cd207a7afd7949c7d983dd3eb`。

记录提交：`0620b7b Record clean segmentation analysis fingerprint`。

### 阶段 4：双向在线关系估计

状态：完成。

- 保留旧 `KinematicConnectionDetector`，新增阶段无关四状态
  `OnlineRelationEstimator`；
- 加入实测指关节开度/速度、六维相对运动、窗口稳定性、物体/末端速度相关性、可选
  接触、非对称连接/丢失阈值、时间滞回和连续置信度；
- 全部阈值由五条冻结演示标定，测试 seed 不参与；
- 增加独立 `relation_dynamac` 标签、源码/配置指纹、逐步关系/置信度/夹爪轨迹、
  建立/释放/丢失延迟和两个新扰动；
- 冻结演示回放：平均建立偏移 -8 ms、释放延迟 60 ms，假阳性 0.01845、假阴性
  0.00681；
- 四个确定性机制测试覆盖空抓、成功搬运、闭合夹爪掉落和外部物体运动；
- seed 6200 仿真：六个普通条件全成功；建立延迟 120 ms、释放延迟 60 ms；掉落
  40 ms 撤销，空抓从不连接。两个反例任务因无重抓/重规划而失败。

验证：

- `conda run -n env_isaaclab python -m pytest -q`：16 项通过
- 五条完整回放和置信度/状态图人工检查
- 八个隔离 worker，JSON/NPZ 完整
- compileall 与 `git diff --check` 通过

干净复现：

- `relation_calibration_v1_clean` 源码哈希
  `23056a2b48bdca97620f545ba5c73a47e22545d62dc227e769093fcf44786a11`，
  分析指纹
  `d74669c3ece5682d3c4d76ff276899867a87774f1976cb81a2359c245ca195cb`；
- `relation_smoke_v1_clean` 有 8/8 对 JSON/NPZ、schema 4，共同源码哈希
  `66fd9063d7032306e1d0ba8c5187e6248b546a2fc749567b6103772f9f6454ca`。

实现提交：`1143f17 Prototype bidirectional online relation estimation`。
记录提交：`7362714 Record clean online relation evaluation`。

### 阶段 5：扩展单臂评测

状态：完成。

- 预留全新 seed 6300–6309，未用于冒烟或阈值标定；
- 方法：World、Static、SkillDynaMAC、Mask-only、旧在线 Full、双向 Relation；
- 条件：原六扰动加 `drop_after_grasp` 和 `close_without_grasp`；
- 每条试验增加精确目的阶段路径分区；schema 5 同时保留关系延迟、动作跳变、最大
  速度、推理时间、恢复和失败分类；
- 计划并执行 6 方法 × 8 条件 × 10 seed = 480 个隔离进程，结果不反向修改阈值。

提交 `3673dd2` 对应矩阵验收：

- 480/480 完整唯一试验、480 对 JSON/NPZ、十个各 48 条 seed 切片、统一源码/数据
  哈希、schema 5，阶段路径残差 ≤ `6.67e-16 m`；
- 六普通条件：World 0/60、Static 9/60、SkillDynaMAC 38/60、Mask 51/60、
  旧 Full 51/60、Relation 51/60；
- 所有方法在掉落和空抓任务上都失败。Relation 仍能在 40 ms 撤销全部掉落并拒绝
  全部空抓；旧在线方法要到 0.88–0.91 s 后才撤销，并在每个空抓中误连接；
- 普通条件平均路径：Mask 1.222 m、旧 Full 1.084 m、Relation 1.117 m；
  Relation 计算仍低于 1 ms，但为旧 Full 的 3.6 倍；
- [单臂最终报告](single_arm_final_report.md) 记录 Wilson 区间、seed 平衡 bootstrap、
  阶段路径、动作/速度/计算、失败分类和明确声明边界。

验证：17 项冻结前测试通过；480 个独立 worker 完成；瞬时 Isaac 退出警告未导致文件
或指标缺失；身份、计数、指纹、路径分区和哈希审计全部通过。

指标提交：`3673dd2 Prepare expanded single-arm evaluation metrics`。
报告提交：`e8df22c Run expanded single-arm DynaMAC evaluation`。

### 阶段 6：双臂交接骨架

状态：完成。

审计发现：已有任务具备两台 Franka、独立绝对 IK、独立夹爪、13 状态专家、隔离采集
和五条冻结 v1；但观测管理器只暴露关节状态和前一动作，v1 的单一 `carrier` 也不能
表示短时 `both`。因此 v1 保持不可变，修正模式写入独立 v2。

补充内容：

- 显式左右末端、物体、目标和实测指关节观测；
- 与状态对齐的 `none → left_only → both → right_only → none` 监督；
- 保留物理 `carrier` 字段以兼容旧策略；
- 仿真无依赖的模式单元测试和 v1 向后兼容测试；
- 一条隔离 headless 冒烟，再采集并独立审计五条 v2。

验收结果：

- 纯测试验证完整四值序列，并拒绝被破坏的 transfer 标签；v1 以
  `legacy_carrier_only` 正常加载；
- seed 7300 在 575 步完成，最终误差 10.62 mm；运行时确认六个必需观测形状和
  16 维动作；
- 正式 v2 在八次隔离尝试中接受 seed 7400、7403、7404、7406、7407；
- 五条轨迹均包含状态 0–12、全部四标签、恰好 15 个 `both` 步、0–40 mm 实测
  夹爪运动、连续 20 ms 时间戳，且无复位式跳变；
- 冻结哈希
  `91706df18abfea606c9e6836f1864e675610633ce5cb0c3c23846a1ea4f5fe18`，
  最大最终误差 11.04 mm，最小初始间距 13.91 mm；
- 采集和审计都拒绝覆盖冻结 v2，manifest 不变；论文、单臂 v1、交接 v1 未改；
- [双臂交接说明](bimanual_handover_setup.md) 记录契约、命令、审计和科学限制。

验证：

- `conda run -n env_isaaclab python -m pytest -q`：20 项通过
- 一条独立 headless 冒烟成功
- 八个正式隔离 worker：五条接受、三条由原成功门拒绝
- 冻结前后完整审计得到相同摘要哈希
- 覆盖/重复冻结反例均被拒绝，manifest 不变

实现提交：`5c4921e Prepare audited bimanual handover dataset v2`。
数据提交：`7489495 Add bimanual handover environment and scripted demonstrations`。

### 阶段 7：真实物理双臂交接

状态：v2 已通过最小科学验收，尚未通过物理数据采集门槛。

- 新任务使用重力、碰撞、摩擦和四个过滤式指体传感器，未使用物体位姿写入或几何 carrier；
- 物理关系由左右两条独立边产生，能够表示 `none → left_only → both → right_only → none`；
- 开发种子 7400–7404 为 5/5，随后在提交 `81c66f6` 和标签 `physical-handover-protocol-v1` 冻结正式协议；
- v1 的 20 个未见种子 7600–7619 唯一一次正式结果为 6/20，Wilson 95% 区间 `[14.5%, 51.9%]`；
- v1 六个成功样本全部完成完整关系转移，`both` 为 1.82–1.84 s，最终 XY 误差为 1.79–2.51 mm；
- v1 的 14 个失败全部为发送端 `left_pick_failed`：一指接近 0 N，另一指达到 75–103 N，物体没有离桌；
- v1 未达到预注册的 18/20 科学验收线和 20/20 数据采集门槛，因而没有创建 `handover_physical/v1`；
- v1 的冻结边界、失败视频和下一版允许动作见 [v1 正式报告](physical_handover_report.md)。
- v1 失败轨迹进一步证明旧 `LEFT_GRASP` 会追逐被指体推走的物体；提交 `a371f59` 只把抓取目标改为阶段入口一次冻结；
- v2 新开发种子 7700–7709 为 10/10，随后由提交 `828f6f1` 和标签 `physical-handover-protocol-v2` 冻结新协议；
- v2 在全新正式种子 7800–7819 上得到 `18/20`，Wilson 95% 区间 `[69.9%, 97.2%]`，恰好通过科学线；
- 18 个成功样本全部完成精确关系生命周期，最终 XY 误差 `0.67–4.37 mm`；条件于发送拾取成功，后续交接为 `18/18`；
- seed 7800 在桌面低力推走，seed 7813 抬升后因载荷不对称脱落，均在接收臂介入前失败；
- v2 仍未达到 `20/20` 数据门槛，因此继续不创建 `handover_physical/v1`，详细边界见 [v2 正式报告](physical_handover_report_v2.md)。
- v3 将发送端下降前和闭合前的 command 对齐门收紧至 15 mm；历史失败诊断 `2/2`、全新开发种子 7900–7909 为 `10/10`；
- v3 由提交 `71b4477` 和标签 `physical-handover-protocol-v3` 冻结，在正式种子 8000–8019 上首次运行得到 `20/20`；
- 20 个正式样本均为精确关系生命周期，`both` 为 `1.50–1.86 s`，最终 XY 误差 `0.85–11.38 mm`，末段全部稳定；
- v3 已通过 Phase 4 严格退出与物理数据采集门槛，完整统计和外推边界见 [v3 正式报告](physical_handover_report_v3.md)。
- 物理数据协议由提交 `d15a451` 和标签 `physical-handover-dataset-protocol-v1` 冻结，独立 seed 8200–8219 为 `20/20`；
- `data/handover_physical/v1` 已冻结，数据集 SHA-256 为 `a4a39ed4837558cecaaf73e7c5db9b6ff88e7eddfb3bcf9923df862df9e65e52`；
- 数据关系来自物理边而非 phase：20 条中共有 2007 个逐步标签差异，左右 connected 与双指接触最低一致率 99.55%/99.62%；
- 冻结数据的详细 schema、统计、覆盖拒绝和后续使用边界见 [物理数据冻结报告](physical_handover_dataset_report.md)。

### 阶段 8：双臂在线关系图

状态：离线与在线扰动开发验证完成，正式泛化评测未完成。

- 新增 `BimanualOnlineRelationEstimator`，左右臂—物体关系各自维护滞回状态，组合成四值关系图；
- 估计器每步只读左右末端、物体、实测指体间距及其速度，不读取接触真值、脚本 phase 或未来状态；
- 冻结数据 seed 8200–8209 用于标定，互斥的 8210–8219 用于开发回放；该划分不是正式预注册 cohort；
- 初始十条回放的四值准确率为 94.33%，接收边存在建立延迟；占用平台与一致运动学解除修复后为 98.30%，左/右 micro F1 为 98.54%/99.44%；
- 五类反事实单元测试验证两条边不会因另一臂空抓、释放、丢失、长时双持或暂停而被 phase 计时器错误改写；
- 新增七条件真实物理在线评测；最终开发 seed 8302 的所有干预均由物理真值确认成立，推断序列全部逐段正确；
- [双臂在线关系说明](bimanual_relation_estimator.md)记录开发反例、完整指标、证据哈希和下一阶段正式门槛；当前不进入完整双臂策略训练。

### 最终交付

状态：完成。

- [最终报告](overnight_final_report.md) 汇总阶段结果、提交与复现来源、已解决/未解决
  问题、声明边界、自研方向和次日三项检查；
- 最终验收重跑纯测试、语法编译、冻结数据审计、480 条身份审计、受保护资产哈希和
  干净工作树检查；
- 20 项测试通过，480 条实验与三个冻结数据集均从磁盘复核一致；
- 最终汇总提交：`ab1d4ef Document overnight DynaMAC research outcomes`。
