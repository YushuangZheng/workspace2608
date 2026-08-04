# 文档更新记录

## 2026-08-04：双臂在线关系估计 Phase 5 离线开发

- 新增两条独立滞回关系边，并组合为 `none / left_only / both / right_only`；运行时接口不接收接触真值、任务 phase 或未来状态；
- 用冻结物理数据的 seed `8200–8209` 标定、seed `8210–8219` 做互斥开发回放，未将该划分表述为预注册正式评测；
- 十条回放四值准确率均值为 94.33%，左右 micro F1 为 97.82%/95.41%，假阳性为 0/2；
- 真值 901 个 `both` 步只推断 568 个，明确记录接收边偏保守和交接区间漏检问题；
- 五类纯状态机反例覆盖接收空抓/延迟、发送提前释放、接收后丢失、延长双持和单臂暂停；
- 标定配置支持验证后反序列化，并拒绝只设置建立或解除距离阈值的非法组合；
- 新增 `docs/bimanual_relation_estimator.md`，记录输入边界、数据划分、完整指标、证据哈希和进入策略训练前的在线门槛。

## 2026-08-04：冻结真实物理双臂交接数据集 v1

- 新增精确 seed 批次采集器和独立物理数据审计，不复用 v3 正式评测轨迹，不允许失败替补或成功子集筛选；
- 提交 `d15a451` 和标签 `physical-handover-dataset-protocol-v1` 冻结 seed `8200–8219`、schema 与验收门；
- 20 个 seed 全部一次成功并通过预审计，冻结为 `data/handover_physical/v1`；
- 数据集 SHA-256 为 `a4a39ed4837558cecaaf73e7c5db9b6ff88e7eddfb3bcf9923df862df9e65e52`；
- 20 条数据均为精确关系生命周期，`both` 为 `1.46–1.86 s`，最终 XY 误差 `1.10–9.43 mm`；
- 左右关系与双指接触最低一致率为 99.55%/99.62%，物理关系与 phase 映射共有 2007 个逐步差异；
- 冻结后的重复采集和重复冻结均被拒绝，代表性哈希未变化；
- 新增 `docs/physical_handover_dataset_report.md`，明确 privileged 监督和单任务脚本数据的使用边界。

## 2026-08-04：真实物理双臂交接 Phase 4 v3

- v2 的两个正式失败进一步定位为下降前横向误差过大；v3 要求发送端在净空高度进入 command 的 15 mm 范围后才下降，并在闭合前再次验收；
- 历史失败诊断为 `2/2`，全新开发种子 `7900–7909` 为 `10/10`，均不计入正式成功率；
- 提交 `71b4477` 和标签 `physical-handover-protocol-v3` 冻结种子 `8000–8019`、源码指纹与 `20/20` 严格退出门槛；
- 唯一一次正式结果为 `20/20 = 100%`，Wilson 95% 区间 `[83.9%, 100%]`；
- 20 个样本全部形成精确关系生命周期，`both` 为 `1.50–1.86 s`，最终 XY 误差为 `0.85–11.38 mm`，末段全部稳定；
- v3 达到 `handover_physical/v1` 采集和 Phase 5 启动门槛，但正式报告明确限制该结论只适用于冻结扰动分布；
- 只读审计入口扩展为同时支持 v2 与 v3，不修改任一冻结结果。

## 2026-08-04：真实物理双臂交接 Phase 4 v2

- v1 的 14 个正式失败用于诊断，确认 `LEFT_GRASP` 持续追逐被接触推动后的物体，形成正反馈；
- 提交 `a371f59` 将发送端抓取起点、目标和指令在进入闭合阶段时一次冻结，其余物理与成功标准不变；
- v1 诊断样本为 `5/5`，全新开发种子 `7700–7709` 为 `10/10`，均不计入正式成功率；
- 提交 `828f6f1` 和标签 `physical-handover-protocol-v2` 预注册种子 `7800–7819` 与双门槛；
- 唯一一次 v2 正式结果为 `18/20 = 90.0%`，Wilson 95% 区间 `[69.9%, 97.2%]`，达到 `18/20` 科学门槛；
- 18 个成功样本全部得到精确 `none → left_only → both → right_only → none`，`both` 为 `1.82–1.84 s`，最终 XY 误差为 `0.67–4.37 mm`；
- seed 7800 为桌面低力推走，seed 7813 为抬升后脱落；二者均在接收臂介入前失败，进入交接阶段的样本为 `18/18` 完成；
- v2 未达到 `20/20` 数据门槛，故仍不创建 `handover_physical/v1`，不事后筛选成功 rollout 作为训练数据；
- 新增 `docs/physical_handover_report_v2.md`，保留 v1 报告，不覆盖两次冻结 cohort；
- 新增只读审计脚本，硬检查冻结成员、指纹、JSON/NPZ 配对、trace 对齐、关系生命周期和聚合计数。

## 2026-08-04：真实物理双臂交接 Phase 4

- 新增 `Essay2608-Bimanual-Physical-Handover-v0`，启用物体重力、真实碰撞、摩擦、双指夹持和四指过滤式接触传感；
- 脚本专家不读取 privileged 接触或关系标签，评测不直接写入物体位姿/速度，也不使用几何 carrier；
- 新增独立双边物理关系真值，连接建立需要双指接触、近邻和相对运动一致，连接保持不会被仍有双指接触的短时沉降错误解除；
- 开发种子 7400–7404 为 5/5，冻结实现提交 `040ad95`；
- 预注册提交 `81c66f6` 和标签 `physical-handover-protocol-v1` 固定 20 个正式种子、关系序列、成功阈值和禁止 test 调参规则；
- 唯一一次正式结果为 6/20，低于 18/20 验收线；六个成功样本均形成 1.82–1.84 s 的真实 `both` 并以 1.79–2.51 mm XY 误差完成放置；
- 14 个失败全部为 `left_pick_failed`，逐指轨迹与 seed 7602 视频均显示发送端单指侧压，未发现正式集接收或放置阶段失败；
- 新增 `docs/physical_handover_protocol.md` 与 `docs/physical_handover_report.md`，明确不创建 `handover_physical/v1`、不进入完整双臂策略训练。

## 2026-08-04：RelationDynaMAC 恢复研究 Phase 3 正式结果

- 完成预注册的 4 方法 × 7 条件 × 20 held-out seeds，共 560 次隔离试验；
- 新增 `docs/recovery_final_report.md`，报告逐条件原始计数、配对主比较、检测延迟、恢复时间、额外路径、安全限幅、误触发和失败分类；
- 六个扰动条件中，Relation 无恢复为 23/120，Relation＋Recovery 为 105/120；三类 drop 的两种恢复方法均为 60/60；
- 正式保留 `miss_small_shift` 成功率不足、正常条件 5/20 短暂误触发和两种恢复方法各 2 次大位移恢复失败，不使用 test seeds 后调参；
- 新增 `scripts/audit_recovery_results.py`，硬检查 560 个唯一组合与指纹、源码和数据哈希、预注册参数覆盖、JSON/NPZ 配对、trace schema、逐 step 对齐及终端快照一致性；
- 最终 summary SHA-256 为 `2f58eef7a7493c2871854fa0dc9a7c060f43fb3ee7e14f7e49ab8d0acd69914f`。

## 2026-08-04：RelationDynaMAC 恢复研究 Phase 3 预注册

- 新增七个恢复协议条件，drop 覆盖三个任务时点、四个距离、四个方向和两种夹爪行为，miss 覆盖小位移、大位移与边缘抓持；
- 固定 calibration、development 和 20 个 held-out test seeds，后者在协议冻结前未运行；
- 新增 `configs/experiments/recovery_protocol_v1.json` 与 `docs/recovery_protocol.md`，固定方法、阈值、最大重抓、指标、失败分类、560-trial 命令和禁止 test 调参规则；
- 聚合新增相对同 method/seed 正常条件的额外路径统计；所有具体 seed 位移进入实验 fingerprint。

## 2026-08-04：RelationDynaMAC 恢复研究 Phase 2

- 新增 `OracleRelationRecoveryPolicy`，Oracle 只给恢复图提供当前步 privileged relation，不提供动作或未来信息；
- 当前 Isaac Sim 4.5 的过滤式指尖 ContactSensor 会在 articulation 初始化时关闭应用，因此撤销该配置，保留干净的原任务环境；
- Oracle 明确定义为已知 Franka—方块几何下的夹爪占用开度与末端—物体距离联合谓词，并在 `docs/oracle_recovery_ablation.md` 中记录适用边界；
- seed 6400 的 static、drop 和 miss 均成功；Oracle drop 立即发现关系断开，在线估计器延迟 40 ms，两者恢复图均完成任务。
- 三个 development seeds 的统一 36-trial 消融中，两个无恢复方法在 drop/miss 均为 0/3，在线恢复与 Oracle 恢复均为 3/3；两种恢复方法的 static 误触发均为 0/3。

## 2026-08-04：RelationDynaMAC 恢复研究 Phase 1

- 新增独立 `RelationRecoveryController` 与 `RelationDynaMACRecoveryPolicy`，恢复层覆盖动作时暂停任务 phase clock；
- 实现 MISS、LOSS、撤离、重定位、重新接近、重抓、验证、恢复和有界失败完整状态；
- 新 schema 保存 active frames、恢复状态、触发来源、重抓次数和动作后终端快照；
- 新增 `docs/recovery_graph.md`，记录状态图、安全限制、冻结训练示范标定、反例修复和 development 烟测；
- development seeds 6400–6402 的 9 个 trial 全部成功：static 误触发 0/3，drop 与 miss 均恢复 3/3；该结果不替代后续预注册 test seeds。

## 2026-08-04：RelationDynaMAC 恢复研究 Phase 0

- 新增 `docs/trace_visual_audit.md`，记录十个固定代表性 trial 的轨迹重建视频、失败分类复核和证据边界；
- 视频与 manifest 写入新的 `outputs/recovery_scientific/trace_audit_v1`，冻结单臂结果保持只读；
- 审计发现冻结 v1 在环境终止 trial 中缺少动作后的终端观测。新评测 schema 已把终端快照与 action-aligned 序列分开持久化，避免添加没有对应动作的伪 step；
- 十个样本的 failure taxonomy 语义均一致；Relation 空抓旧 NPZ 的最后观测与 JSON 终端误差不对齐，已在文档和 manifest 中显式保留，未回写冻结结果；
- 新增渲染、active-frame 重建、终端对齐和失败语义回归测试。全部文档继续使用中文，代码字段保持原始英文标识。

## 2026-08-04：全部用户可见研究记录统一为中文

### 更新目的

将仓库中面向研究协作、实验复现和论文撰写的 Markdown 记录统一改为中文，减少中英文
混排造成的理解偏差。同时保证方法名、代码标识、数据字段、命令、数学公式、提交号和
哈希值不被翻译或改写，以维持复现能力。

### 本次更新范围

| 文件 | 更新内容 |
|---|---|
| `README.md` | 重写为中文项目入口，补充当前研究状态、关键报告、环境标识、验收命令和声明边界 |
| `docs/bimanual_handover_setup.md` | 双臂交接环境、观测、关系监督、v2 数据审计与限制全部中文化 |
| `docs/bimanual_minimal_loop.md` | 双臂交接与托盘工程试验、单 seed 结果和声明边界中文化 |
| `docs/method_provenance.md` | 论文方法、项目简化和自研机制的逐项来源映射中文化 |
| `docs/online_relation_estimator.md` | 四状态关系估计、标定、反例和仿真结果中文化 |
| `docs/overnight_final_report.md` | 六阶段最终结果、提交、复现、未解问题和次日检查中文化 |
| `docs/overnight_progress.md` | 全部阶段过程、验证、哈希和提交记录中文化 |
| `docs/segmentation_analysis.md` | 速度分段流程、结果、解释及 TAPAS 边界中文化 |
| `docs/single_arm_final_report.md` | 480 次单臂评测、统计、关系反例、路径和失败分类中文化 |
| `docs/single_arm_minimal_loop.md` | 单臂冻结数据、三 seed 试验和扩散基线中文化 |
| `docs/single_arm_scientific_audit.md` | 成功标准、59 mm 高度偏差和逐阶段归因中文化 |
| `docs/web_review_audit.md` | 外部 GPT 点评的确认、修复和未复现问题中文化 |
| `essay2608_conversation_summary.md` | 保留原中文内容，修复转义损坏的 LaTeX 命令与不可见控制字符 |

### 保持不变的技术事实

- 不修改 `DynaMAC.pdf`；
- 不修改任何冻结 NPZ、manifest 或 `FROZEN`；
- 单臂冻结数据哈希仍为
  `8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8`；
- 双臂交接 v2 哈希仍为
  `91706df18abfea606c9e6836f1864e675610633ce5cb0c3c23846a1ea4f5fe18`；
- 480 次扩展评测的数字、方法名、条件名、seed、schema 和复现命令保持不变；
- 原有英文文件名保持不变，避免破坏仓库链接和外部引用。

### 翻译约定

- 类名、函数名、脚本名、路径、JSON 字段和方法标签使用反引号保留原文；
- World Gaussian、SkillDynaMAC、RelationDynaMAC、TAPAS、MiDiGaP 等专名保留；
- success、recovery、rollout、seed、schema 等在首次出现时结合中文语境解释，必要时
  保留原词，避免与代码字段脱节；
- 所有科学限制继续保留，不因中文化而扩大结论。

### 检查记录

文档提交前执行：

```bash
git diff --check
conda run -n env_isaaclab python -m pytest -q
conda run -n env_isaaclab python -m compileall -q \
  source/essay2608/essay2608 scripts tests
```

另行检查：

- Markdown 相对链接目标存在；
- 用户可见文档一级标题均为中文；
- Markdown 中不存在异常 ASCII 控制字符；
- 受保护 PDF 与冻结数据哈希未变化；
- 文档中的关键提交号、实验数量、seed 和哈希与实际产物一致。

### GitHub 记录

本次更新已按依赖顺序合并到默认分支：

| 项目 | GitHub 记录 | 合并提交 | 状态 |
|---|---|---|---|
| 分阶段 DynaMAC 研究基础 | [PR #1](https://github.com/YushuangZheng/workspace2608/pull/1) | `20ffbb6e8e08d45a634f2b702833b3a526358c69` | 已合并 |
| 科学审计、扩展评测、双臂 v2 与中文文档 | [PR #2](https://github.com/YushuangZheng/workspace2608/pull/2) | `dc3071b2693ebb31133b05ae29cd0feecf3d816c` | 已合并 |

中文统一提交为 `323dd2fe15c3df58ea2d82c8ccc422f4bf3902a0`。合并 PR #2 后，
GitHub 开放 PR 数为 0；默认分支为 `master`。本段实际合并来源通过一个仅含文档记录的
后续 PR 补入，避免在事件发生前预写 PR 编号或合并哈希。
