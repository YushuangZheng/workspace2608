# 文档更新记录

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
