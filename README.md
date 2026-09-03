# Relational-Progress Closed-Loop Manipulation

本仓库包含两部分相互区分的实现：冻结的 DynaMAC V4 复现，以及建立在该动作策略之上的关系—进度闭环技术路线。核心算法保持 benchmark 无关；RLBench 只负责观测、控制器、数据和评测协议适配。

当前状态：

- DynaMAC V4 baseline 已冻结，标签为 `dynamac-v4.0`；
- 技术路线阶段一至五的环境无关模块与阶段六 RLBench 集成已经完成；
- 阶段六正式矩阵包含正常、平滑动态无故障和四类故障，共 192 个单元、14,400 回合；
- 当前唯一正式统计入口为 `evaluations/development/phase6_formal_evaluation/results/v2/`，深度审计已通过。

## 项目结构

| 路径 | 内容 |
|---|---|
| `source/policy/dynamac.py` | 冻结 DynaMAC 多流动作策略接口 |
| `source/policy/closed_loop/` | 阶段一至五及阶段六顶层策略的环境无关核心算法 |
| `source/data/` | 通用示范数据结构与校验 |
| `configs/` | 闭环任务模型、信念更新、执行、边界与恢复配置 |
| `integrations/rlbench/rlbench_dynamac/` | DynaMAC 的 RLBench 数据、控制器、评测和报告适配 |
| `integrations/rlbench/rlbench_closed_loop/` | 闭环策略的 RLBench 观测、联合快照与进程协议适配 |
| `integrations/rlbench/configs/` | RLBench 任务、分段、运动源和干预协议 |
| `integrations/rlbench/data/` | 五条正常示范与封存评测集；大文件不进入 Git |
| `integrations/rlbench/models/` | V4 baseline、阶段六动作模型与闭环 sidecar；权重不进入 Git |
| `integrations/rlbench/results/` | V4 baseline 与阶段六正式原始结果；大文件不进入 Git |
| `evaluations/` | 评测总入口；`development/` 归档技术路线开发期验收，后续论文实验直接按实验目的建目录 |
| `tests/closed_loop/` | 闭环核心模块和跨阶段回归测试 |
| `integrations/rlbench/tests/` | RLBench 适配、执行器和评测协议测试 |
| `新方法代码开发计划.md` | 当前实现口径和阶段开发计划 |
| `阶段开发记录.md` | 各阶段最终保留的实现，不记录已淘汰方案 |
| `开发日志.md` | 开发、诊断和修复过程记录 |
| `技术路线报告.md` | 理论动机、模型定义和整体技术路线 |
| `iclr2027/` | 论文正文、实验设计、提纲与参考文献；当前作为本地写作目录管理 |
| `release-artifacts/` | 已发布版本的本地打包副本，不参与运行时加载 |

## 当前正式资产

### DynaMAC V4 baseline

- 正常示范：`integrations/rlbench/data/training/`
- 封存评测集：`integrations/rlbench/data/evaluation/`
- 模型：`integrations/rlbench/models/v4/`
- 结果与回放：`integrations/rlbench/results/v4/`
- 协议：[integrations/rlbench/V4_PROTOCOL.md](integrations/rlbench/V4_PROTOCOL.md)
- RLBench 使用说明：[integrations/rlbench/README.md](integrations/rlbench/README.md)

### 闭环技术路线

- 阶段六动作模型：`integrations/rlbench/models/phase6_v1/`
- 闭环任务模型：`integrations/rlbench/models/closed_loop_phase6_v1/`
- 正式协议：[evaluations/development/phase6_formal_evaluation/PROTOCOL.md](evaluations/development/phase6_formal_evaluation/PROTOCOL.md)
- 最终统计：`evaluations/development/phase6_formal_evaluation/results/v2/`
- 正式原始结果：`integrations/rlbench/results/phase6_formal_v1/`
- 正常矩阵专项审计：[evaluations/development/phase6_formal_evaluation/NORMAL_AUDIT.md](evaluations/development/phase6_formal_evaluation/NORMAL_AUDIT.md)

阶段一至五的最终组件验收分别保存在 `evaluations/development/phase23_component_ab/`、`evaluations/development/phase4_*`、`evaluations/development/peer_execution_dependency/` 和 `evaluations/development/phase5_*`。每个目录只保留当前正式结果版本。

## 核心与集成边界

`source/policy/closed_loop/` 不依赖 RLBench。它实现任务模型、关系与进度信念、动态流角色、边界事务、主动关系验证、恢复和合法重入。`integrations/rlbench/` 将这些接口连接到 RLBench 低维观测、CoppeliaSim 和共享执行器。迁移到其他 benchmark 时，应优先新增对应集成层，而不是在核心算法中加入任务名或 benchmark 专属分支。

## 环境与验证

核心包要求 Python 3.10 或更新版本：

```bash
python -m pip install -e '.[test,midigap]' ruff
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider tests/closed_loop
ruff check --no-cache source tests evaluations integrations
```

RLBench、PyRep、TAPAS 和 CoppeliaSim 为本地第三方依赖，不纳入主仓库版本控制。固定修订、补丁和 Python 3.8/3.10 环境要求见 [integrations/rlbench/THIRD_PARTY.md](integrations/rlbench/THIRD_PARTY.md) 与 [integrations/rlbench/README.md](integrations/rlbench/README.md)。

验证阶段六最终统计文件：

```bash
cd evaluations/development/phase6_formal_evaluation/results/v2
sha256sum -c SHA256SUMS
```

训练数据、模型权重、逐回合原始结果、视频、第三方源码和论文副本均作为本地或 Release 资产管理；Git 只保留实现、协议、紧凑清单、测试和最终汇总。
