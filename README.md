# Essay2608：Isaac Lab 中的动态多机械臂协作

## 项目概览

本仓库在自定义 Isaac Lab 任务中分阶段研究 DynaMAC。当前包含：

- 已完成科学审计的单臂动态抓取与放置闭环；
- 双臂交接（Handover）环境、专家和冻结数据；
- 双臂托盘搬运（Lift Tray）工程原型；
- 一个低维条件扩散基线。

所有正式验收的演示数据集均带版本号、逐文件 SHA-256 和冻结标记。
当前结论、限制与复现入口见：

- [夜间研发最终报告](docs/overnight_final_report.md)
- [单臂代表性轨迹可视化审计](docs/trace_visual_audit.md)
- [RelationDynaMAC 单臂恢复图](docs/recovery_graph.md)
- [Oracle relation 恢复消融](docs/oracle_recovery_ablation.md)
- [单臂关系恢复实验预注册协议](docs/recovery_protocol.md)
- [单臂关系触发式恢复正式报告](docs/recovery_final_report.md)
- [真实物理双臂交接预注册协议](docs/physical_handover_protocol.md)
- [真实物理双臂交接正式报告](docs/physical_handover_report.md)
- [单臂最终评测报告](docs/single_arm_final_report.md)
- [双臂交接环境与数据说明](docs/bimanual_handover_setup.md)
- [方法来源与实现边界](docs/method_provenance.md)
- [文档更新记录](docs/documentation_changelog.md)

## 主要能力

- 与 Isaac Lab 主仓库隔离，项目代码和数据可独立维护；
- 两台 Franka 的独立绝对位姿 IK 与夹爪控制；
- 冻结数据、内容寻址实验缓存和可复现实验指纹；
- 世界、物体、目标及虚拟末端参考系下的高斯策略；
- 运行时参考系屏蔽、双向关系估计和机制反例；
- 逐次试验 JSON/NPZ 证据、阶段路径、动作跳变和失败分类。

## 安装

1. 按照 [Isaac Lab 安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   安装 Isaac Sim 与 Isaac Lab。建议使用 conda 或 uv 环境。
2. 将本仓库放在 Isaac Lab 主目录之外。
3. 使用带 Isaac Lab 的 Python 解释器，以可编辑模式安装扩展：

```bash
python -m pip install -e source/essay2608
```

本项目验证时使用的环境为：

```text
/home/zys/miniconda3/envs/env_isaaclab/bin/python
```

也可统一使用：

```bash
conda run -n env_isaaclab python <脚本>
```

## 环境注册检查

列出任务：

```bash
python scripts/list_envs.py
```

关键任务标识：

```text
Essay2608-Dynamic-Pick-Place-v0
Essay2608-Bimanual-Handover-v0
Essay2608-Bimanual-Physical-Handover-v0
Essay2608-Bimanual-Lift-Tray-v0
```

使用零动作或随机动作检查环境：

```bash
python scripts/zero_agent.py --task=<TASK_NAME>
python scripts/random_agent.py --task=<TASK_NAME>
```

## 核心复现命令

验证单臂冻结数据：

```bash
conda run -n env_isaaclab python scripts/audit_dataset.py \
  --data_dir data/pick_place_static/v1
```

运行单臂完整评测：

```bash
conda run -n env_isaaclab python scripts/eval_single_arm.py --headless \
  --methods world_gaussian static_multistream skill_dynamac mask_only \
  full_dynamac relation_dynamac \
  --conditions static smooth_object sudden_object smooth_target sudden_target \
  arm_offset drop_after_grasp close_without_grasp \
  --seeds 6300 6301 6302 6303 6304 6305 6306 6307 6308 6309 \
  --output_dir outputs/single_arm_scientific/v1
```

审计预注册的单臂关系恢复正式结果：

```bash
conda run -n env_isaaclab python scripts/audit_recovery_results.py
```

复现真实物理双臂交接开发样本：

```bash
conda run -n env_isaaclab python scripts/eval_physical_handover.py \
  --headless --seeds 7400 --max_steps 1400 \
  --output_dir outputs/physical_handover/dev_reproduction
```

验证双臂交接 v2 冻结数据：

```bash
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir data/handover_static/v2
```

运行纯单元测试：

```bash
conda run -n env_isaaclab python -m pytest -q
```

## IDE 配置（可选）

在 VS Code 中按 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，运行
`setup_python_env`，并按提示填写 Isaac Sim 的绝对路径。任务会生成
`.vscode/.python.env`，帮助 Pylance 索引 Isaac Sim 和 Omniverse 模块。

若 Pylance 未能索引扩展，可在 `.vscode/settings.json` 的
`python.analysis.extraPaths` 中添加：

```json
{
  "python.analysis.extraPaths": [
    "<path-to-ext-repo>/source/essay2608"
  ]
}
```

若索引内容过多导致 Pylance 崩溃，应从 `extraPaths` 中排除未使用的
`omni.anim.*`、`omni.kit.*`、`omni.graph.*` 等扩展，而不是扩大系统交换区作为首选方案。

## 作为 Omniverse 扩展加载（可选）

1. 打开 `Window` → `Extensions`。
2. 在扩展管理器设置中加入本仓库的 `source` 目录。
3. 同时加入 Isaac Lab 的 `IsaacLab/source` 目录。
4. 刷新后，在 `Third Party` 分类中启用 `essay2608`。

示例界面扩展位于：
`source/essay2608/essay2608/ui_extension_example.py`。

## 代码格式与检查

仓库提供 pre-commit 配置：

```bash
pip install pre-commit
pre-commit run --all-files
```

最低验收命令：

```bash
conda run -n env_isaaclab python -m pytest -q
conda run -n env_isaaclab python -m compileall -q \
  source/essay2608/essay2608 scripts tests
git diff --check
```

## 科学声明边界

本仓库复现的是自定义 Isaac Lab 任务中的相对几何、动态参考系和关系生命周期机制，
不是 TAPAS、MiDiGaP、RLBench、DynaBench 或论文完整黎曼策略的逐项复现。
旧的 `handover_static/v1/v2` 仍使用几何附着，只能作为接口和数据骨架。新增物理任务
已经产生真实接触关系转移，但预注册正式集仅为 `6/20`，不能声称脚本专家稳定，也未
采集 `handover_physical/v1`。所有论文表述应同时遵守[夜间研发最终报告](docs/overnight_final_report.md)
和[真实物理双臂交接正式报告](docs/physical_handover_report.md)中的声明边界。
