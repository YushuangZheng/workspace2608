# DynaMAC / MiDiGaP / DP 仿真实验

本仓库以 RoboDojo 为统一仿真任务库，独立复现 DynaMAC、MiDiGaP 和真值状态 Diffusion
Policy。RoboDojo 作为只读 Git 子模块接入；大体积资产、专家演示和原始视频按任务缓存，
不复制上游任务代码，也不提交到 Git。

## 只需要看的文件

1. `source/policy/dynamac.py`：DynaMAC Algorithm 1 与双臂并发策略；
2. `source/policy/midigap.py`：MiDiGaP、约束更新和 VAPOR；
3. `source/policy/diffusion_policy.py`：state U-Net DP 独立复现；
4. `source/data/robodojo.py`：RoboDojo 任务、资产、运行层和论文表协议；
5. `scripts/run.py`：唯一命令行入口；
6. `logs/research_log.md`：中文研究记录与结论边界。

`source/` 物理结构保持一层，不再保留临时场景脚本：

```text
source/
├── __init__.py
├── data/
└── policy/
```

## RoboDojo 接入结构

```text
third_party/RoboDojo/              # 唯一 RoboDojo 根目录：官方代码、Assets、选择性 data
third_party/RoboDojo/Assets/
third_party/RoboDojo/data/RoboDojo/<task>/arx_x5/data/
third_party/RoboDojo/.cache/essay2608/ # 下载清单与 HF 根文件，不进 Git
.runtime/robodojo/          # 可重建运行层，不进 Git
results/robodojo/captures/  # 项目 GUI 回放补采 JSONL，不进 Git
results/robodojo/source_layouts/ # 项目源布局估计，不进 Git
results/robodojo/raw/       # GUI 逐回合 JSON/视频，不进 Git
results/robodojo/           # 可审计 CSV 与论文 Markdown 表
```

运行层把唯一 RoboDojo 根目录中的 `env/task/src/utils/scripts/XPolicyLab`、`Assets` 和原始结果目录组合起来。项目在
运行层生成可组合的 `essay2608_single_{x5,franka}_{left,right}` 和
`essay2608_dual_{x5,franka}` 配置；上游 `arx_x5` 等原生配置仍可直接使用。新组合复用
官方物体布局，并在生成配置时记录布局来源，不修改 RoboDojo 子模块。

官方边界固定如下：`third_party/RoboDojo` 保持上游锁定提交且工作树只读；任务逻辑、奖励函数、
官方 `arx_x5` 配置、资产目录和评测结果判据均不在项目内改写。项目适配只发生在可丢弃的
`.runtime/robodojo` 和 `source/data/robodojo_gui.py`：增加项目策略 WebSocket、任务位姿注入、
近景视口以及单臂字段兼容。要复现官方流程，仍以 RoboDojo 上游的 `scripts/robodojo.sh`、
`env_cfg/arx_x5.yml` 和官方布局为准；`capture/capture-batch` 是项目额外的训练数据诊断工具，
不是对上游评测实现的替换。

RoboDojo 上游清单中的 54 个可运行配置全部作为项目任务候选池，可用
`python scripts/run.py robodojo resources` 同时查看任务、场景、机器人、环境配置和
XPolicyLab policy。当前论文正式预注册子集为：

| 机械臂 | RoboDojo 任务 | DynaMAC 论文近邻 | 选择理由 |
| --- | --- | --- | --- |
| 单臂 | `push_T` | SweepDust | 单活动臂平面接触推动与精确目标对齐 |
| 单臂 | `pour_liquid_into_cup` | StoreBottle | 单活动臂瓶—容器相对位姿与倾倒约束 |
| 双臂 | `sweep_blocks` | HandOver / SweepDust | 任务定义包含跨手交接和协同清扫 |

官方 Hugging Face 数据集的根文件和 `Assets/` 直接归档在唯一的 `third_party/RoboDojo/` 下，
同步时明确排除官方 `data/` 与 `ckpt/` 大目录；专家数据再按任务/条数选择性放入
`third_party/RoboDojo/data/RoboDojo/`。机器人资产库下载 RoboDojo 当前提供的 X5 与 Franka。默认只下载论文子集的物体资产；
需要把全部 54 个任务作为候选时，使用 `--all`，下载过程可恢复但可能占用大量空间。

## 环境与运行

RoboDojo 当前要求 Python 3.11、Isaac Sim 5.1 和 Isaac Lab 2.3。本项目使用独立的
`RoboDojo` Conda 环境，不改原有 `env_isaaclab`（Isaac Sim 4.5）。

```bash
conda activate RoboDojo

# 检查接入状态、生成运行层
python scripts/run.py robodojo status
python scripts/run.py robodojo prepare

# 同步官方 HF main 的根文件与全部 Assets；官方仓库实际目录名是 ckpt，
# 此命令明确排除 data/ 和 ckpt/，不会下载 checkpoint。
python scripts/run.py robodojo official-sync

# 官方下载物集中在唯一的 third_party/RoboDojo；这里下载论文子集的官方资产与前五条专家演示
python scripts/run.py robodojo assets
python scripts/run.py robodojo demos --episodes 5
# 下载全部 54 个任务的资产/演示（按需执行）
python scripts/run.py robodojo assets --all
python scripts/run.py robodojo demos --all --episodes 5

# 真实位姿训练数据不是手工示教：官方 HDF5 已包含专家动作，下面命令会让 Isaac Sim
# GUI 按官方 ee 动作接口逐条回放，并同步写出每一帧的任务位姿 JSONL。push_T 的源布局也会
# 尝试从 HDF5 RGB 重建；源布局与演示无法一一对应时会严格失败，绝不伪造训练数据。
python scripts/run.py robodojo capture-batch \
  --task push_T --episodes 5 --auto-reconstruct \
  --output-root results/robodojo/captures \
  --source-layout-root results/robodojo/source_layouts \
  --replay-action ee_pose

# 如需 RGB-D 轨道，可使用内置 DINOv2/SAM 候选前端（首次运行会下载模型权重），
# 或替换为自己的 module:function；默认仍是 Oracle Pose。
export ESSAY2608_RGBD_POSE_ESTIMATOR='builtin:dino_sam'
python scripts/run.py robodojo capture-batch \
  --task push_T --episodes 5 --observation-mode rgbd_pose

# 对已补采的一个任务使用前五条演示训练真实位姿 DP；--episodes 可扩展到 100。
python scripts/run.py robodojo fit \
  --policy dp --task push_T --episodes 5 \
  --capture-root results/robodojo/captures \
  --output results/robodojo/checkpoints/push_T/dp_real_pose.npz

# 选择任务、场景、机器人和 Oracle Pose 做 GUI 评测；命令不提供 headless 选项
python scripts/run.py robodojo eval \
  --task push_T --policy dynamac \
  --checkpoint results/robodojo/checkpoints/push_T/dynamac.npz \
  --scene default --robot essay2608_single_franka_right \
  --observation-mode oracle_pose --seed 0 --episodes 1

# 轨道 B：RGB-D Pose。先提供 module:function，估计器不得回退到真值
export ESSAY2608_RGBD_POSE_ESTIMATOR='my_pose_module:estimate'
python scripts/run.py robodojo eval \
  --task push_T --policy dynamac \
  --checkpoint results/robodojo/checkpoints/push_T/dynamac.npz \
  --robot essay2608_single_franka_right \
  --observation-mode rgbd_pose --seed 0 --episodes 1

# 接入任意 RoboDojo/XPolicyLab policy：先按上游 deploy.yml 启动 ws 服务，
# 再由项目 GUI 客户端连接；服务端不复制进本项目 policy 目录
python scripts/run.py robodojo external-eval \
  --policy-name ACT --policy-server-url ws://127.0.0.1:PORT \
  --task push_T --scene default --robot arx_x5 --episodes 1

# 从 RoboDojo 原始结果生成论文级逐次 CSV 和三种子汇总表
python scripts/run.py robodojo table
```

所有正式评测固定 GUI、单环境顺序执行。静态主表按每方法 3 个种子、每种子 50 回合，
报告种子均值±样本标准差与合并 Wilson 95% CI；`smooth` 和 `teleport` 是论文扩展动态条件，
必须单表呈现。GUI 运行会保存逐回合视频，便于人工复核失败类型。

## 算法与数据边界

### 任务/场景/机器人/policy 的开放接口

`resources` 是唯一资源注册入口：任务来自上游 54 项可运行清单，场景来自
`env_cfg/scene/*.yml`，机器人来自 `env_cfg/robot/*.yml`，policy 来自
`XPolicyLab/policy/*/deploy.yml`。`prepare` 会把这些目录挂入 `.runtime/robodojo`，并
生成环境组合配置；因此后续新增任务、换场景或换机械臂只需在评测命令改
`--task/--scene/--robot`，不再复制脚本。项目内置 `dp/midigap/dynamac` 走同一 GUI
策略服务器；其它上游 policy 走 `external-eval`，其 WebSocket 消息协议由对应的
`deploy.yml` 负责。

### 两条观测轨道

- `oracle_pose`：从 RoboDojo layout manager 读取任务物体真值位姿，注入为
  `source=robodojo_simulator_ground_truth`，用于复现论文的真值参数条件。
- `rgbd_pose`：读取 GUI 相机的 RGB、深度、内参和外参，调用
  `ESSAY2608_RGBD_POSE_ESTIMATOR=模块:函数`，输出 `{label: xyz+wxyz}`；接口定义在
  `source/data/robodojo_pose.py`。估计器缺失、标定缺失或输出标签不完整会直接失败，绝不
  静默回退到 Oracle。两条轨道最终都进入同一 policy 输入协议，结果表会记录
  `observation_mode`。

- DynaMAC 实现 `R3 × S3` 任务参数流、黎曼高斯 marginal、Product-of-Experts、式 (5)
  链接判定、式 (6) 流选择、虚拟末端帧、MiDiGaP 模态与技能转移；双臂是两套并发策略，
  对侧末端作为候选任务参数。
- RoboDojo 演示加载器使用 TAPAS 风格的运动学后端：由末端平移/旋转速度低谷和夹爪变化
  生成连续技能边界，并把阈值、平滑窗口和技能上限写入训练 provenance。它替代固定
  `[0,0.30),[0.30,0.68),[0.68,1.0]` 切段，但不声称包含论文外部的 DINO/SAM 视觉候选前端。
- MiDiGaP 静态帧对照保留任务参数、轨迹模态和 PoE，但不做 DynaMAC 的运动学链接过滤与
  虚拟帧补偿。
- VAPOR 同时提供 `slsqp` 和 `augmented_lagrangian_fd` 两个可审计后端；后者复现增广
  拉格朗日更新，但因项目没有论文所用的 Kineverse 机器人符号模型，只使用有限差分
  `forward_kinematics`，不能宣称与 Kineverse Jacobian 完全等价。
- RGB-D 轨道提供 `builtin:dino_sam`：Transformers DINOv2 提取候选外观特征、SAM 网格
  提示生成候选掩码，再结合深度和相机标定反投影为位姿。它是可运行的通用候选前端，输出
  `visual_candidate_XXX`；论文专用的 TAPAS 标签提示策略仍需按任务固定标签审计。
- DP 是真值状态条件的一维时序 U-Net DDPM 独立复现，checkpoint 使用无 pickle NPZ。
  当前双臂 DP 为左右臂独立模型，必须单列，完成联合动作 DP 前不能冒充论文同配置基线。
- RoboDojo 官方 HDF5 不直接提供 DynaMAC 所需的动态物体真值位姿。正式训练数据必须在
  GUI 仿真中按布局回放并同步补采任务帧；静态初始布局不能冒充整段物体轨迹。`fit` 的
  `--episodes N` 支持任意已下载数量，`--capture-root` 会强制逐条校验补采文件、步数、
  帧集合、位姿有限性和 `.audit.json` 的 `accepted_for_training=true`，缺失或原生任务
  失败时直接拒绝训练。
- 数据流程分为两步：`robodojo demos` 下载官方专家 HDF5（动作来源），`robodojo
  capture-batch` 自动启动 GUI 回放并生成带任务帧的 JSONL（训练输入）。第二步不是手动
  示教；只有布局匹配失败或 GUI 原生成功验收失败时才需要人工检查，不能直接拿下载的 HDF5
  冒充 DynaMAC/MiDiGaP 的完整训练集。
- `data/dynamac_demos.npz` 只用于算法结构回归测试，不是 RoboDojo 训练数据，也不能产生
  论文成功率。

论文表只会收录满足 GUI 标记、完整回合数和 0/1/2 三种子的真实结果。缺失实验显示为
“待评测”，不会填入估计值或占位成功率。

## 验证

```bash
python -m pip install -e '.[test,midigap,baselines]'
pytest -q
ruff check source scripts tests
```

RoboDojo 代码仓库的 `LICENSE` 为 MIT，但上游 README 同时写有非商业研究用途说明，资产
还可能有独立条款。本项目当前只作非商业论文研究，发布资产、数据或 checkpoint 前需按
固定 revision 再次审计授权。
