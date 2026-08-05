# DynaMAC 研究日志

## 2026-08-04：按论文语义重新收缩项目

### 最重要的纠偏

先前项目把“DynaMAC”逐渐扩展成了在线关系估计、接触门控和故障恢复系统。精读论文后
确认这不是论文 Algorithm 1：

- 运动学链接从演示中的技能级流协方差离线推断；
- 每个技能的有效任务参数集合在拟合阶段固定；
- 推理时实时变化的是保留参考系的世界位姿，不是链接判定；
- MiDiGaP 主要按离散轨迹索引切换技能，不等待真实抓取或接触完成；
- 双臂是两个并发单臂策略，对侧末端只是候选动态任务参数。

所以旧的 `OnlineRelationEstimator`、`RelationRecoveryController` 和双臂恢复 supervisor
都属于论文之外的研究扩展。它们已从当前代码树删除，不能再作为 DynaMAC 本体引用。

### 图 2 中 marginals 的准确含义

每一条 stream 都是在一个局部参考系中拟合的条件分布。推理时用该参考系的当前位姿把
局部均值和协方差变换到世界系，所得 `p(xi_ee | f)` 就是该 stream 的世界系 marginal。
橙色是技能起点冻结的静态末端帧，蓝色是杯子帧，绿色不是第三个时间阶段，而是两条
marginal 的 Product-of-Experts。

高斯乘积中联合精度为各精度之和：

`Lambda_joint = sum Lambda_f`，`Sigma_joint = inverse(Lambda_joint)`。

联合均值等于 `Sigma_joint @ sum(Lambda_f @ mu_f)`，所以会“乘以 joint 协方差”：它把
累加的信息向量重新换回均值尺度。在流形上，本实现对各均值取共同切空间 Log，应用同一
闭式增量，再用 Exp 返回 `R3 × S3`。

### 图 3 的精度尖峰

绿色阴影是压缩后的六维标准差摘要，蓝/橙线是完整六维精度矩阵行列式，二者不是一个
标量的简单镜像。线性右轴高达约 `1e15`，低精度阶段即使增长许多数量级也看似贴零；
单个很小的特征值、维度相关性或正则化下限都足以让行列式瞬间暴涨。论文因此使用式 (5)
的几何平均标准差，并说明短暂 pre-grasp 假阳性可忽略或做时间过滤。

### 公开代码状态

- 论文项目页：`https://dynamac.cs.uni-freiburg.de`；
- 官方仓库：`https://github.com/robot-learning-freiburg/DynaMAC`；
- 2026-08-04 查询结果：公开仓库描述为 `Coming soon`，没有源码；
- MiDiGaP 官网也写明代码将于之后发布。

因此当前实现是对论文可观察规范的独立复现，不能称作官方代码镜像。

### 论文未唯一规定、必须显式冻结的选择

论文给出 `tau_M = 0.001` 的解释，却没有公开所有任务的 `tau_omega`、协方差正则化、
短尖峰过滤窗口、链接如何从逐时刻归约为技能级标签，以及 MiDiGaP 模态聚类的完整运行
配置。本复现把这些全部放入 `configs/dynamac.json`：

- `tau_m = 0.001`；
- `tau_omega = 0.2`（项目预声明选择，不宣称来自论文）；
- 链接至少覆盖技能一半且连续不少于 3 个离散点；
- 位置/旋转方差下限均为 `1e-8`；
- 轨迹级确定性 k-means + BIC，最多 3 个模态，每模态至少 2 条演示。

实现采用 MiDiGaP 论文明确列出的第二种模态划分：在降采样后的乘积流形 `M^T` 上做
Riemannian k-means，并用 BIC 选择模态数。BIC 的具体似然参数化在公开论文中仍未唯一
给定，本实现冻结为共享各向同性残差模型。每个技能保留演示的模态标签，按 MiDiGaP
式 (12) 用相邻技能演示集合交集计算转移矩阵，再按式 (13) 采样整条路径；确定性运行则
用 Viterbi 取全局最大概率路径，不再逐技能独立贪心。

位姿重采样对四元数使用 SLERP，逐时刻姿态均值使用 Karcher/Fréchet 迭代。运动学链接
使用实测末端位姿计算，策略流使用控制目标位姿拟合；这样不会因控制器跟踪误差把“指令
相关性”误判为真实刚性链接。虚拟帧在新技能第一次 `act` 的观测上捕获，而非上一技能
的末帧旧观测。

MiDiGaP 论文的逐时刻协方差采用对角形式；本实现遵守这一点。其修订版还给出 `1e-6`
对角正则作为另一种可选方案，并非唯一设置；当前配置冻结为分位置/旋转的 `1e-8` 方差
下限。位置单位为米、旋转单位为
弧度，式 (5) 仍混合两种尺度，这也是原论文已承认但未在实验中加权的限制。

### 随附数据边界

`data/dynamac_demos.npz` 的 SHA-256 为
`41810267ae86bccc67f31f5478aef77f9ebd80057591cb71be4e22ee0880f5eb`。
它包含原冻结数据的前五条单臂和前五条物理双臂演示，原文件 SHA 与 seed 保存在包内。
为减少文件数，旧的几十个 NPZ/manifest/trial JSON 不再保留在当前树。

这些数据的技能标签来自脚本 state 的粗粒度合并，不是 TAPAS；目标帧变化不足，单臂数据
甚至会把部分“高度确定但并未运动学绑定”的帧判成链接。这正好证明式 (5) 不能在缺乏
任务实例变化的数据上被解释为万能因果检测器。随附数据只验代码路径，不产出论文性能。

## 历史实验摘要（代码已归档到 Git 历史）

- 单臂 RelationDynaMAC 扩展完成过检测、Oracle 消融与恢复图，但它是项目新增机制；
- 真实接触双臂脚本专家经三版迭代从 `6/20`、`18/20` 到 `20/20`；
- 冻结了 20 条物理交接演示，数据哈希为
  `a4a39ed4837558cecaaf73e7c5db9b6ff88e7eddfb3bcf9923df862df9e65e52`；
- 双臂在线关系估计 Phase 5 v2 为 `70/70` 硬审计通过，但只证明在线检测；
- 双臂关系恢复最终开发批次为 `30/30`，正式 v1 按用户指令在 `95/200` 时中止，无
  summary、无正式审计，不能写成正式结果；
- 旧文件最后完整汇总在提交 `5bc4d84` 之前，Git 历史可恢复全部协议、报告和脚本。

这些内容保留为研究思考，不再污染论文忠实 DynaMAC 的命名空间。

## 2026-08-04：依据 MiDiGaP 补齐复现并建立 Franka 场景

本轮对照 `MiDiGaP.pdf` 后确认，原实现已经包含 DynaMAC 所需的 DiGaP、轨迹级模态聚类、
模式先验和联合示范的技能转移，但缺少 MiDiGaP 独立接口及论文后半部分。现已补齐：

- 可变长度演示重采样、`M^T` 黎曼 k-means+BIC、逐时刻对角 DiGaP；
- 式 (14) 的技能边界高斯 KL 兼容度和未知技能序列转移矩阵；
- 式 (15)--(16) 的黎曼高斯 Monte Carlo 截断、Fréchet 均值与对角矩匹配；
- 式 (17)--(22) 的可达球、碰撞半空间与非凸占据约束；
- 式 (24) 的模态证据更新及该证据向前一技能入边的传播；
- 式 (29)--(32) 的 VAPOR 全轨迹关节优化。

VAPOR 公开公式已复现，但求解器边界必须诚实注明：论文使用未公开的 Kineverse Jacobian
和增广拉格朗日，本项目用 SciPy SLSQP 与有限差分。它验证数学目标与约束，不复现论文
宣称的约 `50 ms` 实现耗时。

代码目录从 `source/essay2608/essay2608/` 扁平化到 `source/`，逻辑包名仍由 setuptools
映射为 `essay2608`。同时增加两套 Isaac Lab 规格：单 Franka 抓取/放置，以及左右 Franka
相向交接。两套配置都实际在 Isaac Sim 4.5、RTX 4090 上完成资产实例化、物理 reset 和
至少一个仿真步；双臂实体列表同时包含 `left_robot` 与 `right_robot`。当前机器 OmniHub
不可达会令 Isaac Sim 4.5 的官方 `close()` 无限等待，命令行入口给官方清理五秒后会结束
专用仿真进程，这不影响已经完成的场景步进。

## 2026-08-04：RoboDojo 直接接入与论文评测协议

### 接入决策

RoboDojo 以 Git 子模块固定在提交 `25691aa78fb34bbbe798aa3d880d57ef2788f696`，其
XPolicyLab、Isaac Lab 和 CuRobo 嵌套子模块均初始化到上游固定提交。项目不修改上游工作
树；`.runtime/robodojo` 组合上游代码、项目资产库、单臂配置与原始结果目录，因此运行层
可以随时删除重建，后续也能明确审计上游版本差异。

上游原生是双 X5 `arx_x5`。本项目在运行层新增可组合的单 X5/Franka 左右臂和双 X5/
Franka 配置，布局 JSON 仍复用 `arx_x5` 的物体位姿，因为布局不包含机器人状态。RoboDojo
当前提供的 X5、Franka 资产统一纳入 `assets/robodojo/Assets/Robots` 管理。

### 候选任务与论文近邻

- 单臂 `push_T`：对应 SweepDust，单活动臂完成平面接触推动与精确目标对齐；
- 单臂 `pour_liquid_into_cup`：对应 StoreBottle，单活动臂操纵瓶子满足瓶—容器相对位姿；
- 双臂 `sweep_blocks`：对应 HandOver/SweepDust，任务文本明确要求扫帚跨手交接；

三项均保留 RoboDojo 原生任务类、奖励和布局。`smooth` 与 `teleport` 只由项目 GUI 包装器
对目标物体施加可复现扰动，属于动态扩展，不能混入 RoboDojo 静态主表。

### 数据与评测审计

官方 Hugging Face 数据集总量很大，本项目冻结为按任务下载：共用房间、相机、材质、全部
机器人资产、候选任务物体与对应布局；专家数据只取每任务前五条 HDF5。不能只按任务名称
臆测单/双臂：首条 `hang_mugs` 和 `pack_objects_into_box` 演示的左右臂都实际运动，因此
从单臂子集移除；`push_T` 中右臂路径为 1.4220 m、左臂仅 0.0027 m，
`pour_liquid_into_cup` 中右臂完全静止，二者归为单活动臂；`sweep_blocks` 的左右末端路径
分别为 2.2543 m 和 2.8090 m，确认是双臂任务。RoboDojo 上游 54 个可运行配置仍全部保留
在候选池，正式子集为两个单臂近邻任务加一个双臂交接/清扫任务。

官方 HDF5 有机器人状态、动作与相机数据，却没有 DynaMAC/MiDiGaP 训练所需的逐时刻物体
真值位姿。只从布局读取初始物体位姿会破坏运动学链接判定，不能使用。项目 GUI 评测包装器
已经能从 LayoutManager 注入 `xyz+wxyz` 任务帧；下一步必须在 GUI 中回放官方动作并同步
补采这些帧，再冻结训练包和技能分段。

策略侧已接到 XPolicyLab WebSocket 协议：环境仍运行 RoboDojo 原生评测循环，策略服务器
分别加载 DP、MiDiGaP 或 DynaMAC 的无 pickle checkpoint。结果目录使用真实方法名，并在
`additional_info` 记录动态条件、checkpoint 摘要和 `gui=true`。命令行没有正式评测的
`headless` 选项；缺少 `DISPLAY` 会直接拒绝运行。

主表预注册为每任务、每方法种子 0/1/2、每种子 50 回合。聚合器报告成功率和 RoboDojo
分数的种子均值±样本标准差，以及 150 回合合并 Wilson 95% CI；存在缺种子、缺回合、非
GUI 或同种子重复运行时不产出“完整”结果。当前 `results/robodojo/paper_table.md` 全部为
“待评测”，这是正确状态，不使用估计值补表。

### 运行环境边界

新建独立 Conda 环境 `RoboDojo`（Python 3.11），不改已有 Isaac Sim 4.5 环境；ffmpeg
已装入该环境以支持逐回合流式视频。上游当前要求 Isaac Sim 5.1 / Isaac Lab 2.3。机器为
RTX 4090 24 GB，但 NVIDIA 对 Isaac Sim 5.1 测试的 Linux 驱动版本高于本机 550.90.07；
是否实际兼容只以本机 GUI 启动和物理场景验证为准，不能仅凭安装成功宣称可运行。

## 2026-08-05：全资源组合与两条观测轨道验证

本轮把 RoboDojo 的任务、场景、机器人和 XPolicyLab policy 统一纳入动态资源注册表：
`robodojo resources` 当前发现 54 个可运行任务、`default/conveyor` 两个场景、上游及
项目覆盖机器人配置和 41 个 policy adapter。`assets/demos` 保留论文子集的默认轻量入口，
`--all` 才会解析全部 54 个任务，避免误触发大型数据集下载。

环境组合器支持 `--scene`、`--robot` 和 `--observation-mode`。`oracle_pose` 从
RoboDojo LayoutManager 读取真值任务帧；`rgbd_pose` 通过
`ESSAY2608_RGBD_POSE_ESTIMATOR=模块:函数` 调用用户的 RGB-D 位姿估计器，估计器缺失或
标签不全会立即报错，绝不回退到真值。两条轨道在策略侧保持同一帧协议，结果记录明确写入
`observation_mode`。

完成一次真实 GUI 烟测：`push_T`、DynaMAC、`seed=0`、Franka 单右臂、Oracle Pose，
Isaac Sim 5.1 成功创建房间、相机、桌面、T 形物体和 Franka 资产，策略 WebSocket 正常
连接并完成 600/600 步；该单回合 RoboDojo 原生判定为失败（`fail=1`），因此只能作为
“组合链路可运行”证据，不能作为论文性能结果。首次用 `seed=3` 的失败仅因本地冻结布局
只有 `0/1/2` 三个官方种子，已在记录中与代码错误区分。

此外，`external-eval` 接口允许任意上游 XPolicyLab policy server 复用同一 GUI 环境；项目
内置 `dp/midigap/dynamac` 仍是唯一论文策略入口，避免把上游 adapter 误称为项目复现。

## 2026-08-05：审计缺口修正——TAPAS 后端与真实位姿训练入口

根据本轮复现审计，修正了三个会影响科学闭环的工程缺口：

- 新增 `source/data/tapas.py`。它使用末端平移/旋转速度低谷、夹爪变化候选和最小段长约束
  生成连续技能边界；RoboDojo 的通用、`push_T` 和 `sweep_blocks` 加载路径均改用该分割，
  不再使用固定的 30%/68% 时间切段。运动学阈值、平滑窗口和技能上限均在实现中显式冻结，
  训练 provenance 记录分割版本。这里是可独立复现的 TAPAS 风格后端，不包含论文外部的
  DINO/SAM 视觉候选生成器，不能把后者写成已完成。
- `robodojo fit` 新增 `--episodes N` 和 `--capture-root`。加载器现在支持任意已下载的演示
  数量；若给出补采根目录，会逐条读取 `gui_capture.v1` JSONL，校验连续 step、步数、帧名
  集合和 `xyz+wxyz` 有限性，并优先使用逐时刻 Oracle/RGB-D 任务帧。没有补采时只保留旧的
  静态布局/对侧末端回退，并在元数据中明确限制。
- DP 的真实位姿训练命令固定为：先用 GUI `robodojo capture` 为同一任务的每条演示生成
  `data/robodojo/captured/<task>/episode_XXXXXXX.jsonl`，再执行
  `python scripts/run.py robodojo fit --policy dp --task push_T --episodes 5
  --capture-root data/robodojo/captured --output results/robodojo/checkpoints/push_T/dp_real_pose.npz`。
  若已下载并补采 100 条演示，将 `--episodes 5` 改为 `--episodes 100`；缺文件时入口会直接
  失败，不会用静态位姿伪造真实训练集。

本轮验证：`RoboDojo` 环境下 `ruff check source scripts tests` 通过，`pytest -q` 共 34 项
通过。MiDiGaP 的 Kineverse/增广拉格朗日求解器和 TAPAS 的外部视觉前端仍属于明确的外部
依赖边界，当前实现继续标注为独立数值复现。

## 2026-08-05：自动化训练数据补采流程

澄清训练数据边界：RoboDojo 官方 HDF5 是“专家动作数据”，不是完整的 DynaMAC/MiDiGaP
训练输入，因为其中没有逐时刻动态任务物体真值位姿。现在新增 `robodojo capture-batch`：

1. 自动读取已下载的 `episode_XXXXXXX.hdf5`；
2. 对 `push_T` 缺失的源布局自动执行 HDF5 RGB 轮廓重建；
3. 逐条启动 Isaac Sim GUI，由 `RoboDojoReplayCaptureModel` 自动按官方关节动作回放；
4. 同步把 GUI 的 Oracle Pose 或 RGB-D Pose 任务帧写入
   `data/robodojo/captured/<task>/episode_XXXXXXX.jsonl`；
5. 每条回放结束后执行原生成功、步数和真值帧完整性验收，并写出 `capture_batch.json`。

因此用户不需要手动拖动机器人或逐帧标注；只需保持 GUI 可见并观察回放。只有源布局不匹配、
原生任务失败或 RGB-D 估计器缺失时才需要人工检查。默认轨道是 Oracle Pose；设置
`ESSAY2608_RGBD_POSE_ESTIMATOR=模块:函数` 并加 `--observation-mode rgbd_pose` 才会走视觉估计。

首条 `push_T` 自动回放实际生成了 360 帧 JSONL，但 RoboDojo 原生成功判据为失败；该样本
被保留用于诊断视频，不进入训练。加载器现已强制检查同名 `.audit.json` 的
`accepted_for_training=true`，所以“文件存在”不再等同于“数据可训练”。这次失败说明源布局
或回放物理一致性仍需修正后再批量采集，不能为了凑齐数量放宽门禁。

本轮还加入两个可选后端：`VAPORConfig(solver="augmented_lagrangian_fd")` 的有限差分增广
拉格朗日求解器，以及 `ESSAY2608_RGBD_POSE_ESTIMATOR=builtin:dino_sam` 的 Transformers
DINOv2/SAM 候选前端。前者复现约束目标和增广更新但没有 Kineverse 符号 Jacobian；后者会
在首次运行时下载模型权重并输出通用 `visual_candidate_XXX`，不能把它误写成论文未公开
的任务专用 TAPAS 提示/标注实现。
