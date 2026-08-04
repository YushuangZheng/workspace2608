# Essay2608：DynaMAC 核心机制复现与 Isaac Lab 项目整理

> 更新时间：2026-08-03
> 项目用户：`zys`
> 项目路径：`/home/zys/workspace/essay2608`
> 当前主线：在 Isaac Lab 中自行复现 DynaMAC 的核心机制，而非严格复现原论文的 RLBench / RLBench2 benchmark。

---

## 1. 项目总览

### 1.1 研究对象

重点论文：

**One Hand Watches The Other: Dynamic Multi-Agent Cooperation for Sample-Efficient Bimanual Manipulation in Dynamic Environments**

论文的核心不是提出新的扩散模型或流匹配模型，而是研究：

- 如何利用物体中心的相对几何关系提高模仿学习的数据效率；
- 如何在动态环境中处理参考坐标系失效；
- 如何让双臂通过动态参考系实现协作，而不固定主臂和从臂。

### 1.2 当前复现目标

当前不是完整复现论文所有实验表格，而是先复现以下核心机制：

1. 物体中心相对轨迹是否比世界坐标轨迹具有更低方差；
2. 抓取后，物体与末端刚性连接是否导致物体 stream 异常高精度；
3. 动态屏蔽失效参考系是否能避免专家乘积坍缩；
4. 虚拟末端参考系是否能补充抓取后的空间参照；
5. 后续扩展到双臂时，能否把另一只机械臂末端作为动态参考系。

当前最终平台决策为：

\[
\boxed{\text{Isaac Lab 外部项目}+\text{Isaac Lab 自采静态演示}+\text{DynaMAC 核心机制}}
\]

不再优先搭建 RLBench，也不直接把 RLBench2 数据导入 Isaac Lab 训练。

---

## 2. 论文核心概念

### 2.1 归纳偏置

“归纳偏置”不是贬义的偏见，而是模型在训练前被人为加入的结构假设。

本论文中的核心归纳偏置是：

> 机器人动作主要由机械臂、物体和协作者之间的相对三维几何关系决定，而不是由它们在世界坐标系中的绝对位置决定。

例如抓杯子时，模型学习：

\[
T_{\text{ee}}^{\text{cup}}
=
\left(T_{\text{cup}}^W\right)^{-1}T_{\text{ee}}^W
\]

而不是只记住末端在世界坐标系中的固定位置。

这种设计减少了模型需要从数据中自行发现的规律，因此能提高少样本效率和几何泛化。

### 2.2 相对几何关系

机器人操作中的几何关系不只是距离，主要包括：

- 相对位置、距离和方向；
- 相对朝向；
- 点、轴、面之间的对齐；
- 包含与进入关系；
- 接触和邻近关系；
- 安全间隙；
- 相对速度和角速度；
- 双臂或夹爪之间的固定相对位姿；
- 可见性、遮挡和几何可达性。

统一的六自由度相对位姿表达为：

\[
{}^A T_B =
\begin{bmatrix}
{}^A R_B & {}^A p_B \\
0 & 1
\end{bmatrix}
\]

其中：

- \({}^A p_B\)：B 相对于 A 的位置；
- \({}^A R_B\)：B 相对于 A 的朝向。

### 2.3 适合的任务

相对几何特别适合：

- 动态目标抓取；
- 双臂交接；
- 双臂协同搬运刚体；
- 动态对准；
- 插入前的粗对准；
- 场景位置变化但局部操作模式不变的任务。

这些任务通常满足：

- 对象主要是刚体；
- 成功主要取决于位置、方向、姿态和相对运动；
- 场景整体平移或旋转后，局部操作规律仍成立。

### 2.4 局限任务

相对几何对以下任务能力有限：

- 精密插入的最终接触阶段；
- 拧瓶盖、拧螺丝等强扭矩任务；
- 倒水、颗粒物和流体操作；
- 柔性物体，如布料、绳索、电缆；
- 强力觉、触觉依赖任务；
- 隐藏状态任务；
- 拓扑变化任务；
- 长程语义规划任务。

判断标准：

> 当“相对位姿正确”基本等价于“任务状态正确”时，这类方法很强；当几何正确不再等价于物理或语义正确时，能力会明显下降。

---

## 3. 专家乘积、注意力与相对几何

多个局部参考系对应多个局部策略：

\[
\pi(a)\propto \prod_i \pi_i(a)
\]

若每个专家输出高斯分布：

\[
\pi_i(a)=\mathcal N(\mu_i,\Sigma_i)
\]

融合权重与精度有关：

\[
\Lambda_i=\Sigma_i^{-1}
\]

精度越高，专家在融合中的影响越大。

三者的层级不同：

| 概念 | 回答的问题 | 层级 |
|---|---|---|
| 归纳偏置 | 模型应该怎样理解问题 | 架构先验 |
| 注意力 | 当前应重点读取什么信息 | 特征交互 |
| 专家乘积 | 多个局部策略如何形成统一动作 | 概率决策融合 |

专家乘积更像“约束求交”，注意力更像“信息加权读取”。

---

## 4. 扩散模型、流匹配和坐标表示

### 4.1 扩散模型是否学习绝对位置

不一定。扩散模型描述的是如何生成动作分布：

\[
p(A\mid o)
\]

动作 \(A\) 可以是：

- 世界坐标绝对位姿；
- 机器人基座坐标位姿；
- 当前末端的增量动作；
- 物体中心相对动作；
- 关节位置、关节增量、速度或力矩；
- action chunk。

因此：

\[
\text{扩散模型}\neq\text{绝对坐标模型}
\]

### 4.2 流匹配是否学习相对位置

也不一定。流匹配学习速度场：

\[
\frac{dA_\tau}{d\tau}=v_\theta(A_\tau,\tau,o)
\]

但动作 \(A\) 仍可以是绝对、相对、关节或潜空间表示。

因此：

\[
\text{流匹配}\neq\text{相对坐标模型}
\]

### 4.3 泛化来源

生成式机器人策略的泛化可能来自：

- 闭环观测条件化；
- 数据覆盖和数据多样性；
- 大规模预训练；
- 多模态动作建模；
- 模型结构；
- 坐标表示；
- 显式几何归纳偏置。

不能把所有泛化都简单称为“大数据后的涌现”。

---

## 5. DynaMAC 方法定位

### 5.1 是否属于流匹配

论文当前实验实现不属于流匹配，主要基于：

- MiDiGaP；
- 高斯局部轨迹模型；
- Product-of-Experts；
- 动态参考系处理。

DynaMAC 更接近 policy-agnostic 的动态参考系框架，理论上可与流匹配或扩散策略结合，但论文当前实现不是流匹配。

### 5.2 是否只是因为学习相对位置而更强

不能简单概括为“因为学习相对位置”。

MiDiGaP 本身已经采用物体中心相对坐标；DynaMAC 相比已有多流方法真正新增的是：

1. 检测物体与机械臂之间的运动学连接；
2. 屏蔽已经不再外生的物体参考系；
3. 加入虚拟末端参考系；
4. 在双臂中把对侧末端作为动态参考系。

更准确的总结是：

\[
\boxed{\text{相对几何}+\text{动态参考系的因果解耦}}
\]

---

## 6. 物体中心从哪里来

论文中的“物体中心”不是完全由策略端到端自行涌现出来的。

### 仿真中

- 物体身份和位姿由仿真器直接提供；
- 策略使用 ground-truth object pose。

### 真机中

- 物体相关坐标由外部 RGB-D / DINO 特征模块估计；
- 不是 DynaMAC 策略本身从像素中自主发现完整对象概念。

### 自动完成的部分

DynaMAC 自动处理：

- 哪些候选参考系与当前技能有关；
- 何时物体与末端发生运动学连接；
- 何时屏蔽物体 stream；
- 何时创建虚拟末端参考系；
- 双臂中对侧末端何时成为有效参考系。

---

## 7. 与老师讨论时的讲法

### 7.1 30 秒概括

> 这篇论文不是做一个更大的模型，而是发现传统多流模仿学习隐含假设所有物体参考系都是外部条件。物体一旦被抓住，就由机械臂决定运动，原因果方向反转；此时物体相对末端几乎不变，专家乘积会把它误认为极可靠专家并产生策略坍缩。DynaMAC 动态检测这种运动学耦合，移除失效参考系，并用虚拟末端坐标补充空间记忆。在双臂任务中，再将另一只手作为动态参考系，实现无固定主从角色的协调。

### 7.2 论文价值

- 不是靠扩大模型和数据；
- 通过结构先验获得少样本动态泛化；
- 指出了传统多流学习中的外生性假设；
- 用轻量机制解决动态抓取后参考系失效。

### 7.3 局限

- 依赖外部物体位姿；
- 依赖技能分段；
- 更适合刚体和几何主导任务；
- 连接检测主要依赖低方差，仍可能误判；
- 对遮挡、感知噪声、柔性和强接触任务支持有限。

---

## 8. 短期和长期研究规划

### 8.1 短期目标

1. 搭建最小仿真任务；
2. 采集 5 条静态成功演示；
3. 验证世界坐标、物体坐标和目标坐标中的轨迹方差；
4. 验证抓取后物体 stream 低方差 / 高精度现象；
5. 实现 Product-of-Experts；
6. 实现动态 frame mask；
7. 实现 virtual EE frame；
8. 加入动态物体移动；
9. 后续扩展双臂交接。

### 8.2 中期候选创新

- 不确定性感知的动态参考系门控；
- 从硬屏蔽改为软门控；
- 自动发现任务相关局部参考系；
- 与扩散模型或流匹配结合；
- 多参考系冲突检测；
- 感知误差鲁棒性。

### 8.3 长期方向

从固定候选参考系扩展为动态关系图：

\[
\text{Task Graph}+\text{Dynamic Relation Graph}+\text{Generative Policy}
\]

长期可能包括：

- 机械臂—物体关系；
- 物体—目标关系；
- 左臂—右臂关系；
- 接触、支撑、遮挡、包含关系；
- 力觉、触觉和动力学；
- 多机器人协作；
- 长程任务规划与完成状态记忆。

---

## 9. 最终平台决策

### 9.1 已放弃的路线

#### 先搭 RLBench / RLBench2 再迁移

不采用，原因：

- 现有设备和控制链路已经在 Isaac Sim / Isaac Lab；
- 维护两套仿真环境成本高；
- 当前目标是核心机制复现，不是严格 benchmark 对齐。

#### 直接导入 RLBench2 数据到 Isaac Lab

不采用，原因：

- 两个平台机器人、坐标系、控制方式和物理参数不同；
- RLBench2 轨迹不能直接在 Isaac Lab 中闭环执行；
- 可能出现不可达、碰撞、抓取点不一致等问题。

### 9.2 当前最终路线

\[
\boxed{\text{Isaac Lab 为唯一主平台，重新搭任务并自采演示}}
\]

RLBench2 只用于：

- 任务设计参考；
- 实验指标参考；
- 双臂任务流程参考；
- 后期可能的跨平台验证。

---

## 10. 硬件与系统环境

### GPU

```text
NVIDIA GeForce RTX 4090
显存：24564 MiB
驱动：550.90.07
nvidia-smi 显示 CUDA 12.4
```

### 内存

```text
物理内存：60 GiB
可用内存：约 48 GiB
Swap：2 GiB
```

### 磁盘

```text
/dev/nvme0n1p2
总容量：3.7T
可用：3.2T
```

硬件足够运行：

- Isaac Sim / Isaac Lab；
- 单环境和多环境仿真；
- 高斯多流策略；
- Diffusion Policy 基线；
- 后续双臂任务。

---

## 11. zys 用户环境

已确认：

```text
whoami: zys
HOME: /home/zys
Conda base: /home/zys/miniconda3
PATH 中没有 /home/ps
```

Conda 环境：

```text
base
env_isaaclab
openvla-oft
```

Isaac Lab 环境：

```text
/home/zys/miniconda3/envs/env_isaaclab/bin/python
Python 3.10.19
```

Isaac Sim / Isaac Lab：

```text
Isaac Sim：4.5.0.0（pip 安装）
Isaac Lab：/home/zys/IsaacLab
Isaac Lab git：main 分支，v2.2.1 后续提交
```

Isaac Sim 不是独立二进制安装，因此：

- 没有 `isaac-sim.sh`；
- 没有 `python.sh`；
- 没有 `_isaac_sim` 软链接属于正常；
- 通过 `isaacsim` 和当前 Conda Python 启动。

PyTorch：

```text
PyTorch：2.7.0+cu128
torch.version.cuda：12.8
CUDA available：True
GPU：RTX 4090
```

已完成实际 GPU 运算和 Isaac Sim / Isaac Lab 冒烟测试，全部通过，因此当前环境不再改动。

---

## 12. Isaac Lab 仓库状态

当前 Isaac Lab 仓库存在 3 个本地修改：

```text
source/isaaclab_tasks/.../lift/config/franka/joint_pos_env_cfg.py
source/isaaclab_tasks/.../lift/config/openarm/joint_pos_env_cfg.py
source/isaaclab_tasks/.../reach/reach_env_cfg.py
```

已建议备份：

```bash
git diff > ~/workspace/isaaclab_local_changes.patch
```

原则：

> `~/IsaacLab` 只作为底层依赖，不再在其中开发 Essay2608 代码。

---

## 13. Essay2608 项目结构

项目根目录：

```text
/home/zys/workspace/essay2608
```

当前采用 Isaac Lab external project，同时参考 RoboTwin 和 Diffusion Policy 的简洁组织原则。

```text
essay2608/
├── source/
│   └── essay2608/
│       ├── config/
│       ├── setup.py
│       ├── pyproject.toml
│       └── essay2608/
│           ├── tasks/
│           │   └── manager_based/
│           │       ├── essay2608/
│           │       └── dynamic_pick_place/
│           └── expert.py
│
├── scripts/
│   ├── smoke_test.py
│   ├── test_dynamic_pick_place.py
│   ├── scripted_pick_place.py
│   ├── collect_demos.py
│   └── analyze_relative_frames.py
│
├── data/
│   └── static_demos/
│
├── outputs/
│   └── relative_frame_analysis/
│
├── README.md
├── pyproject.toml
└── .gitignore
```

职责划分：

- `tasks/`：Isaac Lab 环境；
- `expert.py`：可复用脚本化专家；
- `scripts/`：采集、评测和分析入口；
- `data/`：演示数据；
- `outputs/`：分析图和结果；
- 后续 `policy/`：DynaMAC 算法实现。

---

## 14. 项目初始化与导入

已成功执行 editable 安装：

```bash
python -m pip install -e ./source/essay2608
```

普通 Python 直接 `import essay2608` 会因为未启动 Isaac Sim 而缺少 `pxr`。

正确方式：

```python
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import essay2608
```

已验证导入成功。

模板任务：

```text
Template-Essay2608-v0
```

已完成：

- Gym 注册；
- 环境创建；
- reset；
- 100 步 step；
- GUI 和 headless 运行。

---

## 15. Git 状态与提交

基线提交：

```text
eeeccbb Initialize working Isaac Lab project
tag: baseline-template
```

开发分支：

```text
feature/dynamic-pick-place
```

动态拾取放置环境提交：

```text
9b8e472 Add dynamic pick-and-place environment
```

---

## 16. Dynamic Pick-and-Place 环境

任务 ID：

```text
Essay2608-Dynamic-Pick-Place-v0
```

当前继承 Isaac Lab 已验证的 Franka Cube Lift 环境。

场景包括：

- Franka Panda；
- 实验桌；
- 可抓取方块；
- 目标位姿标记；
- 末端 FrameTransformer；
- 夹爪控制；
- 绝对位姿 IK 控制。

动作格式：

```text
[x, y, z, qw, qx, qy, qz, gripper]
```

动作维度：

```text
8
```

其中：

- `gripper = +1`：张开；
- `gripper = -1`：闭合。

GUI 中：

- 白色机械臂：Franka Panda；
- 小方块：抓取对象；
- 桌面中央坐标轴：目标位姿；
- 夹爪附近坐标轴：末端执行器坐标系；
- 红色：X；
- 绿色：Y；
- 蓝色：Z。

---

## 17. 脚本化 IK 专家

状态机阶段：

```text
REST
APPROACH_ABOVE_OBJECT
APPROACH_OBJECT
GRASP_OBJECT
LIFT_OBJECT
MOVE_ABOVE_TARGET
LOWER_TO_TARGET
RELEASE_OBJECT
RETREAT
COMPLETE
```

机器人动作流程：

```text
移动到方块上方
→ 下探
→ 闭合夹爪
→ 抬升
→ 移动到目标
→ 下放
→ 张开夹爪
→ 撤离
```

该专家已经成功运行。

---

## 18. 静态演示采集

已经建立：

```text
scripts/collect_demos.py
source/essay2608/essay2608/expert.py
```

已采集：

```text
5 条静态成功演示
```

默认目录：

```text
data/static_demos/
```

每条演示保存为：

```text
demo_000.npz
demo_001.npz
...
manifest.json
```

每条数据包含：

```text
time          [T]
state         [T]
ee_pose       [T, 7]
object_pose   [T, 7]
target_pose   [T, 7]
action        [T, 8]
joint_pos     [T, 9]
joint_vel     [T, 9]
control_dt
final_error
```

约定：

```text
四元数顺序：wxyz
坐标系：local_environment
```

`data/` 和 `outputs/` 不提交到 Git。

---

## 19. 相对坐标分析

已经给出离线分析脚本：

```text
scripts/analyze_relative_frames.py
```

分析内容：

1. 世界坐标末端轨迹；
2. 物体中心末端轨迹；
3. 目标中心末端轨迹；
4. 各状态下 5 条演示的 RMS 标准差；
5. 抓取后的 Object–EE 刚性指标。

核心变换：

\[
T_{\mathrm{ee}}^{\mathrm{object}}
=
\left(T_{\mathrm{object}}\right)^{-1}T_{\mathrm{ee}}
\]

输出目录：

```text
outputs/relative_frame_analysis/
```

输出文件：

```text
relative_frame_statistics.npz
summary.json
position_std.png
rotation_std.png
```

当前状态：

- 数据采集已确认完成；
- 相对坐标分析脚本已提供；
- 尚未在对话中收到分析结果数值，因此下一步应运行分析并检查方差表。

---

## 20. 当前项目状态

### 已确认完成

- zys 独立 Conda 环境；
- Isaac Sim 4.5 可运行；
- Isaac Lab 可运行；
- GPU 计算测试通过；
- Essay2608 external project 创建；
- editable 安装；
- 模板任务注册和运行；
- Dynamic Pick-and-Place 环境；
- Franka GUI 场景；
- 绝对位姿 IK；
- 脚本化 pick-and-place 专家；
- 静态演示采集代码；
- 5 条演示采集；
- 相对坐标分析脚本。

### 尚未确认完成

- 运行 `analyze_relative_frames.py` 后的实际方差结果；
- 根据真实数据设置连接检测阈值；
- Gaussian Stream 拟合；
- Product-of-Experts；
- 动态参考系 mask；
- Virtual EE Frame；
- 动态目标扰动评测；
- 双臂 handover。

---

## 21. 下一步

运行离线分析：

```bash
cd ~/workspace/essay2608
conda activate env_isaaclab

python scripts/analyze_relative_frames.py   --data_dir data/static_demos   --output_dir outputs/relative_frame_analysis
```

查看结果：

```bash
cat outputs/relative_frame_analysis/summary.json

xdg-open outputs/relative_frame_analysis/position_std.png
xdg-open outputs/relative_frame_analysis/rotation_std.png
```

重点关注：

- `approach` / `grasp` 时物体坐标方差是否小于世界坐标；
- `lift` / `move_target` / `lower_target` 时 Object–EE 相对位姿方差是否接近零；
- 是否出现抓取后异常高精度现象。

下一步根据真实数值定义连接度量：

\[
M_t^{(f)}
=
\left|\det(\Lambda_t^{(f)})ight|^{-1/(2d)}
\]

并设置连接检测阈值，而不是直接照抄论文阈值。

---

## 22. 后续算法文件建议

当相对方差分析通过后，再新增：

```text
source/essay2608/essay2608/policy/dynamac/
├── dataset.py
├── gaussian_stream.py
├── product_of_experts.py
├── frame_manager.py
└── policy.py
```

职责：

- `dataset.py`：读取演示、分段、重采样、坐标转换；
- `gaussian_stream.py`：局部轨迹高斯模型；
- `product_of_experts.py`：世界坐标融合；
- `frame_manager.py`：连接检测、mask、virtual frame；
- `policy.py`：完整 DynaMAC 推理入口。

---

## 23. 开发原则

1. 不修改 Isaac Lab 核心源码；
2. 所有研究代码放在 `essay2608`；
3. 每个阶段先做最小可运行版本；
4. 先使用仿真器真值状态，不引入视觉噪声；
5. 先单臂验证核心因果问题，再扩展双臂；
6. 每个关键阶段单独 Git 提交；
7. 数据与输出不提交 Git；
8. 阈值根据实测数据确定；
9. 不同时引入 DynaMAC、视觉、扩散和双臂，避免无法定位问题；
10. 所有文件修改优先提供可直接执行的终端命令，不要求手工找文件编辑。

---

## 24. 一句话总结

> Essay2608 当前已在 Isaac Lab 中完成从环境、Franka 绝对位姿 IK、脚本化专家到 5 条静态演示采集的完整链路；下一步是用真实采集数据验证物体中心轨迹低方差和抓取后 Object–EE 刚性连接现象，然后实现 Gaussian Streams、Product-of-Experts、动态参考系屏蔽和虚拟末端参考系。

---

## 25. 2026-08-03 单臂最小闭环更新

静态演示已完成自动验收并冻结：

```text
data/pick_place_static/v1
dataset_sha256: 8956857d034694090ec0d1bf39c33364f95cac723954ac3baedcbd1fd8e479f8
```

冻结 manifest 已记录每条数据的 SHA-256、seed、初始位姿、状态序列、步数、最终误差、最大单步位姿跳变、抓取后 Object–EE 稳定性和验收时 Git commit。采集器会拒绝覆盖包含 `FROZEN` 标记的数据目录。

单臂最小闭环已经实现：

- world/object/target 相对坐标方差分析；
- World Gaussian；
- Static object/target Gaussian Multi-stream + PoE；
- Mask-only；
- Full DynaMAC（连接检测、动态 mask、virtual EE frame）；
- 6 种静态/动态条件的独立进程评测和诊断指标。

seed `6200` 的首轮工程验收中，World Gaussian 与 Static Multi-stream 为 `0/6`，Mask-only 与 Full DynaMAC 为 `6/6`。Full DynaMAC 相比 Mask-only 的平均轨迹长度从约 `1.333 m` 降至 `1.039 m`，推理时间约 `0.236 ms`。这些只是单 seed 最小闭环结果，不能作为论文统计结论；投稿实验仍需至少三个随机种子、置信区间和失败案例分析。
