# 双臂交接环境与数据骨架

## 范围

本阶段只验收 `Essay2608-Bimanual-Handover-v0` 的环境、脚本专家、采集、
数据审计和 headless 冒烟链路，不主张已经实现接触丰富的物理交接或完整双臂
DynaMAC。仓库中更早的双臂策略试验早于本次审计，未参与 v2 数据集验收。

v2 采集所用实现来源为提交 `5c4921e`。旧的
`data/handover_static/v1` 保持逐字节不变，并继续通过兼容的旧版
`carrier` 模式加载。

## 环境契约

场景包含两台独立根节点的 Franka Panda、一个交接方块、一张桌子和固定放置目标。
16 维动作定义如下：

| 切片 | 含义 | 控制器 |
|---|---|---|
| 0:7 | 左末端位姿，`xyz + wxyz` | 绝对 DLS 微分 IK |
| 7 | 左夹爪 | 独立二值位置命令 |
| 8:15 | 右末端位姿，`xyz + wxyz` | 绝对 DLS 微分 IK |
| 15 | 右夹爪 | 独立二值位置命令 |

策略观测组直接暴露所需几何状态：

| 观测 | 形状 | 来源 |
|---|---:|---|
| `left_ee_pose` | 7 | 局部环境坐标中的左工具实测位姿 |
| `right_ee_pose` | 7 | 局部环境坐标中的右工具实测位姿 |
| `object_pose` | 7 | 刚体物体根节点实测位姿 |
| `target_pose` | 7 | 固定放置目标 |
| `left_gripper_state` | 2 | 左侧两个指关节实测位置 |
| `right_gripper_state` | 2 | 右侧两个指关节实测位置 |
| `actions` | 16 | 用于诊断的上一时刻动作 |

seed 7300 的 headless 环境构造确认了上述名称、形状和四个动作项；同一进程完成
575 步完整交接，最终误差为 10.62 mm。

## 专家与关系监督

专家按顺序执行全部 13 个状态：

```text
REST → LEFT_APPROACH → LEFT_GRASP → LEFT_LIFT → LEFT_TO_HANDOVER
→ RIGHT_APPROACH → RIGHT_GRASP → TRANSFER → LEFT_RELEASE
→ RIGHT_TO_TARGET → RIGHT_RELEASE → RETREAT → COMPLETE
```

每个记录步都有与专家状态对齐的 `relation_label`：

| 标签 | 专家状态 | 含义 |
|---|---|---|
| `none` | 0–1、10–12 | 不认为任何机械臂与物体连接 |
| `left_only` | 2–6 | 左臂搬运，右臂接近并闭合 |
| `both` | 7 | 左臂释放前的短时共同持物 |
| `right_only` | 8–9 | 右臂搬运至目标 |

控制周期为 20 ms；每条验收通过的 v2 演示中，`TRANSFER` 均持续 15 步，
即 0.30 s。旧版整数 `carrier` 字段仍单独保留，用来选择几何附着模拟器所跟随的
单个末端；它不能被解释为四值关系真值。

## 独立进程采集与冻结 v2

`scripts/collect_handover.py` 是统一的轻量入口。控制器为每次尝试启动新的仿真
进程，只接受完整且低于固定 60 mm 成功阈值的 episode，并且只把成功 NPZ 移入
目标目录。

单条冒烟数据使用 seed 7300 和独立输出目录。正式 v2 从 seed 7400 开始，8 次
尝试中接受 5 次成功：7400、7403、7404、7406、7407。失败 worker 被拒绝，
没有轨迹进入冻结集。

冻结数据集：

- 路径：`data/handover_static/v2`；
- 演示数：5，每条 582–604 步；
- 数据集 SHA-256：
  `91706df18abfea606c9e6836f1864e675610633ce5cb0c3c23846a1ea4f5fe18`；
- 最大最终误差：11.04 mm；
- 初始物体两两最小距离：13.91 mm；
- 最大单步笛卡尔跳变：29.21 mm；
- 左/右物体连接位置 RMS 标准差最大值：2.01/3.38 mm；
- 五条数据的关系模式均为 `four_value_state_aligned_v2`；
- 五条数据的两侧指关节测量均覆盖约 0–40 mm。

审计检查必需数组、有限数值、16 维动作、7 维位姿、连续时间戳、完整状态和关系
序列、标签与状态一致性、夹爪开闭实测、复位式跳变、不同初始位姿、最终误差、
逐文件 SHA-256 和连接稳定性。采集与冻结脚本都拒绝带 `FROZEN` 的目录；两个
拒绝反例均返回非零，且 manifest 哈希保持不变。

## 复现

采集一条独立冒烟演示：

```bash
conda run -n env_isaaclab python scripts/collect_handover.py --headless \
  --num_demos 1 --max_attempts 3 --seed 7300 \
  --output_dir outputs/handover_scientific/smoke_v2
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir outputs/handover_scientific/smoke_v2
```

采集一个新的、未冻结的五演示版本。不得复用 `v1` 或 `v2` 作为输出目录：

```bash
conda run -n env_isaaclab python scripts/collect_handover.py --headless \
  --num_demos 5 --max_attempts 10 --seed 7500 \
  --output_dir data/handover_static/v3
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir data/handover_static/v3
conda run -n env_isaaclab python scripts/audit_handover_dataset.py \
  --data_dir data/handover_static/v3 --freeze \
  --dataset_version handover_static_v3
```

## 科学限制

方块禁用了重力，并由程序写入单一载体末端位姿；短时 `both` 是脚本监督，不是
力或接触测量。当前成功表示几何流程完成和最终位置误差达标，不是经过验证的物理
抓取稳定性指标。五条演示建立了可复现的数据与监督接口，但不能证明任何现有双臂
学习策略具有泛化能力或优于基线。

进入策略研究前，下一项环境升级应加入接触传感和非运动学附着物体，再把脚本标签
与实测双臂接触及相对运动证据进行比较。
