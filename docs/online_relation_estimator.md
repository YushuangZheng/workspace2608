# 双向在线关系估计器

## 范围与来源

`OnlineRelationEstimator` 是 essay2608 新增机制，不是 DynaMAC 论文中的连接分析。
论文在策略拟合前，根据技能内部演示精度识别连接；本估计器读取运行时机器人和物体
状态，既能建立关系，也能撤销关系，并且不读取当前阶段。旧的
`KinematicConnectionDetector` 保持不变，仍是只看平移、以张开命令复位的检测器。

`RelationDynaMACPolicy` 使用独立方法标签 `relation_dynamac`。它根据新的关系判定
屏蔽物体流，并在进入 `CONNECTED` 时捕获虚拟末端参考系。当前虚拟流的激活仍使用
阶段 4，因此只有关系逻辑是完全阶段无关的，整个控制器还不是。

## 输入与状态机

每次更新读取：

- Franka 两个指关节开度之和及其实测速度；
- 物体相对末端的位置和朝向；
- 瞬时相对线速度和角速度；
- 窗口内相对位置 RMS 变化与朝向跨度；
- 窗口内物体/末端速度相关性与最小共同运动速度；
- 存在接触传感器时的可选接触证据。

当前自定义任务没有物体接触传感器，因此接触记为不可用，且不是必需条件。估计器
不会使用 `world_model.mean_gripper` 作为门控。

| 状态 | 正向转移 | 取消或恢复 |
|---|---|---|
| `DISCONNECTED` | 连接分数 ≥ 0.65 → `CANDIDATE_CONNECTED` | 保持断开 |
| `CANDIDATE_CONNECTED` | 连续三个合格样本 → `CONNECTED` | 分数 < 0.40 → `DISCONNECTED` |
| `CONNECTED` | 丢失分数 ≥ 0.70 → `CANDIDATE_LOST` | 保持连接 |
| `CANDIDATE_LOST` | 连续三个丢失样本 → `DISCONNECTED` | 丢失分数 ≤ 0.35 → `CONNECTED` |

`CANDIDATE_LOST` 在丢失确认前仍保留连接控制决策，避免单步相对位姿尖峰切换参考系。
建立关系要求共同运动与速度相关性，维持关系则不要求持续运动。夹爪张开、空抓时完全
闭合，或持续的相对平移、旋转、位置发散、朝向发散都能触发丢失。置信度为 [0, 1]
范围内、系数 0.30 的指数移动平均：断开时跟随连接证据，连接时跟随一减丢失证据。

## 冻结数据标定

标定只使用五条冻结演示，不使用任何仿真测试 seed。人工状态 4–6 只在标定阶段标识
连接正窗口；状态 0–2、8–9 提供真实张开夹爪分布。阶段标签也用于离线回放评分，但
绝不传给 `update()`。

验收标定包含 263 个完整十步正窗口。连接侧运动阈值由第 99 百分位数乘 1.25 裕量
得到，并用位置/朝向下限避免亚分辨率阈值；丢失阈值刻意比建立阈值宽。

| 量 | 建立侧 | 丢失侧 |
|---|---:|---:|
| 占用开度区间 | 0.02257–0.06267 m | 张开 ≥ 0.07129 m 或接近空夹紧 |
| 实测夹爪速度 | ≤ 0.02045 m/s | 不单独触发丢失 |
| 相对线速度 | ≤ 0.02197 m/s | ≥ 0.075 m/s |
| 相对角速度 | ≤ 0.02775 rad/s | ≥ 0.105 rad/s |
| 相对位置 RMS 标准差 | ≤ 0.00050 m | ≥ 0.002 m |
| 相对朝向跨度 | ≤ 0.005 rad | ≥ 0.030 rad |
| 共同运动速度 | ≥ 0.06166 m/s | 维持时不要求 |
| 速度相关性 | ≥ 0.79942 | 维持时不要求 |

开度下界取“夹住物体”开度第 1 百分位数的一半，因此可拒绝完全闭合的空抓：成功抓取
时方块占据夹爪，指关节总开度停在约 45.3 mm；空夹爪则可接近零。

## 离线回放

在五条演示的实测关节和位姿数组上回放，相对脚本状态 4 起点的平均建立偏移为
-8 ms（范围 -220 至 +80 ms），相对状态 7 的释放延迟为 60 ms。与状态 4–6
比较时，平均假阳性比例为 0.01845，假阴性比例为 0.00681。其中一条演示提前建立，
发生在抓取停留末期，此时被物体占用的指关节和刚性共同运动已经提供证据；所以脚本
阶段只是比较约定，不是直接物理真值。

## 机制反例与仿真冒烟

确定性单元测试覆盖四类必需机制：

1. 完全闭合但错过静止物体的夹爪始终停留在 `DISCONNECTED`；
2. 指间有物体且刚性相关搬运时，经过 `CANDIDATE_CONNECTED` 到达 `CONNECTED`；
3. 夹爪仍闭合但物体被强制移走时，经过 `CANDIDATE_LOST` 回到 `DISCONNECTED`；
4. 未抓取物体的外部运动不会建立关系。

首次 Isaac Lab 冒烟只使用 seed 6200 做留出机制检查，不参与调参。
`relation_dynamac` 在六个原始条件全部成功，连接建立延迟 120 ms、正常释放延迟
60 ms，连接状态假阳性比例约 0.009。抓取前物体平滑或突然外移都不会形成持续误连接。

两个新增瞬时扰动暴露了缺失的恢复层：

- `drop_after_grasp`：搬运中夹爪不张开，把物体瞬移 18 cm 至支撑面。估计器在
  40 ms 内撤销关系，但阶段时钟不会重新抓取，最终放置失败。
- `close_without_grasp`：夹爪闭合前把物体移走 18 cm。全过程不声明连接，最大
  置信度 0.067；策略仍继续固定阶段，因此最终放置失败。

这两项都是“检测正确、任务失败”。双向关系估计是恢复的必要条件，但不充分；恢复
还需要重规划或重新抓取转移策略。

## 复现

```bash
conda run -n env_isaaclab python scripts/analyze_relation_estimator.py \
  --data_dir data/pick_place_static/v1 \
  --output_dir outputs/single_arm_scientific/relation_calibration_v1_clean

conda run -n env_isaaclab python scripts/eval_single_arm.py --headless \
  --methods relation_dynamac \
  --conditions static smooth_object sudden_object smooth_target sudden_target \
  arm_offset drop_after_grasp close_without_grasp \
  --seeds 6200 \
  --output_dir outputs/single_arm_scientific/relation_smoke_v1_clean
```

验收运行使用实现提交 `1143f17`。标定回放源码哈希为
`23056a2b48bdca97620f545ba5c73a47e22545d62dc227e769093fcf44786a11`，
分析指纹为
`d74669c3ece5682d3c4d76ff276899867a87774f1976cb81a2359c245ca195cb`。
八个仿真试验共用源码哈希
`66fd9063d7032306e1d0ba8c5187e6248b546a2fc749567b6103772f9f6454ca`
和 schema 4；每个条件都有一对 JSON/NPZ 文件。
