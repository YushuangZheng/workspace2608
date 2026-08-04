# 双臂在线关系生命周期估计器

## 结论

Phase 5 的第一步已经建立一个不读取任务阶段、接触真值或未来状态的双臂在线关系估计器。
它分别估计 `left—object` 和 `right—object` 两条边，再组合为
`none / left_only / both / right_only` 四值关系图。

在冻结物理数据集 `handover_physical/v1` 上，先用 seed 8200–8209 标定阈值，再对互斥的
seed 8210–8219 做离线开发回放。最终开发版十条回放的四值逐步准确率均值为 **98.30%**；
左、右边 micro F1 分别为 **98.54%** 和 **99.44%**。真值含 901 个 `both` 步，估计器输出
879 个。

同一估计器随后接入真实接触仿真。在最终开发 seed 8302 的七个条件中，所有干预均由物理
真值确认成立，七条推断关系序列逐段等于对应物理序列。正式 v1 因三条长双持干预没有
维持真实双持而失败；只修复该动作机制后，正式 v2 的七条件 × 十个未见 seed 共 70 条
全部通过。完整双臂策略训练尚未开始。

## 机制与输入边界

`BimanualOnlineRelationEstimator` 内部运行两个互不覆盖状态的
`OnlineRelationEstimator`。每条边都有独立的四状态滞回机：

```text
DISCONNECTED -> CANDIDATE_CONNECTED -> CONNECTED
                                      -> CANDIDATE_LOST -> DISCONNECTED
```

左右连接布尔值按下表组合：

| 左边 | 右边 | 四值标签 |
|---:|---:|---|
| 0 | 0 | `none` |
| 1 | 0 | `left_only` |
| 1 | 1 | `both` |
| 0 | 1 | `right_only` |

每步允许输入只有：

- 左右末端六维位姿；
- 物体六维位姿；
- 左右实测指体间距和由相邻采样计算的速度；
- 当前控制周期。

估计证据包括夹爪占用区间、相对平移/旋转刚性、窗口稳定性、共同运动、速度相关性和
末端—物体距离。连接开度的 q01–q99 是满分平台，而不是只让中位数满分；双臂配置中的
运动学解除还要求瞬时相对运动与滑窗几何漂移共同成立，避免载荷转移尖峰伪造断裂。夹爪
张开、空闭合或末端—物体距离脱离仍可单独解除。

运行时函数没有 `contact`、`relation_label`、`state/phase` 或未来窗口参数。
物理接触导出的 `left_connected/right_connected` 只在离线标定和结果评分时使用。

## 数据划分与标定

源数据是只读冻结数据集 `data/handover_physical/v1`：

```text
数据集 SHA-256:
a4a39ed4837558cecaaf73e7c5db9b6ff88e7eddfb3bcf9923df862df9e65e52

标定：seed 8200–8209
开发回放：seed 8210–8219
```

标定只在连续真值连接窗口中统计运动特征分位数，并在连接/断开样本中确定夹爪占用与
释放区间。接触真值因此属于“训练监督”，不是运行时传感输入。左右臂使用不同阈值，
因为两只夹爪在物理交接中的闭合宽度、末端—物体相对几何和传感噪声并不相同。

这个 10/10 划分是在 Phase 5 开发时声明并执行，未在代码冻结前预注册，所以只能用于
机制开发和后续协议设计。初始回放暴露出接收边偏保守；随后只依据标定子集的连接开度
q01–q99 构造占用平台，并依据在线开发 seed 8300 的载荷转移反例把单通道速度尖峰改为
“瞬时断裂与滑窗漂移共同成立”。最终指标是在这些开发选择之后重算，不能当作未见测试。

存储形状 `(T, 2, 1, 3)` 的指体位置会先严格化为每步唯一指间距。提交的配置
`configs/experiments/bimanual_relation_offline_v1.json` 保存两臂独立阈值、标定 seed 和
冻结数据哈希，并支持验证后反序列化。

## 离线开发结果

| 指标 | 结果 |
|---|---:|
| 四值逐步准确率均值 | 98.30% |
| 四值逐步准确率范围 | 97.86%–98.49% |
| 左边 TP / FP / FN | 4495 / 9 / 124 |
| 左边 micro precision / recall / F1 | 99.80% / 97.32% / 98.54% |
| 左边单条最低 F1 | 98.02% |
| 右边 TP / FP / FN | 5220 / 29 / 30 |
| 右边 micro precision / recall / F1 | 99.45% / 99.43% / 99.44% |
| 右边单条最低 F1 | 99.24% |
| 真值 / 推断 `both` 步数 | 901 / 879 |

初始三角占用版的四值准确率为 94.33%，右边 F1 为 95.41%，只推断 568 个 `both` 步；
最终版的提升来自明确的机制修复，不是改评分标签。后续仍须同时报告 precision、recall、
建立/解除延迟和异常误触发，不能只选取改善后的四值准确率。

## 真实物理在线开发结果

在线评测器保存左右末端、物体、指体位置/间距/速度、接触力真值、基础与施加动作、干预
事件、两边状态/置信度/建立分数/解除分数。接触力只进入独立 `PhysicalRelationTracker`，
没有进入在线估计器。任务是否成功与干预是否物理成立分开统计。

最终源码下的开发 cohort 只包含 seed 8302，每个条件一次：

| 条件 | 物理真值序列与推断序列 | 四值准确率 | 左 F1 | 右 F1 | 任务结果 |
|---|---|---:|---:|---:|---|
| `normal` | `none → left_only → both → right_only → none` | 98.49% | 98.69% | 99.52% | 成功 |
| `receiver_miss` | `none → left_only → none` | 98.93% | 98.69% | 无正样本、0 FP | 预期失败 |
| `receiver_delayed` | `none → left_only → both → right_only → none` | 98.77% | 98.93% | 99.74% | 成功 |
| `giver_releases_early` | `none → left_only → none → right_only → none` | 98.14% | 97.50% | 99.24% | 预期失败 |
| `receiver_grasps_then_loses` | `none → left_only → both → left_only → none` | 98.58% | 98.69% | 90.91% | 预期失败 |
| `prolonged_both_hold` | `none → left_only → both → right_only → none` | 98.61% | 98.93% | 99.60% | 成功 |
| `one_arm_paused` | `none → left_only → both → right_only → none` | 98.69% | 98.69% | 99.68% | 成功 |

短暂接收关系只有约 0.44 s，所以两步建立误差与两步解除误差会把右边 F1 降至 90.91%；
但完整序列和边的建立/解除方向均正确。正常条件两边最大匹配延迟为 220/60 ms；最终延迟
条件的右边建立/解除延迟为 20/40 ms。空抓全过程右边真值和推断均不连接。

开发过程中还保留两个负结果：初始占用版在正常条件的右边建立晚 1.22 s；第一版延迟
干预错误地让脚本专家在夹爪张开时越过抓取状态，造成 96 步单指接触假阳性。前者推动
占用平台修复，后者通过在真实抓取位姿冻结时钟、延迟张开 1 s、闭合稳定 1 s 后再进入
transfer 修正。二者均不计入最终开发指标。

## 机制反例测试

纯 NumPy 单元测试覆盖五类反事实，不依赖脚本 phase：

1. 接收臂空抓或延迟时，不提前产生右边；随后真实共同运动可建立 `both`；
2. 发送臂过早释放时，显式暴露 `none` 间隔；
3. 接收臂建立后丢失物体时，只解除右边并回到 `left_only`；
4. 长时间双持不因固定计时器自动解除；
5. 单臂暂停并与物体分离时，只解除该臂自己的边。

这些测试验证状态机语义，不代替 Isaac Sim 中有接触动力学的在线扰动试验。

## 复现与证据身份

```bash
conda run -n env_isaaclab python scripts/analyze_bimanual_relation_estimator.py \
  --data_dir data/handover_physical/v1 \
  --calibration_seeds 8200 8201 8202 8203 8204 8205 8206 8207 8208 8209 \
  --evaluation_seeds 8210 8211 8212 8213 8214 8215 8216 8217 8218 8219 \
  --output_dir outputs/bimanual_relation/offline_dev_v4
```

脚本拒绝覆盖已有输出，逐条保存真值/推断标签、左右置信度和逐边指标。当前产物身份为：

| 产物 | SHA-256 |
|---|---|
| 离线源码指纹 | `a46a1b84a8c6ceab1edee6f73abed8d312953497f93de4ecff6f495633fa7862` |
| 离线 `summary.json` | `cbc52a671de860a2f58e87c015600ff00a8a398ff327f1b99689fd9fd2947d9d` |
| 离线 `config.json` | `ec8d431bd2602449d3326a0f64b16fc019a2c889427f502caacae4f3e0f40cd5` |
| 离线 `calibration.json` | `e08ce762463a043722a30e47199c74f34f09d137b1eeeb1015d61c945398d7c9` |
| 提交配置 | `9878468cfa88160eec1ba218d38bf239ce38eba406383d5ac9f04db15bb68ade` |
| 在线源码指纹 | `95d6e5560ad3af6a089b32a946a2e79b658516b08a7f871d0e052bf0c5b1f23b` |
| 在线开发 `summary.json` | `f772ba841e2e2a24a9eb0a60db38d4c3fe94f56129479002194554052805a9e7` |

最终在线开发命令为：

```bash
conda run -n env_isaaclab python scripts/eval_bimanual_relation.py --headless \
  --conditions normal receiver_miss receiver_delayed giver_releases_early \
    receiver_grasps_then_loses prolonged_both_hold one_arm_paused \
  --seeds 8302 --max_steps 1800 \
  --output_dir outputs/bimanual_relation/online_dev_final_seed8302
```

## 正式门槛与下一阶段

正式 v2 使用 seed 8500–8509，七类条件共 70 条，得到 `70/70` 物理条件成立、`70/70`
预注册推断序列精确以及四个应成功条件 `40/40` 任务成功。硬审计从 NPZ 重算全部逐条门
并返回通过；加权四值准确率为 98.46%，左右 micro-F1 为 98.56%/99.42%。完整数字和 v1
到 v2 的单一动作修复见 [v2 正式报告](bimanual_relation_report_v2.md)。

因此目前的准确表述是：**双臂在线关系估计已通过冻结任务和七类预注册干预上的未见
seed 正式验收。** 这解除 Phase 6 的前置检测阻塞，但不等于关系估计已经产生任务恢复，
更不等于双臂学习策略完成。下一阶段必须分别报告普通任务、检测、关系门控和恢复能力，
privileged 物理关系继续只作评分或明确标注的 Oracle 消融。
