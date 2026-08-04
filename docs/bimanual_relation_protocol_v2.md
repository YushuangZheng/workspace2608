# 双臂在线关系估计正式评测预注册协议 v2

## 1. v1 负结果与唯一实现变更

v1 在 seed 8400–8409 上完成了全部 70 条正式试验，但 `prolonged_both_hold` 只有
`7/10` 在完整两秒窗口保持物理 `both`，因此按硬门判为失败。三条失败不是估计器漏检：
固定的双臂笛卡尔目标产生内部力和载荷沉降，发送边真实解除，估计器也识别出了该变化。

v2 只修改这一项干预动作：延长双持的 100 个控制步中，左右笛卡尔目标每步跟随各自实测
末端位姿，同时保持两只夹爪闭合并冻结脚本 phase clock。物理关系真值、接触门、脚本专家
正常路径、在线估计器、标定配置和全部验收阈值均未修改。

全新开发 seed 8303–8307 的五条长双持试验为 `5/5` 物理成立、`5/5` 任务成功，真值和
推断压缩序列均精确为 `none → left_only → both → right_only → none`。平均四值准确率为
98.33%，左右平均 F1 为 98.69%/99.52%。该开发结果只用于确认动作机制，不能计入正式
成功率。

## 2. 冻结身份与数据边界

```text
实现提交：914bdca6dcb9e140f5ddba224f13251aad8fd33d
在线源码指纹：98dba7ce9dd64517ac7b0c5110b59e4a60f2a5e684ee032e254dd74f36ceb0c7
估计配置 SHA-256：9878468cfa88160eec1ba218d38bf239ce38eba406383d5ac9f04db15bb68ade
物理数据集 SHA-256：a4a39ed4837558cecaaf73e7c5db9b6ff88e7eddfb3bcf9923df862df9e65e52
v2 开发 summary SHA-256：471ed34c37efce3fd532f119358c45546dddda79434234bb0fe6c42cc646d946
```

机器可读协议是 `configs/experiments/bimanual_relation_protocol_v2.json`。正式运行后不得在
同一个 v2 名义下修改任何上述身份、干预时序、成功门或 seed。v1 的结果和报告保持只读。

## 3. 固定试验矩阵

正式 seed 为从未用于标定、开发或 v1 正式评测的 `8500–8509`。每个 seed 运行以下七种
条件，共 `10 × 7 = 70` 个相互隔离的 Isaac Sim worker：

1. `normal`：正常物理交接；
2. `receiver_miss`：接收夹爪保持张开；
3. `receiver_delayed`：在真实抓取位姿延迟闭合；
4. `giver_releases_early`：发送臂在接收接近时提前张开；
5. `receiver_grasps_then_loses`：建立短暂双持后强制接收臂张开；
6. `prolonged_both_hold`：以 v2 双臂实测位姿随动方式额外双持两秒；
7. `one_arm_paused`：接收臂持物运输前原位暂停两秒。

每次最多 1800 控制步，周期 20 ms。按条件优先、seed 次序运行；不允许失败重试、替换
seed、覆盖已有目录或只保留物理成立的子集。虽然代码变更只影响第六个条件，v2 仍重跑
完整矩阵，避免把不同源码指纹的 v1/v2 条目拼接为一个 cohort。

## 4. 信息边界与逐步证据

在线估计器每步只接收左右末端、物体位姿、左右实测指体间距及其速度和控制周期，不接收
接触力、物理关系真值、expert phase、干预条件、干预事件或未来观测。expert phase 仅由
干预器用于确定施加动作的可复现时点；接触传感只进入独立真值和事后评分。

每条 NPZ 必须保存原始估计输入、接触真值、两边状态/置信度/建立与解除分数、基础动作、
施加动作、干预事件、phase-clock 冻结标记和 source/config/experiment 指纹。

每条正式 trial 必须同时满足：

- JSON/NPZ 成对存在，逐步数组等长、有限且身份与协议一致；
- 干预由接触物理真值确认成立，不能只凭动作命令宣称成立；
- JSON 指标能从 NPZ 原始数组逐项重算；
- 推断压缩序列与条件的预注册序列完全相同；
- 四值逐步准确率不低于 95%，左边 F1 不低于 95%；
- 左边建立/解除最大延迟不超过 0.50 s；
- 右边 F1：正常/延迟/长双持/暂停不低于 97%，提前释放不低于 95%，短抓后丢失不低于 80%；
- 右边最大延迟：短抓后丢失不超过 0.10 s，其余有右边正样本的条件不超过 0.20 s；
- 空抓右边假阳性步数为 0；延迟张开期间不得推断右边；
- 延长双持的全部干预步同时保持物理和推断 `both`；
- 单臂暂停的全部干预步保持推断右边；
- `privileged_contact_used_as_estimator_input` 必须为 false。

## 5. 固定整体验收与结论边界

- 精确存在 70 个协议组合，无重复、额外或缺失 JSON/NPZ；
- 70/70 条干预物理成立，70/70 条逐条关系门全部通过；
- `normal`、`receiver_delayed`、`prolonged_both_hold`、`one_arm_paused` 各至少 `9/10`
  任务成功；
- 三个破坏持有关系的条件允许任务因预期因果后果失败，但不能把它包装为恢复成功；
- summary 成员、源码/配置哈希和逐条件计数必须与磁盘重算一致。

只有全部通过才称为 Phase 5 双臂在线关系估计正式门槛通过。通过只证明估计器在冻结任务
和七种预注册干预上的检测能力，不证明完整双臂 DynaMAC 学习策略或任意分布泛化。任何硬
门失败都将保留为 v2 负结果，且不允许在本 cohort 上继续调参。

## 6. 固定命令

正式评测只运行一次：

```bash
conda run -n env_isaaclab python scripts/eval_bimanual_relation.py --headless \
  --conditions normal receiver_miss receiver_delayed giver_releases_early \
    receiver_grasps_then_loses prolonged_both_hold one_arm_paused \
  --seeds 8500 8501 8502 8503 8504 8505 8506 8507 8508 8509 \
  --max_steps 1800 \
  --config configs/experiments/bimanual_relation_offline_v1.json \
  --output_dir outputs/bimanual_relation/formal_v2
```

正式结果只读审计：

```bash
conda run -n env_isaaclab python scripts/audit_bimanual_relation_results.py \
  --protocol configs/experiments/bimanual_relation_protocol_v2.json \
  --results_dir outputs/bimanual_relation/formal_v2
```

正式运行前必须提交本协议并创建 annotated tag `bimanual-relation-protocol-v2`。审计通过前
继续禁止完整双臂策略训练。
