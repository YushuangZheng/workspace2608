# 双臂在线关系估计正式评测预注册协议 v1

## 1. 前置证据与冻结边界

真实物理脚本专家 v3 已在 seed 8000–8019 上达到 `20/20`，独立物理数据集
`handover_physical/v1` 已冻结。双臂在线关系估计器使用数据 seed 8200–8209 标定，并在
冻结数据的 8210–8219 以及在线开发 seed 8300–8302 上完成机制开发。

正式实现固定为：

```text
实现提交：e0668acbc9e5560bdef11ada4d17eeebd9ff4186
在线源码指纹：95d6e5560ad3af6a089b32a946a2e79b658516b08a7f871d0e052bf0c5b1f23b
估计配置 SHA-256：9878468cfa88160eec1ba218d38bf239ce38eba406383d5ac9f04db15bb68ade
物理数据集 SHA-256：a4a39ed4837558cecaaf73e7c5db9b6ff88e7eddfb3bcf9923df862df9e65e52
```

协议机器可读版本是 `configs/experiments/bimanual_relation_protocol_v1.json`。正式结果运行后
不得再修改估计阈值、状态机、干预时序、成功门或 seed，并继续称为 v1。

## 2. 固定试验矩阵

正式 seed 为 `8400–8409`，与标定、冻结数据、物理专家正式集和全部在线开发 seed 互斥。
每个 seed 运行七种条件，共 `10 × 7 = 70` 个独立 Isaac Sim worker：

1. `normal`：正常物理交接；
2. `receiver_miss`：接收夹爪保持张开，验证空抓不虚构右边；
3. `receiver_delayed`：在真实接收抓取位姿张开延迟 1 s，再闭合稳定 1 s 后继续；
4. `giver_releases_early`：发送臂在接收接近时提前张开，允许出现无持有者间隔；
5. `receiver_grasps_then_loses`：物理 `both` 持续约 0.4 s 后强制接收臂张开；
6. `prolonged_both_hold`：在 transfer 中额外保持双持 2 s；
7. `one_arm_paused`：接收臂持物运输前原位暂停 2 s。

每次最多 1800 控制步，控制周期 20 ms。控制器按条件优先、seed 次序运行；不允许失败
重试、替换 seed 或只保留物理成立的子集。目标目录已存在时评测器必须在启动仿真前拒绝
覆盖。

## 3. 信息边界与逐步证据

在线估计器每步只接收左右末端、物体位姿、左右实测指体间距/速度和控制周期，不接收：

- 指体接触力或 `PhysicalRelationTracker` 输出；
- expert `state/phase`；
- 干预条件或事件；
- 未来观测。

expert phase 只供干预器在可复现时点施加动作。接触力只进入独立物理真值和离线评分。
每条 NPZ 必须保存原始估计输入、接触真值、左右状态/置信度/建立分数/解除分数、基础动作、
施加动作、干预事件、phase-clock 是否冻结、source/config/experiment 指纹。

## 4. 固定逐条验收

每条正式 trial 必须同时满足：

- worker JSON/NPZ 存在、逐步数组等长、数值有限、身份字段与协议一致；
- 干预由接触物理真值确认成立，不能只依据动作命令声称条件成立；
- JSON 指标可由 NPZ 原始数组逐项重算；
- `privileged_contact_used_as_estimator_input` 为 false；
- 推断压缩序列与该条件的预注册序列完全相等；
- 四值逐步准确率不低于 95%，左边 F1 不低于 95%；
- 左边所有真值建立/解除的最大匹配延迟不超过 0.50 s；
- 除空抓外，右边 F1 门为：正常/延迟/长双持/暂停 97%，提前释放 95%，短抓后丢失 80%；
- 对应右边最大匹配延迟为：短抓后丢失 0.10 s，其余有正样本条件 0.20 s；
- 空抓条件右边假阳性步数必须为 0；
- 延迟接收的强制张开区间不得推断右边；
- 延长双持的全部干预步必须推断 `both`；
- 单臂暂停的全部干预步必须保持推断右边。

短抓后丢失只有约 0.4 s 正样本，F1 对两三个迟滞步高度敏感，因此预注册门为 80%；但
完整 `left_only → both → left_only` 序列和 0.10 s 双向延迟仍是硬门，不因窗口短而豁免。

## 5. 固定整体验收

- 精确存在 70 个协议组合，无重复、额外或缺失 JSON/NPZ；
- 70/70 条干预均物理成立，70/70 条逐条关系门全部通过；
- `normal`、`receiver_delayed`、`prolonged_both_hold`、`one_arm_paused` 各至少 9/10 任务成功；
- `receiver_miss`、`giver_releases_early`、`receiver_grasps_then_loses` 的任务失败属于预期因果
  后果，不作为估计器关系失败，也不允许被包装成恢复成功；
- summary 的成员、源码/配置哈希和逐条件计数与磁盘重算一致。

只有全部通过才称为 Phase 5 正式关系估计门槛通过。任何一条硬门失败都保留为 v1 负结果；
若要修复，必须创建 v2、使用新 seed 并解释修复依据。

## 6. 固定命令

正式评测只运行一次：

```bash
conda run -n env_isaaclab python scripts/eval_bimanual_relation.py --headless \
  --conditions normal receiver_miss receiver_delayed giver_releases_early \
    receiver_grasps_then_loses prolonged_both_hold one_arm_paused \
  --seeds 8400 8401 8402 8403 8404 8405 8406 8407 8408 8409 \
  --max_steps 1800 \
  --config configs/experiments/bimanual_relation_offline_v1.json \
  --output_dir outputs/bimanual_relation/formal_v1
```

正式结果只读审计：

```bash
conda run -n env_isaaclab python scripts/audit_bimanual_relation_results.py \
  --protocol configs/experiments/bimanual_relation_protocol_v1.json \
  --results_dir outputs/bimanual_relation/formal_v1
```

正式运行前必须先提交本协议并创建标签 `bimanual-relation-protocol-v1`。完整双臂学习策略
在正式审计通过前继续禁止启动。
