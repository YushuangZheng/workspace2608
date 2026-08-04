# 双臂关系门控恢复正式评测预注册协议 v1

## 1. 研究问题

本协议检验：在重力、碰撞、摩擦和真实双指夹持的物理双臂交接中，在线关系生命周期估计
能否驱动一个独立恢复监督层，使接收空抓、共同持物期间丢失，以及发送臂释放后的接收丢失，
恢复为最终任务成功；同时不在正常任务中误触发，不把短暂关系抖动升级为不必要的撤离重抓。

本协议不训练新的学习策略。四种方法共享冻结物理专家，比较的是无关系监督、只门控、在线
关系恢复和 Oracle 关系恢复。正式通过只证明关系触发式运行时恢复机制在本冻结任务和故障
分布上成立，不证明完整双臂 DynaMAC 学习策略或任意分布泛化。

## 2. 冻结身份

```text
实现提交：4214730a1d74bef353bac28629adab82fbe03a87
源码指纹：eff06caaedc64725a69ef148d8408f57bd167c0b0eba0e116ea4ce23d46cf1c9
关系配置：9878468cfa88160eec1ba218d38bf239ce38eba406383d5ac9f04db15bb68ade
恢复配置：c62c9d22ee502f4af5fbbeea151fa3350e585e2b19b614c88f642b7eeb6007b5
物理数据：a4a39ed4837558cecaaf73e7c5db9b6ff88e7eddfb3bcf9923df862df9e65e52
最终开发：ad779c50a41480710b99314b017c69bf02a620f8ea39d9e221116c71ab99962b
```

机器可读协议为 `configs/experiments/bimanual_recovery_protocol_v1.json`。关系配置沿用已通过
Phase 5 v2 的冻结在线估计器；恢复配置固定 0.50 s 空抓验证、0.30 s 丢失确认、0.10 s
关系验证、最多两次重抓、双臂位置目标每步 0.05 m 限幅。正式结果产生后，不得在 v1 名义
下修改实现、配置、seed、阈值或结果成员。

## 3. 数据隔离与固定矩阵

关系标定使用 seed 8200–8209；关系估计开发使用 8300–8307；既往关系正式评测使用
8400–8409 和 8500–8509；恢复开发使用 8600–8619。正式恢复只使用此前从未运行的
8700–8709。

每个 seed 按固定顺序运行四种方法和五种条件，共 `4 × 5 × 10 = 200` 条独立 Isaac Sim
worker。每条最多 2200 步，控制周期 20 ms。禁止失败重试、替换 seed、覆盖目录、删除
不利条目或把不同源码指纹的开发结果拼入正式 cohort。

四种方法：

1. `clocked_expert`：原时钟专家，无关系门和恢复；
2. `relation_gate`：使用在线估计关系冻结时钟和阻止不安全释放，但无几何恢复；
3. `relation_recovery`：使用在线估计关系驱动完整恢复图；
4. `oracle_relation_recovery`：仅用当步物理真值替换关系判断，几何动作与在线恢复相同。

五种条件：正常、首次空抓、0.12 s 短丢失、共同持物期 0.8 s 强丢失，以及 giver 已真实
释放后的 0.8 s 强丢失。每种故障必须由接触物理真值证明实际成立；释放后条件还必须证明
故障开始时左边已经断开。

## 4. 信息边界与原始证据

在线方法每步只能读取当前左右末端、物体位姿、左右实测指间距及其速度、控制周期和在线
关系状态，不读取接触力、物理关系真值、故障名称或未来观测。接触传感只进入独立物理真值
和事后评分。Oracle 只读取当前物理关系布尔值，不读取未来关系，也不从真值生成动作目标。

每条 NPZ 必须保存：原始估计输入、接触真值、在线置信度和分数、推断状态、控制实际使用
的关系状态、基础/监督/施加动作、恢复状态和转移、故障事件、phase-clock 冻结、终端物体/
目标/末端位姿、专家与环境终态，以及 source/config/dataset/experiment 指纹。

硬审计必须从 NPZ 独立重算任务结论、故障成立、四值关系指标、释放安全、恢复起止和时间、
giver 保持、接收边再次丢失、路径、重抓次数、动作跳变和末端速度。summary 与逐条 JSON
必须逐字一致，JSON/NPZ 集合必须精确为预注册笛卡尔积。

## 5. 逐条硬门

所有关系监督方法均不得在接收边未建立时释放。在线和 Oracle 恢复的每个故障 trial 还必须：

- 最终任务成功且安全释放，关系在故障后真实重建；
- 空抓、共同持物强丢失、释放后强丢失必须执行几何恢复；
- 短丢失必须取消，不得进入撤离或重抓；
- giver 尚连接的恢复必须全程保持 giver 边；giver 已释放后不得伪造 giver 保持要求；
- 恢复后到计划释放前不得再次物理丢失接收边；
- 最多两次重抓；恢复时间上限依次为 2.5/0.8/3.0/3.5 s；
- 恢复期间施加动作和监督动作的位置目标最大相邻跳变均不超过 0.08 m；
- 恢复期间实测末端最大速度不超过 0.75 m/s；
- 在线方法每条四值关系准确率不低于 94%。

在线和 Oracle 的正常 trial 必须零恢复触发。正常任务成功允许 1/10 随机物理失败，但任何
误触发都直接使该 trial 失败硬门。

## 6. 整体硬门

- 精确存在 200 对 JSON/NPZ，200/200 worker 返回 0，200/200 条条件由物理真值成立；
- 时钟专家和关系门控在正常、短丢失上各至少 9/10 成功；
- 在线恢复和 Oracle 在五种条件上各至少 9/10 成功；
- 在线恢复在三种强故障相对时钟专家的 30 个配对中至少赢 27 个，且不得出现时钟专家成功
  而在线恢复失败的配对；
- 在线恢复与 Oracle 在全部 40 个故障 trial 上的成功数差距不超过 2；
- 全部逐条硬门、信息边界、身份和 summary 重算均通过。

时钟专家在强故障上是否失败不是通过协议的先验必要条件；若它意外成功，将如实计入，并
降低配对胜场。关系门控在强故障上允许安全阻塞，但不能把阻塞记为恢复成功。

## 7. 固定命令

协议提交并创建 annotated tag `bimanual-recovery-protocol-v1` 后，只运行一次：

```bash
conda run -n env_isaaclab python scripts/eval_bimanual_recovery.py --headless \
  --run_kind formal \
  --methods clocked_expert relation_gate relation_recovery oracle_relation_recovery \
  --conditions normal receiver_miss_once receiver_brief_loss receiver_loss_once \
    receiver_loss_after_release \
  --seeds 8700 8701 8702 8703 8704 8705 8706 8707 8708 8709 \
  --max_steps 2200 \
  --recovery_config configs/experiments/bimanual_recovery_online_v1.json \
  --output_dir outputs/bimanual_recovery/formal_v1
```

随后只读审计：

```bash
conda run -n env_isaaclab python scripts/audit_bimanual_recovery_results.py \
  --protocol configs/experiments/bimanual_recovery_protocol_v1.json \
  --results_dir outputs/bimanual_recovery/formal_v1
```

任何硬门失败都保留为 v1 负结果，不得用正式结果继续调参。审计通过前继续禁止完整双臂
黑盒策略训练。
