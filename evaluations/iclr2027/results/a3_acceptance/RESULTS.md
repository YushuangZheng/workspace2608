# A3 服务器 A 方法接入与 Horizon-3 验收

状态：**PASS（development-only）**  
机器可读记录：`A3_ACCEPTANCE.json`

本阶段只验收服务器 A 负责的方法接入、统一恢复接口、正文消融、正常边界运行参数与 Horizon-3 资产；没有运行 sealed test，也不产生论文正式成功率。

## 已完成内容

- M1 `DynaMAC + Restart`：使用前一目标需求与下一周期真实位移形成因果 no-progress 规则。
- M2 `Trajectory-Likelihood + Retry`：只读取当前活动流、PoE 权重和连续 SE(3) 马氏残差；完整 NLL/logdet 仅留审计，正式阈值等待 A-only 正常校准。
- M1/M2/M6 与 Generic-retry 消融共用一套 Skill-Retry：最多1次、400周期，不复位模拟器、不传送实体。
- M5 Full 与 M6 Ours-Monitor+Retry 已导出冻结方法配置；M6直接读取 M5 的持久失配报警，不建立第二个本文检测器。
- Motion-only、Open-loop progress、Generic retry 三个正文消融已实现为明确的权限开关。
- Horizon-3 共8个任务 level；`per-stage` 与 `single-event` 各1600个冻结 episode，均具备 DynaMAC 与闭环模型。

## 正常边界参数

只使用每任务5条正常成功示范完成 Main-10 的边界运行参数构建：60/60 个边界完成标定，300/300 个“边界×示范”检查通过；1个真实联合事务组通过5/5条示范检查。控制周期固定为0.05秒，未读取故障数据。H 的分布为：43个边界取1周期、11个取2周期、5个取3周期、1个取7周期。

## A3 中纠正的两个通用缺口

1. 恢复后的任务重入不是 episode reset。现在只重置进度对齐和动作引用，保留此前由有效运动证据确认的关系方向及正式事件生命周期。
2. 离线动作相关性已取消动作权限的参考系仍可提供状态解释，但其低激励 `Unknown` 不再冻结真正的执行流；一旦后续可靠判成与期望相反的关系，统一失配与恢复逻辑仍照常生效。

第二项修复消除了新任务上的因果死锁：同一 StackCups development 样本中，M5/M6 原先在 `k0:t0` 停满1000周期；修复后两者均在222周期成功完成。

## 真实 development 冒烟

| 方法/条件 | 任务 | 结果 | 周期 | 说明 |
|---|---|---:|---:|---|
| M1 nominal | CloseJar | 成功 | 162 | 无误报警 |
| M1 actuation delay | CloseJar | 成功 | 196 | 第34周期报警并执行1次 Skill-Retry |
| M2 未校准 shadow | CloseJar | 成功 | 162 | 阈值为空，不允许干预 |
| M5 Full | StackCups | 成功 | 222 | 完整闭环执行链 |
| M6 Monitor+Retry | StackCups | 成功 | 222 | 监控输出接入共享重试链 |

另有一个 CloseJar development 样本在接触段落到五示范运动支持之外，完整方法因此保守 HOLD，而开放时钟 DynaMAC 在该样本上完成。该现象不属于接口或基础设施错误，本阶段未用单一样本调阈值或增加任务特例；它保留给后续预先规定的 development dry-run/性能审计判断。

## 校验

- 相关阶段二至五、边界和 A3 方法测试：`141 passed`。
- 8份服务器 A 方法配置全部可加载。
- `evaluations/iclr2027`、闭环核心与 RLBench ICLR 集成代码通过 `compileall`。
- A2 的 B 交付索引已重新生成并审计：接口10个文件，failure-train 4001个文件；均不包含 normal calibration 或 sealed test。

## B 交付状态

按用户决定，A2 接口和 failure-train 数据**尚未发送给 B**。首次发送前若 A3 或后续未接触 sealed test 的开发检查发现确证的数据/协议缺陷，允许修复规范源文件并重新生成两份索引与 SHA256；B 一旦开始训练，任何变更都必须显式版本化，不能静默替换。
