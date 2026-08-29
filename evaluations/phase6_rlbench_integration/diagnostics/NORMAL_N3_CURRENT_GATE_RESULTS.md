# 阶段六当前代码正常任务 N=3 门控

日期：2026-08-29

## 目的与口径

本轮在真实 RLBench/CoppeliaSim 物理控制链中，对8类任务各读取封存评测集索引
`0,1,2`，检查当前闭环策略、阶段四 v5 边界参数和 Stage6 通用执行器是否破坏
正常执行。每个周期真实执行“观测—信念更新—动态流—边界/恢复—IK/路径—物理步”，
不是离线示范回放。

所有正式汇总样本均满足：

- `policy_type=closed_loop_multistream`；
- `controller.profile=stage6_hybrid_cartesian_executor_v14`；
- `controller.protocol_id=rlbench-stage6-hybrid-cartesian-continuation-v16`；
- 使用封存评测索引，不重新生成或按结果筛选场景；
- 未启用成功后的诊断性策略续跑；
- 结果明确标记为小规模诊断，不是论文正式成功率。

## 结果

| 任务 | 成功 | 策略周期（索引0/1/2） | InvalidAction | 近目标停滞后 RRT 尝试 | 非线性路径成功 |
|---|---:|---|---:|---:|---:|
| HandoverItem | 3/3 | 381 / 448 / 302 | 0 | 2 | 1 |
| LiftTray | 3/3 | 121 / 106 / 107 | 0 | 9 | 0 |
| StoreBottle | 3/3 | 273 / 272 / 298 | 0 | 44 | 5 |
| SweepDust | 3/3 | 137 / 136 / 134 | 0 | 46 | 0 |
| OpenMicrowave | 3/3 | 195 / 557 / 161 | 0 | 37 | 0 |
| PlaceCups | 3/3 | 185 / 184 / 183 | 0 | 28 | 0 |
| StackWine | 3/3 | 165 / 165 / 165 | 0 | 0 | 0 |
| WipeDesk | 2/3 | 238 / 255 / 239 | 0 | 0 | 0 |
| **合计** | **23/24（95.83%）** | **5407** | **0** | **166** | **6** |

WipeDesk 索引1在策略完整执行和10个最终静置物理步后仍有随机污点未被清除，结果为
`policy_complete_after_final_settling`。完全相同的封存索引由冻结 DynaMAC V4 路径执行时
同样失败，并产生9个 `InvalidAction`。RLBench 的该任务随机生成50个污点，而当前低维任务
输入只提供海绵位姿、不提供污点分布；因此这条证据说明当前闭环方法没有相对基线引入该
失败，但不能把失败样本改记为成功，也不能据此宣称方法优于基线。

## 本轮发现并收口的通用偏差

1. 阶段四旧运行配置给所有边界附加了与数据无关的0.10秒全局最短连续周期。v5改为
   每个边界取“正常边界前最长偶发就绪连续段 + 1”，再检查正常末端保持能否持续满足。
   40/40个边界完成标定，200/200个边界×正常示范通过，边界前提前放行0次。
2. LiftTray 恢复曾在目标关系已经由可靠运动证据稳定满足后继续追逐离线 LINK 锚点。
   当前实现把锚点视为建立关系的动作模板，而不是关系之外的额外完成目标；关系稳定达到
   后即可进入重入。若主动探测已使机械臂离开起点，仍必须先原路返回。
3. StoreBottle 索引2曾在距目标约6.4 cm处停滞。执行器旧顺序让碰撞放宽直线抢在
   碰撞感知绕行之前，且近目标不会调用 RRT。当前顺序保留首次近目标的快速局部求解；
   只有同一目标被真实物理反馈确认停滞、局部求解层级耗尽后，才尝试碰撞感知
   RRTConnect，最后才使用碰撞放宽直线。StoreBottle 三条合计5次非线性路径成功，
   3/3完成且0个非法动作。该规则不读取任务名、seed、StateId或物体身份。

## 自动化验证

- Stage6 执行器定向测试：38/38通过，包含“首次近目标不调用RRT”和“确认停滞后先用
  碰撞感知RRT、成功时不再走碰撞放宽直线”的正反例；
- 阶段一至六及 RLBench 相关回归：487项通过、4项跳过、0失败，用时268.51秒；
- 最终全仓回归：611项通过、4项跳过、0失败，用时319.75秒；
- Handover 另有诊断性成功后续执行证据：右臂稳定 linked 并返回 TASK 后，左臂才越过
  释放边界，策略随后完成。该证据不改变本表正式成功口径。

## 原始结果与 SHA256

```text
0388e813ed806c5d8362a26bc4968b9c5db9c35e9dfbe203460f3f5db819faf9  integrations/rlbench/results/diagnostics/phase6_normal_handover_i000_i002_current_v21_h650.json
e40c25255d242946f6116cd5e68991ed547fa271d3cc0521d4d576e5fe17fcf9  integrations/rlbench/results/diagnostics/phase6_normal_lift_tray_i000_i002_current_v21_h800.json
8df6d64ce8a5e35f7cc79fd0e2653cbe5f1640b29cbe8061cfc150e7799a1ff9  integrations/rlbench/results/diagnostics/phase6_normal_store_bottle_i000_i002_current_v21_h800.json
5576d998016ac8c13bd19308ccd007c70114ce489f7a09e61ae9c2b92e06b364  integrations/rlbench/results/diagnostics/phase6_normal_sweep_dust_i000_current_v21_h800.json
70f5da6d187fc98f3d9084ccad29c24054f43db3b83c48db28086665f0a9a04d  integrations/rlbench/results/diagnostics/phase6_normal_sweep_dust_i001_i002_executor_v20_h800.json
c146bd21c48296598cee017cd75f1feba015290086e94af314cd1b0a52671897  integrations/rlbench/results/diagnostics/phase6_normal_open_microwave_i000_current_v21_h800.json
fe13c8055806bc90b1239ab47dfc4e5f7b999d7b8ea7dc0646340258363667d6  integrations/rlbench/results/diagnostics/phase6_normal_open_microwave_i001_i002_executor_v20_h800.json
82e638ea5d6492867ceeb9349d5ec8fceff64ed8887396875ff035980376c824  integrations/rlbench/results/diagnostics/phase6_normal_place_cups_i000_current_v21_h800.json
b792eb17f5049a14b56d43f53cb41d975f35681bda152eb5db87c2422558d96c  integrations/rlbench/results/diagnostics/phase6_normal_place_cups_i001_i002_executor_v20_h800.json
2505c85dc979ee826cf5d6d6cd03c66ecbe1314c438c61a55853593a949abf1a  integrations/rlbench/results/diagnostics/phase6_normal_stack_wine_i000_current_v21_h800.json
d4d908f63d9feff985e240502d3c2f6e7fe20307a1c67c712783a9268be2ef80  integrations/rlbench/results/diagnostics/phase6_normal_stack_wine_i001_i002_executor_v20_h800.json
c6ceb079bd9e5983222e4e380b5eea298e97466b2b079290f8ca958a9a8609ae  integrations/rlbench/results/diagnostics/phase6_normal_wipe_desk_i000_current_v21_h800.json
39cd85899cca0e4e27080e590d00e80d9c3c4566d9e06fdf395ce45baa814c5a  integrations/rlbench/results/diagnostics/phase6_normal_wipe_desk_i001_i002_executor_v20_h800.json
2d1fff5139f71a958cae323bfd290a8138b4d79343b3a9c0f1bdbb9afca51911  integrations/rlbench/results/diagnostics/phase6_normal_wipe_desk_i001_dynamac_v4_baseline_v1_h800.json
```

逐周期 JSONL 位于各结果对应的诊断目录。

## 结论边界

本轮支持“当前实现可以进入故障/扰动评测”，不支持以下更强结论：完整200样本成功率、
统计显著优于DynaMAC V4、故障恢复在真实仿真中已经有效，或阶段六已经最终验收。后续必须
冻结本轮配置，按计划运行时间停滞、关系失配、抓取失败和意外掉落的同集消融对比。
