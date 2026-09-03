# 阶段六 RLBench 集成状态

本目录只保留阶段六的非正式诊断运行工具；论文正式评测的唯一协议、调度器、结果与统计入口位于 evaluations/development/phase6_formal_evaluation/。

当前集成身份为：

- 执行器：stage6_hybrid_cartesian_executor_v19；
- 执行协议：rlbench-stage6-hybrid-cartesian-continuation-v23；
- 封存评测集：rlbench_eval_v2；
- 正式方法：DynaMAC V4、仅闭环进度、闭环进度＋动态角色、完整方法。

旧执行器产生的组件 pilot、逐周期诊断、预正式门控和旧协议结果已经从当前项目交付中删除，不能与正式矩阵混合。修复过程保留在开发日志与 Git 历史，不属于可执行协议或当前实验结果。

当前 v19/v23 身份的正式矩阵已经完成：正常6400回合、平滑动态无故障1600回合、四类故障6400回合，共192个单元和14400回合。最终深度审计通过，固定运动计划身份、动态背景、故障先后顺序和恢复时间戳均无违规，故障注入器修改策略内部状态的次数为0。

正常条件下，完整方法的跨任务宏平均成功率为92.13%，冻结DynaMAC V4为89.75%，成对bootstrap差值为+2.38个百分点，95%区间为[+0.31,+4.94]个百分点，通过预注册的5个百分点非劣效判定。平滑动态无故障条件下二者分别为90.50%和93.50%。四类故障合并后，完整方法为1098/1600（68.63%），冻结基线为621/1600（38.81%），提高29.81个百分点。

正式逐单元结果、Wilson区间、成对比较、Holm校正、宏平均bootstrap、完整审计和校验和保存在 `evaluations/development/phase6_formal_evaluation/results/v2/`；PlaceCups消融末端停滞与WipeDesk覆盖率专项审查保存在 `evaluations/development/phase6_formal_evaluation/NORMAL_AUDIT.md`。旧`results/v1/`汇总已清除，避免与最终Handover最小受影响重跑结果混用。
