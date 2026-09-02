# 阶段六 RLBench 集成状态

本目录只保留阶段六的非正式诊断运行工具；论文正式评测的唯一协议、调度器、结果与统计入口位于 evaluations/phase6_formal_evaluation/。

当前集成身份为：

- 执行器：stage6_hybrid_cartesian_executor_v19；
- 执行协议：rlbench-stage6-hybrid-cartesian-continuation-v23；
- 封存评测集：rlbench_eval_v2；
- 正式方法：DynaMAC V4、仅闭环进度、闭环进度＋动态角色、完整方法。

旧执行器产生的组件 pilot、逐周期诊断、预正式门控和旧协议结果已经从当前项目交付中删除，不能与正式矩阵混合。修复过程保留在开发日志与 Git 历史，不属于可执行协议或当前实验结果。

当前 v19/v23 身份的正式正常矩阵已经完成：8个任务、4种方法、每单元200条封存 episode，共32个单元和6400回合。完整方法的跨任务宏平均成功率为92.44%，冻结 DynaMAC V4为89.75%，成对 bootstrap 差值为+2.69个百分点，95%区间为[+0.38,+5.38]个百分点，通过预注册的5个百分点非劣效判定。

正式逐单元结果、成对比较、宏平均统计和校验和保存在 `evaluations/phase6_formal_evaluation/results/v1/`；PlaceCups消融末端停滞与WipeDesk覆盖率专项审查保存在 `evaluations/phase6_formal_evaluation/NORMAL_AUDIT.md`。故障与扰动矩阵尚待同一正式协议运行完成。
