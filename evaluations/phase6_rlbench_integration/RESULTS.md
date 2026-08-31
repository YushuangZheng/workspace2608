# 阶段六 RLBench 集成状态

本目录只保留阶段六的非正式诊断运行工具；论文正式评测的唯一协议、调度器、结果与统计入口位于 evaluations/phase6_formal_evaluation/。

当前集成身份为：

- 执行器：stage6_hybrid_cartesian_executor_v18；
- 执行协议：rlbench-stage6-hybrid-cartesian-continuation-v22；
- 封存评测集：rlbench_eval_v2；
- 正式方法：DynaMAC V4、仅闭环进度、闭环进度＋动态角色、完整方法。

旧执行器产生的组件 pilot、逐周期诊断、预正式门控和旧协议结果已经从当前项目交付中删除，不能与正式矩阵混合。修复过程保留在开发日志与 Git 历史，不属于可执行协议或当前实验结果。

正式启动前保留的当前机制结论仅包括：

- SweepDust 的 progress_dynamic_roles 与 full 在封存索引0至9均为10/10；
- OpenMicrowave 原4条正式 LINK 入口结构化失败修复后为4/4成功，周期数161、161、196、161，且0 InvalidAction；
- LiftTray 封存索引0至4的完整方法与共享执行器基线具有相同成功集合，未为两条共同物理抓取失败增加任务专属规则；
- 配置、动作事务和正式协议清理后的紧凑自动化回归通过。

这些内容只证明当前实现具备进入正式矩阵的条件，不是论文成功率结果。正常与故障统计必须由当前正式协议完整生成。
