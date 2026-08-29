# 对侧末端执行依赖验证

本目录记录“双臂示范同步不自动成为连续动作参考”的情况通用验证。验证输入为冻结
DynaMAC V4 模型与每任务 5 条正常示范，不使用故障数据，也不按任务名称设置规则。

核心自动化反例/正例位于：

```bash
PYTHONPATH=. pytest -q \
  tests/closed_loop/test_phase1_task_model.py::test_bimanual_peer_execution_dependency_filters_only_redundant_modes \
  tests/closed_loop/test_phase1_task_model.py::test_peer_only_directed_geometry_is_not_confused_with_constant_synchrony \
  tests/closed_loop/test_phase4_boundary_guard.py::test_cross_arm_scene_factor_resolves_each_arm_dedicated_ee_pose \
  tests/closed_loop/test_phase4_boundary_guard.py::test_local_calibration_is_independent_of_directional_guard_wait
```

真实任务重新构建后，从每个 sidecar 的
`builder_config.peer_execution_dependencies` 导出 `classification.csv`。LODO 表仅用于证明
“ held-out 动作预测增益”本身仍会受同步混淆，不能独立承担因果执行依赖判定。
