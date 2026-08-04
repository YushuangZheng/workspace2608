# 双臂 DynaMAC 最小闭环

单臂闭环通过后，双臂扩展只选择了两个任务：交接（Handover）和托盘搬运
（Lift Tray）。两者都使用两台独立 Franka、16 维绝对笛卡尔动作、轻量几何附着，
并为每次采集或评测启动新的 Isaac Lab 进程。这里的附着是工程几何模拟，不代表
接触丰富的抓取物理。

## 双臂交接

旧版五演示冻结集为 `data/handover_static/v1`。其 13 个状态覆盖左臂抓取、左臂
搬运、右臂会合、载体切换、右臂放置、释放和撤离。新的四值关系标签数据集见
`data/handover_static/v2` 和 [双臂交接环境与数据骨架](bimanual_handover_setup.md)。

验证旧版工程试验：

```bash
python scripts/audit_handover_dataset.py --data_dir data/handover_static/v1
python scripts/eval_handover.py --headless --seeds 8208
```

受控试验比较独立双臂、固定交接点、固定时序静态跨臂流和 Full DynaMAC。扰动包括
左右臂偏置、交接点平移、左右臂暂停，以及单臂平滑/突发偏置。Full DynaMAC 在
载体切换时捕获实时右手—物体变换，并用该连接几何求解放置目标。

开发试验中，Full DynaMAC 在八个条件上全部成功，最终误差为 5.6–26.9 mm。
这只是单 seed 工程结果；其他基线和逐试验原始指标可在本地复现后从
`outputs/handover_minimal` 查看。

| 方法 | 成功条件数 / 8 | 主要失败 |
|---|---:|---|
| 独立双臂 | 7 | 交接点平移 |
| 固定交接 | 7 | 交接点平移 |
| 静态跨臂流 | 7 | 右臂突发偏置 |
| Full DynaMAC | 8 | 本次单 seed 试验中无失败 |

## 双臂托盘搬运

冻结数据集为 `data/lift_tray_static/v1`。九个状态覆盖同步接近、双侧抓取、抬升、
搬运、下降、释放和撤离。连接有效时，托盘位姿由两个夹爪中点驱动，因此可直接
测量双臂分歧。

```bash
python scripts/audit_lift_tray_dataset.py --data_dir data/lift_tray_static/v1
python scripts/eval_lift_tray.py --headless --seeds 10208
```

消融比较独立双臂、刻意静态的共享物体流，以及使用捕获虚拟夹爪参考系和对侧夹爪
参考系的 Full DynaMAC。报告最终放置、跨臂宽度误差、总路径、扰动恢复和推理时间。
当前矩阵仍是单 seed 试验，论文级结论前必须扩展到多 seed。

| 方法 | 成功条件数 / 5 | 静态宽度误差 | 解释 |
|---|---:|---:|---|
| 独立双臂 | 5 | 55.7 mm | 放置成功，但双侧同步较弱 |
| 静态共享物体 | 0 | 32.8 mm | 内生物体反馈使路径发散 |
| Full DynaMAC | 5 | 29.2 mm | 屏蔽共享物体回路，并使用跨臂/虚拟参考系 |

## 声明边界

这些环境可支持的窄范围结论包括：相对几何行为复现、在线参考系有效性、仅用静态
演示面对测试时扰动的零样本响应，以及自定义 Isaac Lab 任务中的动态跨臂协调。
它们不能证明论文所报的 35 个百分点增益、20 倍样本效率、完整 DynaBench、
MiDiGaP 或 Diffusion Policy 复现。正式论文表述应优先引用
[夜间研发最终报告](overnight_final_report.md) 中更新后的证据边界。
