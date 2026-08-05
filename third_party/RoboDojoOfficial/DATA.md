# RoboDojo 专家演示缓存

`python scripts/run.py robodojo demos` 从官方 Hugging Face 数据集按任务下载前五条 HDF5
演示到本目录；默认是论文子集，`demos --all` 覆盖动态注册表中的全部 54 个任务。大文件
不进入 Git，来源 revision 和文件清单写入同级 `data_manifest.json`。

这些 HDF5 主要含机器人状态、动作和相机数据，不含 DynaMAC/MiDiGaP 训练必需的动态物体
真值位姿。正式训练前必须在 GUI 仿真中按原布局回放，并由项目适配层同步补采任务帧；不
允许用静态初始布局冒充整段物体轨迹。
