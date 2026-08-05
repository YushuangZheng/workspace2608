# RoboDojo 资产目录

此目录是 RoboDojo 的统一本地资产库，运行时映射为
`third_party/RoboDojo` 所期望的 `Assets/`。二进制资产不提交 Git，只提交本说明。

- 官方来源：`hf://datasets/RoboDojo-Benchmark/RoboDojo/Assets/`
- 上游代码：`third_party/RoboDojo` Git 子模块固定提交
- 本地目录：`assets/robodojo/Assets/`
- 运行期链接：`.runtime/robodojo/Assets`

机器人库统一下载上游当前提供的 X5 和 Franka；默认只下载论文子集
`push_T`、`pour_liquid_into_cup`、`sweep_blocks` 所需的 T 块、目标垫、瓶杯、流体、
扫帚、簸箕、方块、房间、相机、材质和评测布局。使用
`python scripts/run.py robodojo assets --all` 可按动态注册表下载全部 54 个任务的物体与
布局；下载按文件恢复，不把 USD、纹理、布局或轨迹加入主仓库。

场景、机器人和 policy 不在这里复制：场景/机器人配置由 `.runtime/robodojo/env_cfg` 管理，
上游 policy 由 `.runtime/robodojo/XPolicyLab/policy` 管理，项目评测通过
`scripts/run.py robodojo resources` 统一发现。

RoboDojo 仓库当前 `LICENSE` 文件为 MIT，但其 README 同时声明仅限非商业研究；资产数据
还可能包含各自的授权条件。当前只用于本论文的非商业研究，发布数据或模型前须再次核对
对应提交与资产元数据。
