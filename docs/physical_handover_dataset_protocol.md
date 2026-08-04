# 真实物理双臂交接数据集 v1 预注册协议

## 1. 前置条件与目的

物理专家 v3 已在预注册正式 seed `8000–8019` 上达到 `20/20`，满足数据采集前置门槛。
本协议创建新的 `data/handover_physical/v1`，只用于后续双臂关系估计和策略研究。

数据不会从 v3 正式评测成功轨迹中复制或挑选。采集使用一组独立 seed，重新运行真实
重力、碰撞、摩擦和双指夹持环境。旧 `data/handover_static/v1/v2` 保持逐字节不变。

数据管线实现冻结在提交 `9e5f722`。所调用的专家/环境源码指纹保持为：

```text
2e52bf2a0c961e5c79a4ca4a709bcb6416c3cca3e8f6ccf17dcc11339889d31c
```

## 2. 固定成员与零替补规则

- 数据 seed：`8200–8219`，共 20 条；
- 每条最多 1400 控制步，控制周期 `0.02 s`；
- 目标目录：`data/handover_physical/v1`；
- 采集模式：`exact_seed_batch_no_replacement`；
- 每个 seed 只运行一次；
- 任一 seed 失败、worker 缺失或源码指纹变化时，整个批次拒绝生成目标目录；
- 不允许增加 attempt、不允许用后续 seed 替补、不允许只保留成功子集；
- 目标目录已存在或含 `FROZEN` 时，采集器必须在仿真前拒绝覆盖。

失败批次只作为后续版本的诊断信息，不能修改本协议后仍称为数据集 v1。

## 3. 固定 schema

每条 `demo_XXX.npz` 至少保存：

- 连续 `time`、专家 `state` 和 16D 双臂 `action`；
- 左右末端 `xyz + wxyz`；
- 物体 `xyz + wxyz` 与目标 `xyz + wxyz`；
- 物体线速度；
- 四个指体各自的三维接触力与世界位置；
- `left_connected`、`right_connected` 两条独立边；
- `left_confidence`、`right_confidence`；
- 四值 `relation_label`；
- 终端位置、最大高度、最终 XY 误差、共同持物时长和稳定性；
- seed、实验指纹、专家源码指纹和坐标/四元数约定。

数据中没有 `carrier`，也不写入物体位姿或速度。`state` 只用于专家行为分段；
`relation_label` 必须由双指接触、近邻和相对运动证据产生，不得由 `state` 映射生成。

## 4. 固定单条验收

每条数据必须同时满足：

- 专家状态序列完整覆盖 `REST` 至 `RETREAT`；
- 关系生命周期精确为 `none → left_only → both → right_only → none`；
- 关系标签逐步等于左右两条独立 `connected` 边的组合；
- `both` 持续不低于 `0.20 s`；
- 最终 XY 误差小于 `0.04 m`；
- 最终位于支撑面且最后 25 步最大位移不超过 `0.01 m`；
- 左右关系在 connected 步内与双指接触的一致率均不低于 90%；
- 双臂末端和物体均无大于等于 `0.15 m` 的复位式单步跳变；
- 所有逐步数组长度一致、数值有限、时间轴连续。

90% 接触一致率允许固定的 3 步连接/解除迟滞，不允许以此掩盖长时间无接触锁存。

## 5. 固定整体验收与冻结

批次必须满足：

- 精确包含 20 个预注册 seed，顺序和文件一一对应；
- 没有重复 seed、额外文件或缺失 trial JSON；
- 所有文件使用同一专家源码指纹；
- 初始物体 x 与 y 覆盖范围分别不低于 `0.015 m`；
- 每个 NPZ 计算独立 SHA-256；
- ordered `file:sha256` 计算数据集 SHA-256；
- 冻结前后数据集 SHA-256 完全一致；
- 冻结 manifest 保存逐文件审计结果、源码提交、UTC 时间和验收汇总；
- `FROZEN` 创建后，采集和冻结脚本都必须拒绝覆盖。

只有全部条件通过，才允许把该目录称为 `handover_physical_v1`。

## 6. 固定命令

完整采集：

```bash
conda run -n env_isaaclab python scripts/collect_physical_handover_demos.py \
  --seeds 8200 8201 8202 8203 8204 8205 8206 8207 8208 8209 \
          8210 8211 8212 8213 8214 8215 8216 8217 8218 8219 \
  --max_steps 1400 \
  --output_dir data/handover_physical/v1
```

只读预审计：

```bash
conda run -n env_isaaclab python scripts/audit_physical_handover_dataset.py \
  --data_dir data/handover_physical/v1 \
  --expected_seeds 8200 8201 8202 8203 8204 8205 8206 8207 8208 8209 \
                   8210 8211 8212 8213 8214 8215 8216 8217 8218 8219
```

冻结命令在预审计通过后只执行一次：

```bash
conda run -n env_isaaclab python scripts/audit_physical_handover_dataset.py \
  --data_dir data/handover_physical/v1 \
  --expected_seeds 8200 8201 8202 8203 8204 8205 8206 8207 8208 8209 \
                   8210 8211 8212 8213 8214 8215 8216 8217 8218 8219 \
  --freeze --dataset_version handover_physical_v1
```
