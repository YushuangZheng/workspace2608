# DynaMAC / MiDiGaP 算法复现

本仓库只维护论文中的 DynaMAC 与 MiDiGaP 数学和策略实现，不包含仿真器、机器人资产或
外部任务库数据。

## 只需要看的文件

1. `source/policy/dynamac.py`：DynaMAC Algorithm 1 与双臂并发策略；
2. `source/policy/midigap.py`：MiDiGaP、约束更新和 VAPOR；
3. `source/data/__init__.py`：项目自身 NPZ 演示包格式；
4. `scripts/run.py`：唯一命令行入口；
5. `configs/dynamac.json`：冻结的数值选择；
6. `logs/research_log.md`：中文研究记录和结论边界。

## 运行

```bash
python -m pip install -e '.[test,midigap]'

# 验证随附单臂/双臂演示能走完整训练链
python scripts/run.py verify

# 保存单臂 checkpoint
python scripts/run.py fit --task single --output outputs/single_dynamac.npz

# 保存左右两套双臂 checkpoint
python scripts/run.py fit --task bimanual --output outputs/bimanual_dynamac

# 检查 checkpoint
python scripts/run.py inspect outputs/single_dynamac.npz

pytest -q
ruff check source scripts tests
```

## 实现边界

DynaMAC 覆盖 `R3 × S3` 任务参数、黎曼高斯 marginal、Product-of-Experts、链接过滤、
虚拟末端帧、技能序列和双臂并发策略。MiDiGaP 覆盖流形均值、模态聚类、技能转移、
约束更新和 VAPOR 的可审计数值实现。

`data/dynamac_demos.npz` 只用于接口和算法结构回归，不代表论文基准成功率；论文外部的
仿真任务、视觉系统、真机数据和官方评测仍需单独部署。
