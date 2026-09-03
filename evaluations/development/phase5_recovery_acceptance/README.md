# 阶段五恢复元数据验收

该评测使用五条正常示范重新构建全部冻结 V4 任务模型，并检查阶段五能否正确解析正式 LINK 锚点、episode 级 Pending 激活、UNLINK 元数据和合法重入状态。

运行：

```bash
PYTHONPATH="$PWD:$PWD/source" python evaluations/development/phase5_recovery_acceptance/run.py \
  --output evaluations/development/phase5_recovery_acceptance/results/v1
```

这是一项事件元数据与恢复组件接口验收，不等同于仿真故障恢复成功率实验。
