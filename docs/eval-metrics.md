# eval指标的含义

| 指标 | 含义 | 数值 |
| --- | --- | --- |
| coverage | 成功产出可评分结果的 case 占比 = covered/total | 0.98（49/50） |
| invalid-output rate | 状态 completed 但 top-k 预测为空的 case 占比 = 8/50 | 0.16 |
| Top-1 File Accuracy | 首个预测文件命中 gold 非测试文件的 case 占比 = 13/49 | 0.2653 |
| Any File Recall@3 / @5 | Top-3 / Top-5 中至少一个 gold 文件命中 = 14/49 | 0.2857 / 0.2857 |
| All File Recall@3 / @5 | Top-3 / Top-5 覆盖该 case **全部** gold 文件 = 12/49 | 0.2449 / 0.2449 |
| Mean Gold File Recall@5 | 平均找回比例：Σ(命中数/该 case gold 数)/49 = 13/49 | 0.2653 |
| Latency P50 / P95 | 单 case 墙钟时间中位/95 分位 | 93.3s / 149.3s |
| LLM calls/case | 平均每次完整 RCA 的模型调用数（49 个有记录） | 7.84 |
| Tokens/case | 平均精确 token 总量（49/49 全为精确值，无 null） | 41,127 |