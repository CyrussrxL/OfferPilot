# GuardAgent 分类系统评测报告

- 模式: **full**（全链路，含 LLM 仲裁）
- 评测集: 100 条（25 × 4 类，与种子隔离）
- **总体 Accuracy: 91.00%**
- 平均置信度: 0.879
- LLM 仲裁触发率: 4.00%
- 总耗时: 42.4s

## 每类指标

| 类别 | Precision | Recall | F1 | 样本数 |
|---|---|---|---|---|
| coding | 100.00% | 96.00% | 97.96% | 25 |
| career | 100.00% | 92.00% | 95.83% | 25 |
| emotional | 73.53% | 100.00% | 84.75% | 25 |
| chitchat | 100.00% | 76.00% | 86.36% | 25 |

## 混淆矩阵（行=真实，列=预测）

| 真实\预测 | coding | career | emotional | chitchat |
|---|---|---|---|---|
| coding | 24 | 0 | 1 | 0 |
| career | 0 | 23 | 2 | 0 |
| emotional | 0 | 0 | 25 | 0 |
| chitchat | 0 | 0 | 6 | 19 |

## 错分样本（前 10 条）

- [coding → emotional] (conf=0.87) TCP 三次握手的过程说一下...
- [career → emotional] (conf=0.68) 帮我分析一下目标岗位的技能要求差距...
- [career → emotional] (conf=0.92) 投简历一直挂，是不是我背景太差了...
- [chitchat → emotional] (conf=0.7) 你好...
- [chitchat → emotional] (conf=0.68) 给我讲个笑话吧...
- [chitchat → emotional] (conf=0.72) 无聊了，陪我聊聊天...
- [chitchat → emotional] (conf=0.69) 你觉得自己聪明吗...
- [chitchat → emotional] (conf=0.9) 给我一句激励的话...
- [chitchat → emotional] (conf=0.71) 今天过得好无聊啊...