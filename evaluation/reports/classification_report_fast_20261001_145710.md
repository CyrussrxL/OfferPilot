# GuardAgent 分类系统评测报告

- 模式: **fast**（向量+规则层，禁用 LLM 仲裁）
- 评测集: 100 条（25 × 4 类，与种子隔离）
- **总体 Accuracy: 89.00%**
- 平均置信度: 0.867
- LLM 仲裁触发率: 0.00%
- 总耗时: 17.1s

## 每类指标

| 类别 | Precision | Recall | F1 | 样本数 |
|---|---|---|---|---|
| coding | 100.00% | 96.00% | 97.96% | 25 |
| career | 95.65% | 88.00% | 91.67% | 25 |
| emotional | 71.43% | 100.00% | 83.33% | 25 |
| chitchat | 100.00% | 72.00% | 83.72% | 25 |

## 混淆矩阵（行=真实，列=预测）

| 真实\预测 | coding | career | emotional | chitchat |
|---|---|---|---|---|
| coding | 24 | 0 | 1 | 0 |
| career | 0 | 22 | 3 | 0 |
| emotional | 0 | 0 | 25 | 0 |
| chitchat | 0 | 1 | 6 | 18 |

## 错分样本（前 10 条）

- [coding → emotional] (conf=0.87) TCP 三次握手的过程说一下...
- [career → emotional] (conf=0.68) 帮我分析一下目标岗位的技能要求差距...
- [career → emotional] (conf=0.48) 我想知道数据分析岗和算法岗哪个更适合我...
- [career → emotional] (conf=0.92) 投简历一直挂，是不是我背景太差了...
- [chitchat → emotional] (conf=0.7) 你好...
- [chitchat → emotional] (conf=0.68) 给我讲个笑话吧...
- [chitchat → emotional] (conf=0.72) 无聊了，陪我聊聊天...
- [chitchat → emotional] (conf=0.69) 你觉得自己聪明吗...
- [chitchat → career] (conf=0.49) 人是为什么要工作啊，随便聊聊...
- [chitchat → emotional] (conf=0.9) 给我一句激励的话...