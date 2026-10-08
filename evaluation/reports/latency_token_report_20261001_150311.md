# 延迟与 Token 剖析报告

- 场景数: 3 | 总 LLM 调用: 9 | 总 token: 0
- 平均每轮 token: 0.0

| 场景 | 耗时(s) | LLM调用 | token |
|---|---|---|---|
| coding+MCP沙箱 | 37.24 | 3 | 0 |
| career+Gap分析 | 79.01 | 4 | 0 |
| chitchat纯对话 | 42.01 | 2 | 0 |

## 节点平均耗时

| 节点 | 平均耗时(ms) |
|---|---|
| career | 65960 |
| general_chat | 36460 |
| coding | 31300 |
| memory | 5467 |
| guard | 2380 |
| response_composer | 337 |