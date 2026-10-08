# 延迟与 Token 剖析报告

- 场景数: 3 | 总 LLM 调用: 9 | 总 token: 21866
- 平均每轮 token: 7288.7

| 场景 | 耗时(s) | LLM调用 | token |
|---|---|---|---|
| coding+MCP沙箱 | 54.79 | 3 | 4667 |
| career+Gap分析 | 109.12 | 4 | 12555 |
| chitchat纯对话 | 68.87 | 2 | 4644 |

## 节点平均耗时

| 节点 | 平均耗时(ms) |
|---|---|
| career | 92660 |
| general_chat | 62140 |
| coding | 41470 |
| memory | 9640 |
| guard | 2150 |
| response_composer | 380 |