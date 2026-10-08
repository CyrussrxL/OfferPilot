# OfferPilot 项目现状文档

> 多 Agent 协作的智能求职辅导系统：编程辅导 · 求职建议 · 循环模拟面试 · 情绪关怀
> 技术栈：LangGraph + LangChain + ChromaDB + 通义千问（DashScope 兼容模式）+ MCP 官方 SDK + FastAPI + Streamlit

---

## 1. 系统架构

### 1.1 LangGraph 工作流（companion_ai/graph/）

```
guard（情感分析 + 三层分类）
  ↓
memory（双路召回 + 事实提取 + 周期性反思）
  ↓ 条件路由（interview_mode 优先）
  ├─ interview_evaluate →(ask/probe)→ interview_ask ──┐   循环面试状态机
  │            └→(report)→ interview_report ──────────┤   （条件回边）
  ├─ coding（CodingAgent，绑定 MCP 沙箱工具）          │
  ├─ career（CareerAgent，绑定 3 个 MCP 求职工具）     │
  └─ general_chat（通用闲聊）                          │
  ↓                                                   ↓
response_composer（情绪关怀前置 + 拼接最终回复）←──────┘
  ↓
END
```

关键设计：
- **进程级图单例**（`workflow.compile_graph()`）：compiled graph 为模块级单例，检查点采用 SqliteSaver 落盘（`CHECKPOINT_DB_PATH`，依赖缺失时降级 MemorySaver），`interview_session` 依赖检查点跨轮持久化，同一 `thread_id` 的多次 invoke 构成一场连续面试，进程重启后亦可恢复
- **面试模式关闭时重置会话**：`run_workflow` 在 `interview_mode=False` 时写入 `{"active": False}`，避免 checkpoint 中旧 session 残留导致误评估
- **条件回边**：`interview_evaluate → _route_interview → interview_ask / interview_report`，其中 probe（低分追问）走回边

### 1.2 目录结构

```
companion_ai/
├── agents/           # LangGraph 节点
│   ├── guard_agent.py          # 三层分类：代码正则 → 向量相似度 → LLM few-shot 仲裁 → 关键词兜底
│   ├── memory_agent.py         # 双路召回（fact top-2 + episode top-3）、事实提取、每15轮反思
│   ├── coding_agent.py         # 编程辅导，Function Calling 绑定 execute_code_sandbox
│   ├── career_agent.py         # 求职辅导，绑定 evaluate_resume / get_interview_questions / get_job_requirements
│   ├── interview_agent.py      # 循环面试状态机（evaluate/ask/report 三节点）
│   └── response_composer.py    # 情绪关怀统一收口 + general_chat 节点 + 日报
├── graph/
│   ├── state.py                # State TypedDict（含 interview_session/report 等字段）
│   └── workflow.py             # 图定义、条件路由、进程级单例、run_workflow 入口
├── memory/
│   ├── vector_store.py         # ChromaDB 封装：对话/事实/画像/分类种子 + 三因子加权检索
│   └── memory_reflection.py    # extract_facts（LLM 抽事实）+ reflect_on_profile（画像反思）
├── tools/
│   ├── mcp_server.py           # 自建 MCP Tool Server（FastMCP + stdio），4 个工具
│   ├── mcp_stdio_client.py     # 持久 MCP 客户端（后台线程事件循环，崩溃可恢复）
│   ├── mcp_tools.py            # execute_code_sandbox（MCP 优先 + 本地降级）
│   ├── career_tools.py         # 求职工具三层结构（_core / @tool / MCP 注册）
│   └── python_executor.py      # 子进程沙箱（stdin=DEVNULL 防孙进程挂起）
├── backend/main.py             # FastAPI：对话/日报/分类纠错/记忆管理 API
├── frontend/streamlit_app.py   # 前端：Tabs（对话 + 记忆管理面板）
├── data/job_requirements.json # 岗位技能要求知识库（8 个方向）
└── utils/                      # config（pydantic-settings）、helpers、logger
evaluation/                     # 离线评测体系（见 §5）
```

## 2. 核心特性（已实现并验证）

| 特性 | 说明 | 验证结果 |
|---|---|---|
| 置信度分层路由 | 代码正则 → 向量相似度（top-3 投票）→ 低置信(<0.6) LLM few-shot 仲裁 → 关键词兜底；错分反馈回流种子 | 100 条评测集：fast 89% / full 91%，仲裁仅 4% 样本触发 |
| 向量记忆系统 | episode/fact 双路召回；三因子加权（相似度 × 0.95^days 时间衰减 × min(count×0.1, 0.5) 频率，fact ×1.3 加成） | 事实路径 Recall@2 = 100% vs 原文路径 33%，MRR 0.667 |
| 事实提取与反思 | 每轮 LLM 抽取结构化事实；每 15 轮对话触发画像反思（Generative Agents 对齐） | E2E 中每轮提取 2-4 条事实，画像自动更新 |
| 自建 MCP Tool Server | FastMCP + stdio，4 工具：execute_code / evaluate_resume / get_interview_questions / get_job_requirements；@tool 层 MCP 优先 + 本地降级（source 字段可验证） | 协议层 + CodingAgent/CareerAgent E2E 全通 |
| 循环面试状态机 | 出题（MCP 题库+画像定制）→ 评估回答 → 低分(<60)动态追问（每题1次）→ 报告（维度评分/薄弱点/学习建议），薄弱点回流事实记忆 | 4 轮 E2E：追问触发、状态跨轮流转、报告生成、事实回流 |
| JD 驱动 Gap 分析 | get_job_requirements（知识库 8 岗位）+ 画像/事实对比 → 三档 Gap 矩阵 + 学习路径，学习进展经事实提取形成闭环 | E2E：LLM 自主调用工具，Gap 记录写入画像 |
| 用户可控记忆 | 前端记忆面板 + REST API：事实查看/纠正/删除，画像与反思摘要展示，面试战绩展示 | 纠正/删除 E2E 生效 |
| 情绪感知 | 情感分析（可加载模型/关键词降级）+ ResponseComposer 统一情绪关怀（避免各 Agent 重复注入） | 长期运行验证 |

## 3. MCP 工具链路

```
Agent @tool（LangChain）→ StdioMCPClient.call_tool（同步桥接）
  → 后台线程事件循环 run_coroutine_threadsafe → ClientSession
  → stdio 管道 → FastMCP Server（python -m companion_ai.tools.mcp_server）
  → 工具实现（复用 _core 纯函数）→ 原路返回
```

- `mcp_stdio_client`：崩溃后可恢复（start 时重置 `_ready`/`_stop` Event）
- 降级开关：`MCP_ENABLED=false` 即纯本地模式，行为一致可测

## 4. 配置要点（.env）

| 配置 | 默认 | 说明 |
|---|---|---|
| `LLM_PROVIDER` / `ALIYUN_MODEL` | aliyun / qwen3.8-max-0902 | 主 LLM |
| `EMBEDDING_MODEL` | text-embedding-v3 | 向量检索（批量上限 10 条/次，已分批） |
| `MCP_ENABLED` / `CAREER_MCP_ENABLED` | true | MCP 优先开关 |
| `MCP_STDIO_SERVER_CMD` | python -m companion_ai.tools.mcp_server | Server 启动命令 |
| `GUARD_LLM_ARBITRATION` / `GUARD_ARBITRATION_THRESHOLD` | true / 0.6 | LLM 仲裁开关与阈值 |
| `MEMORY_FACT_EXTRACTION` / `MEMORY_FACT_BOOST` / `MEMORY_REFLECTION_INTERVAL` | true / 1.3 / 15 | 记忆系统 |
| `INTERVIEW_GRAPH_ENABLED` / `INTERVIEW_MAX_QUESTIONS` / `INTERVIEW_PROBE_SCORE_THRESHOLD` | true / 3 / 60 | 面试状态机 |
| `CHECKPOINT_DB_PATH` | ./checkpoints.db | LangGraph 检查点落盘路径（面试会话跨进程恢复） |

## 5. 评测体系（evaluation/）

```bash
python evaluation/run_classification_eval.py          # fast：向量+规则层（~20s）
python evaluation/run_classification_eval.py --full   # 全链路含 LLM 仲裁（~45s）
python evaluation/run_memory_eval.py                  # 记忆召回 + 加权方向性（自动清理）
python evaluation/run_latency_profile.py              # 节点延迟 + token（3 场景）
```

最近一次结果（2026-10-01，报告在 `evaluation/reports/`）：
- 分类：fast Acc **89%** / full Acc **91%**（coding F1 0.98 / career 0.96）
- 记忆：fact Recall@2 **100%**，episode Recall@3 33%，MRR 0.667
- 延迟/token：每轮 2-4 次 LLM 调用，平均 **7288 tokens**/轮；节点耗时 coding≈41s、career≈93s（含工具二轮生成）

## 5.1 测试体系（tests/）

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                    # 全量 190 个测试 + 覆盖率（总覆盖率 53%）
pytest tests/unit         # 仅单元测试
pytest -m "not slow"     # 跳过慢速测试
```

- **环境隔离**：`conftest.py` 顶层（早于 companion_ai import）注入 `CHROMA_PERSIST_DIR`/`CHECKPOINT_DB_PATH` 临时目录、`MCP_ENABLED=false`、Embedding 占位 Key（触发 SimpleEmbeddingFunction 本地降级），真实 `chroma_data`/`checkpoints.db` 不被触碰，无需 `.env`
- **全 mock 零外部调用**：不请求 LLM API、不拉 MCP Server（各模块 `_get_llm()` 工厂 monkeypatch MagicMock）
- **分层**：`tests/unit/`（纯函数 → LLM mock 层 → vector_store → workflow 路由/图单例）、`tests/integration/`（FastAPI TestClient，patch run_workflow）
- **CI**：`.github/workflows/tests.yml`，push/PR 触发 ubuntu + Python 3.11 全量测试

## 6. 运行方式

```bash
# 前端（推荐，自动经 run_workflow 走全链路，MCP Server 按需拉起）
streamlit run companion_ai/frontend/streamlit_app.py

# 后端 API（可选）
uvicorn companion_ai.backend.main:app --host 0.0.0.0 --port 8000

# 单独启动 MCP Server（正常无需手动，客户端自动拉起）
python -m companion_ai.tools.mcp_server
```

主要 API：`POST /api/chat`、`GET /api/daily-report/{user_id}`、`POST /api/classification/correction`、`GET/DELETE/PUT /api/memory/facts/{user_id}[/{fact_id}]`

## 7. 已知限制与后续方向

- 事实提取为同步执行，阻塞 MemoryAgent 约 10-18s（待异步化）
- 情感分析默认关键词方案（`SENTIMENT_FALLBACK_ENABLED=False` 快速启动）
- 待做：主动式关怀（Proactive Agent）、事实冲突消解（supersede）、记忆可视化增强

## 8. 近期变更记录

- 2026-10-07：情绪关怀升级——①模板轮换：_generate_emotion_care 分级关怀语由单条固定文案改为同层级多变体轮换（deep×3、负面三档×2、正面两档×2、中性×2），连续多轮负面不再收到一字不差的重复关怀语；层级 emoji 保持稳定作为分级语义锚点，_reset_care_rotation() 供测试重置。②持续低迷通道：在"末尾连续 3 次 <0.4"之外新增滑动窗口均值判定（最近 5 次均值 <0.45 且样本 ≥4），捕捉正负交替的慢性低落。测试 208→214 全绿
- 2026-10-04：pytest 测试体系 + CI——190 个测试全绿（tests/unit 161 + integration 29），总覆盖率 53%；全 mock 零外部调用，conftest 顶层环境隔离（临时向量库/检查点，不触碰真实数据）；GitHub Actions CI（.github/workflows/tests.yml）。测试中发现并修复真实 bug：SimpleEmbeddingFunction 缺失 chromadb 1.5 EF 协议的 embed_query(input=[text])→[embedding] 契约，本地降级模式检索曾全部静默失败（异常被 try/except 吞掉）
- 2026-10-01（三）：持久化与记忆容量治理——检查点由 MemorySaver 升级为 SqliteSaver 落盘（`CHECKPOINT_DB_PATH`，依赖缺失自动降级），面试会话跨进程重启可恢复（两段式进程验证：新进程恢复 session 并正确走评估→追问路径）；compress_memories 由"只写统计不删除"重写为真实删除（episode 超 200 条时按 timestamp 保留最近 100 条，fact 永不删，where 过滤 memory_type≠fact 覆盖 user/assistant 两 role）；ResponseComposer 中 AI 回复入库截断至 500 字（全文回复曾达 3000+ 字，是检索噪音与存储的主要来源；前端完整历史另存 conversations/*.json 不受影响）
- 2026-10-01（二）：全量代码审查修复——移除旧版 HTTP 占位 MCP 客户端（career_mcp_tools.py 删除、mcp_tools.py 仅保留真实沙箱工具、配置清理，消除启动时双"初始化完成"日志）；修复 save_user_profile 依赖 chromadb update 对缺失 id 静默忽略导致新用户画像永不落库（改 upsert）；修复情绪趋势被旧画像快照两次覆盖（memory_agent 写序 + 快照回传）；修复检索计数从未落库导致频率加权失效；retrieve_memories 增加 user_id + 排除 fact 的 where 过滤；修复 career_agent LLM 失败时 executed_tools 未定义；streamlit thread_id 跟随 user_id。修复后 E2E 验证：分类/事实提取/MCP 工具/Gap 分析/情绪趋势/画像全链路通过
- 2026-10-01：循环面试对话图 + 面试报告；用户可控记忆面板与 API；JD 驱动 Gap 分析（get_job_requirements）；评测体系三件套；修复 MCP 客户端崩溃恢复、面试 session 残留、沙箱孙进程挂起（stdin=DEVNULL）等 bug
