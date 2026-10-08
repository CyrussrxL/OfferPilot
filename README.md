# OfferPilot 🧭

基于 LangGraph 的多 Agent 智能求职领航系统：编程辅导 · 求职规划 · 循环模拟面试 · 情绪关怀

## ✨ 功能特性

- **多 Agent 协作**: LangGraph 状态图编排 6 个专业化节点，条件边动态路由，每个 Agent Prompt 聚焦单一职责
- **置信度分层分类**: 代码正则 → 向量相似度（top-3 投票）→ 低置信度 LLM few-shot 仲裁 → 关键词兜底；错分反馈回流种子库
- **混合意图处理**: 单条消息中的多个需求不再丢失——分类同时输出主/次意图（强意图词单次命中 + 弱词双次确认的轻量检测，零额外 LLM 成本），主意图走专业路由，次意图由回复合成器追加回应段
- **向量记忆 + 事实提取 + 反思**: episode/fact 双路召回，三因子加权（相似度 × 时间衰减 × 频率，事实加成），每轮抽取结构化用户事实，周期性画像反思（对齐 Generative Agents）
- **自建 MCP Tool Server**: 基于 MCP 官方 SDK（FastMCP + stdio）将代码沙箱、简历评分、面试题库、岗位要求封装为标准 MCP 工具服务，Agent 经 Function Calling 真实调用，不可用时自动降级本地实现
- **循环面试对话状态机**: 动态出题 → 评估回答 → 薄弱点追问（条件回边）→ 结构化面试报告，报告薄弱点回流长期记忆；前端独立面试视图，切入自动开场、切走自动收尾出报告
- **会话持久化与记忆容量治理**: LangGraph 检查点经 SqliteSaver 落盘，面试会话跨进程重启可恢复；episode 记忆有界保留（超 200 条按时间戳删至最近 100，fact 永不删），AI 回复截断入库控制存储与检索噪音
- **JD 驱动技能 Gap 分析**: 岗位技能要求知识库与用户画像/事实记忆对比，输出三档 Gap 矩阵与学习路径，学习进展回流形成闭环
- **用户可控记忆面板**: 前端查看/纠正/删除 AI 提取的事实记忆，消除记忆黑盒（配套 RESTful API）
- **情绪关怀**: 基于历史情绪趋势的分级关怀机制，连续低落触发深度关怀
- **离线评测体系**: 分类评测集 / 记忆召回 ground truth / 节点级延迟与 token 剖析，指标可复现
- **前后端分离**: FastAPI 后端 + Streamlit 前端（对话 / 模拟面试 / 记忆管理三视图，输入框常驻消息底部，对话按「用户ID_时间_首句话」自动命名）

## 📊 评测结果（100 条分类评测集，报告见 `evaluation/reports/`）

| 指标 | 结果 |
|---|---|
| 分类准确率（向量层 / 全链路） | 89% / **91%** |
| LLM 仲裁触发率（成本控制） | 仅 4% 样本 |
| 事实记忆 Recall@2（语义改写查询） | **100%**（原文路径 33%，验证事实提取价值） |
| 记忆检索 MRR | 0.667 |
| 平均每轮 token 消耗 | ~7288 |

## 🛠️ 技术栈

Python | LangGraph | LangChain | ChromaDB | 通义千问（DashScope 兼容模式） | MCP 官方 SDK | FastAPI | Streamlit | Plotly

## 📐 架构设计

```
用户消息 → GuardAgent
             ├─ 情感分析（规则引擎 / 可选模型）
             ├─ 三层分类: 代码正则 → 向量相似度 → LLM 仲裁 → 关键词兜底
             ├─ 次意图检测（强/弱关键词，混合意图双路响应）
             └─ 行为分析（输入频率、消息长度、时间段）
          MemoryAgent
             ├─ 双路召回: 事实记忆 top-2 + 原文记忆 top-3（三因子加权）
             ├─ LLM 事实提取（结构化用户事实）
             ├─ 周期性画像反思（每 15 轮）
             └─ 主动记忆推送（根据情绪状态）
       ┌──── 条件路由（interview_mode 优先）────┐
       ↓              ↓               ↓         ↓
  Interview状态机  CodingAgent    CareerAgent  GeneralChat
  （循环面试图,    （MCP 沙箱）  （3 个 MCP 工具）
   SqliteSaver）
       │ evaluate → ask(追问/新题) / report（条件回边）
       └────┬──────────┬───────────┬─────────┘
            ↓
    ResponseComposer
       ├─ 情绪关怀（基于历史趋势分级，统一收口）
       ├─ 拼接专业回复
       └─ 存储记忆
            ↓
         最终回复
```

### MCP 工具链路（自建 Tool Server）

```
Agent @tool → StdioMCPClient（同步桥接，后台线程事件循环）
  → stdio 管道 → FastMCP Server（python -m companion_ai.tools.mcp_server）
  → 4 个工具: execute_code / evaluate_resume / get_interview_questions / get_job_requirements
  → 失败自动降级本地实现（返回 source 字段可验证链路）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```env
# LLM 配置（支持阿里云百炼 / DeepSeek / OpenAI）
LLM_PROVIDER=aliyun
ALIYUN_API_KEY=你的API密钥
ALIYUN_MODEL=qwen3.8-max-0902

# Embedding 配置（用于向量分类与记忆检索）
EMBEDDING_API_KEY=你的Embedding API密钥
EMBEDDING_MODEL=text-embedding-v3

# MCP 配置（自建 Tool Server，stdio 协议）
MCP_ENABLED=true
MCP_STDIO_SERVER_CMD=python -m companion_ai.tools.mcp_server

# 检查点持久化（面试会话跨进程重启恢复，可选，默认 ./checkpoints.db）
CHECKPOINT_DB_PATH=./checkpoints.db
```

### 3. 启动项目

```bash
# 一键启动（后端 + 前端）
python start_all.py

# 或单独启动
# 前端: streamlit run companion_ai/frontend/streamlit_app.py
# 后端: uvicorn companion_ai.backend.main:app --reload

# MCP Server 无需手动启动，客户端按需自动拉起
```

## 🌐 访问地址

- **前端界面**（对话 / 模拟面试 / 记忆管理三视图）: http://localhost:8501
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs (Swagger)

主要 API：`POST /api/chat`、`GET /api/daily-report/{user_id}`、`POST /api/classification/correction`（错分纠错闭环）、`GET/DELETE/PUT /api/memory/facts/{user_id}`（用户可控记忆）

## 📁 项目结构

```
OfferPilot/
├── companion_ai/
│   ├── agents/                  # LangGraph 节点
│   │   ├── guard_agent.py       # 情感分析 + 三层分类 + LLM 仲裁
│   │   ├── memory_agent.py      # 双路召回 + 事实提取 + 画像反思
│   │   ├── coding_agent.py      # 编程辅导（Function Calling 绑定沙箱工具）
│   │   ├── career_agent.py      # 求职辅导（3 个 MCP 工具 + Gap 分析）
│   │   ├── interview_agent.py   # 循环面试状态机（evaluate/ask/report）
│   │   └── response_composer.py # 情绪关怀统一收口 + 通用对话 + 日报
│   ├── graph/                   # LangGraph 定义（图单例 + 检查点）
│   ├── memory/                  # vector_store（ChromaDB）+ memory_reflection
│   ├── tools/                   # mcp_server / mcp_stdio_client / career_tools / python_executor 等
│   ├── data/                    # 分类种子 + 岗位技能要求知识库
│   ├── backend/                 # FastAPI（对话/日报/纠错/记忆管理 API）
│   ├── frontend/                # Streamlit（对话 / 模拟面试 / 记忆管理三视图）
│   └── utils/                   # config / logger / helpers
├── evaluation/                  # 离线评测体系
│   ├── data/                    # 100 条分类评测集
│   ├── run_classification_eval.py
│   ├── run_memory_eval.py
│   ├── run_latency_profile.py
│   └── reports/                 # 历次评测报告（markdown）
├── tests/                       # pytest 测试体系（全 mock 零外部调用）
│   ├── conftest.py              # 环境隔离：临时向量库/检查点/禁用 MCP
│   ├── unit/                    # 单元测试（helpers/agents/vector_store/workflow）
│   └── integration/             # FastAPI TestClient 集成测试
├── .github/workflows/tests.yml  # CI：push/PR 自动跑全量测试
├── agent.md                     # 项目现状文档（架构细节/配置/已知限制）
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── start_all.py
└── README.md
```

## 🎯 核心技术亮点

### 1. 置信度分层的语义路由

代码正则 → 向量相似度 → LLM few-shot 仲裁的三层漏斗：高置信样本直出（零 LLM 开销），仅 4% 低置信样本触发仲裁，用最小成本换取 91% 准确率；错分反馈经 API 回流种子库持续改进。

### 2. 事实提取增强的长期记忆

每轮对话 LLM 抽取结构化用户事实（学习目标/技能/求职进展），与原文记忆双路召回；三因子加权（相似度 × 时间衰减 × 频率 × 事实加成）模拟遗忘曲线；每 15 轮触发画像反思。评测证明：语义改写查询下事实路径 Recall 100% vs 原文路径 33%。容量治理保证记忆有界：episode 超 200 条时按时间戳保留最近 100 条（真实删除），fact 永不删除，AI 回复截断 500 字入库。

### 3. 自建 MCP Tool Server（工具服务提供方）

不同于消费第三方 API，本项目将代码沙箱、简历评分、面试题库、岗位要求封装为标准 MCP 工具服务（官方 SDK，stdio 传输），Agent 侧经 Function Calling 真实调用（工具决策 → 执行 → 结果回填 → 二轮生成），服务不可用时自动降级本地实现，链路可用 `source` 字段验证。

### 4. 循环面试对话状态机

带条件回边的多轮面试图：题目经 MCP 题库获取并结合画像定制，评估回答后按得分决策"薄弱点动态追问 / 下一题 / 生成报告"，会话状态经 LangGraph 检查点（SqliteSaver 落盘）持久化，跨轮对话与进程重启均可无缝续接；报告薄弱点写入事实记忆，后续辅导与检索自动感知用户短板。

前端为独立面试视图：切入自动开启面试（方向按用户画像定制，进度条实时显示已问题数与当前题目），切走自动经"结束面试"路径收尾生成报告并记录到画像 `job_progress.interview_reports`，历场面试表现可在记忆管理页回看——面试从"开关 + 手动结束"升级为随视图切换的完整生命周期管理。

### 5. JD 驱动的技能 Gap 闭环

目标岗位技能要求（知识库经 MCP 提供）与用户画像/事实记忆逐项对比，输出三档 Gap 矩阵（已具备/薄弱/缺失）与学习路径；用户学习进展经事实提取回流，形成"诊断 → 学习 → 再诊断"闭环。

### 6. 用户可控记忆（可信 AI）

AI 提取的事实可能有错误或过时——记忆管理面板支持查看、纠正、删除事实记忆（配套 REST API），纠正后立即生效于检索与画像反思，与分类错分纠错闭环共同构成用户反馈闭环。

## ✅ 测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                    # 全量测试 + 覆盖率报告（208 个测试，全 mock 零外部调用）
pytest -m "not slow"     # 跳过慢速测试
pytest tests/unit         # 仅单元测试
```

测试体系分层（纯函数 → LLM mock → 向量存储 → 图路由/API 集成），`conftest.py` 顶层注入临时数据目录与占位 API Key，无需真实 LLM/Embedding 配置即可运行，不触碰真实 `chroma_data` 与 `checkpoints.db`；GitHub Actions 在每次 push/PR 自动执行（`.github/workflows/tests.yml`）。

## 🧪 评测体系

```bash
python evaluation/run_classification_eval.py          # 分类：向量层（~20s）
python evaluation/run_classification_eval.py --full   # 分类：全链路含 LLM 仲裁
python evaluation/run_memory_eval.py                  # 记忆召回 + 加权方向性
python evaluation/run_latency_profile.py              # 节点延迟 + token 剖析
```

所有脚本自动清理评测数据，报告输出至 `evaluation/reports/`，指标可复现。

## 📝 许可证

MIT License
