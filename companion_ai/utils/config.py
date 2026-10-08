import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    项目全局配置，通过 .env 文件或环境变量加载。
    使用 pydantic-settings 实现类型安全的配置管理。
    """

    DEEPSEEK_API_KEY: Optional[str] = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    OPENAI_API_KEY: Optional[str] = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    ALIYUN_API_KEY: Optional[str] = ""
    ALIYUN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ALIYUN_MODEL: str = "qwen-plus"

    LLM_PROVIDER: str = "aliyun"

    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_CONVERSATION: str = "conversations"
    CHROMA_COLLECTION_PROFILE: str = "user_profiles"
    CHROMA_COLLECTION_CLASSIFICATION: str = "classification_seeds"

    EMBEDDING_API_KEY: Optional[str] = ""
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL: str = "text-embedding-v3"

    # MCP 工具服务（自建 Tool Server，stdio 协议）
    MCP_ENABLED: bool = False
    MCP_STDIO_SERVER_CMD: str = "python -m companion_ai.tools.mcp_server"
    # CareerAgent 工具描述开关（工具实际链路由 MCP_ENABLED 控制）
    CAREER_MCP_ENABLED: bool = False

    SENTIMENT_MODEL_NAME: str = "distilbert-base-uncased-finetuned-sst-2-english"
    # 默认使用关键词方案，快速启动
    # 需要模型时设为 True 即可
    SENTIMENT_FALLBACK_ENABLED: bool = False

    HF_ENDPOINT: str = "https://hf-mirror.com"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

    DEFAULT_USER_ID: str = "default_user"

    # LangGraph 检查点持久化（面试会话跨进程重启恢复）
    CHECKPOINT_DB_PATH: str = "./checkpoints.db"

    # GuardAgent 置信度分级仲裁
    GUARD_LLM_ARBITRATION: bool = True          # 低置信度时启用 LLM few-shot 仲裁
    GUARD_ARBITRATION_THRESHOLD: float = 0.6    # 低于该置信度触发 LLM 仲裁

    # 记忆提取与反思
    MEMORY_FACT_EXTRACTION: bool = True         # 对话后抽取结构化用户事实
    MEMORY_FACT_BOOST: float = 1.3              # 事实记忆在检索打分中的加成系数
    MEMORY_REFLECTION_INTERVAL: int = 15        # 每 N 次对话触发一次画像反思

    # 循环面试对话图
    INTERVIEW_GRAPH_ENABLED: bool = True        # 启用面试状态机图（False 时降级旧版 CareerAgent 面试模式）
    INTERVIEW_MAX_QUESTIONS: int = 3            # 每场面试的题目数上限
    INTERVIEW_PROBE_SCORE_THRESHOLD: int = 60   # 回答得分低于该阈值时触发动态追问

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def llm_api_key(self) -> str:
        if self.LLM_PROVIDER == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY", "") or self.DEEPSEEK_API_KEY or ""
        elif self.LLM_PROVIDER == "aliyun":
            return os.getenv("OPENAI_API_KEY", "") or os.getenv("ALIYUN_API_KEY", "") or self.ALIYUN_API_KEY or ""
        return os.getenv("OPENAI_API_KEY", "") or self.OPENAI_API_KEY or ""

    @property
    def llm_base_url(self) -> str:
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_BASE_URL
        elif self.LLM_PROVIDER == "aliyun":
            return self.ALIYUN_BASE_URL
        return self.OPENAI_BASE_URL

    @property
    def llm_model(self) -> str:
        if self.LLM_PROVIDER == "deepseek":
            return self.DEEPSEEK_MODEL
        elif self.LLM_PROVIDER == "aliyun":
            return self.ALIYUN_MODEL
        return self.OPENAI_MODEL

    @property
    def embedding_api_key(self) -> str:
        return os.getenv("EMBEDDING_API_KEY", "") or self.EMBEDDING_API_KEY or self.llm_api_key

    @property
    def embedding_base_url(self) -> str:
        return self.EMBEDDING_BASE_URL

    @property
    def embedding_model(self) -> str:
        return self.EMBEDDING_MODEL


settings = Settings()