"""
OfferPilot 测试公共配置。

核心机制：在 pytest 收集阶段（早于任何 companion_ai 模块 import）通过
环境变量注入测试配置。pydantic-settings 中环境变量优先级高于 .env，
因此 vector_store 模块级单例初始化时读到的是测试值：

  - CHROMA_PERSIST_DIR     → 临时目录（隔离真实 ./chroma_data）
  - EMBEDDING_API_KEY      → 占位值，触发 create_embedding_function 的
                             本地 SimpleEmbeddingFunction 分支（零远程调用）
  - MCP_ENABLED=false      → @tool 层直接走本地降级实现（不拉起 MCP 子进程）
  - CHECKPOINT_DB_PATH     → 临时路径（隔离真实 ./checkpoints.db）
  - SENTIMENT_FALLBACK_ENABLED=false → 情感分析走关键词方案（不加载模型，
                             免受本地 .env 开启模型的影响，保证零网络依赖）
"""

import os
import tempfile
import uuid
from unittest.mock import MagicMock

# ---- 环境变量必须在 import companion_ai 之前设置 ----
_TEST_TMP_DIR = tempfile.mkdtemp(prefix="cai_test_")
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(_TEST_TMP_DIR, "chroma")
os.environ["EMBEDDING_API_KEY"] = "your_embedding_api_key_here"  # 占位值 → 本地随机向量
os.environ["MCP_ENABLED"] = "false"
os.environ["CAREER_MCP_ENABLED"] = "false"
os.environ["CHECKPOINT_DB_PATH"] = os.path.join(_TEST_TMP_DIR, "checkpoints.db")
os.environ["SENTIMENT_FALLBACK_ENABLED"] = "false"

import pytest  # noqa: E402


@pytest.fixture
def unique_user():
    """生成唯一 user_id，测试结束后自动清理该用户在向量库中的全部数据。"""
    from companion_ai.memory.vector_store import vector_store

    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    yield user_id
    for collection in (
        vector_store.conversation_collection,
        vector_store.profile_collection,
    ):
        try:
            collection.delete(where={"user_id": user_id})
        except Exception:
            pass


@pytest.fixture
def mock_llm(monkeypatch):
    """
    LLM mock 工厂。

    用法:
        llm = mock_llm("companion_ai.agents.interview_agent", "_get_eval_llm",
                       return_content='{"score": 80}')
    之后该模块内 _get_eval_llm() 返回的对象 .invoke(...) 均返回
    content=return_content 的 mock，可按需继续配置 side_effect。
    """

    def _factory(module_path: str, factory_name: str = "_get_llm",
                 return_content: str = "") -> MagicMock:
        response = MagicMock()
        response.content = return_content
        llm = MagicMock()
        llm.invoke.return_value = response
        llm.invoke.return_value.content = return_content
        monkeypatch.setattr(
            f"{module_path}.{factory_name}", lambda: llm
        )
        return llm

    return _factory
