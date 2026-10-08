"""FastAPI 后端集成测试（TestClient）。

/api/chat 与 /api/report 通过 mock run_workflow / run_daily_report 隔离
全链路 LLM 调用；其余路由直接操作临时向量库（conftest 注入）。
"""

import pytest
from fastapi.testclient import TestClient

from companion_ai.backend.main import app
from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.helpers import get_timestamp

client = TestClient(app)


@pytest.fixture
def mock_workflow(monkeypatch):
    def _install(final_response="mock 回复内容"):
        def fake_run_workflow(user_id, message, thread_id=None, interview_mode=False,
                              resume_summary=None, jd_text=None):
            return {
                "final_response": final_response,
                "emotion_label": "neutral",
                "emotion_score": 0.5,
                "message_category": "chitchat",
            }

        monkeypatch.setattr(
            "companion_ai.backend.main.run_workflow", fake_run_workflow
        )

    return _install


class TestHealth:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestChat:
    def test_chat_success(self, mock_workflow):
        mock_workflow("这是 mock 的回复")
        response = client.post("/api/chat", json={
            "user_id": "api_test_user", "message": "你好",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["final_response"] == "这是 mock 的回复"
        assert data["message_category"] == "chitchat"

    def test_chat_workflow_exception_500(self, monkeypatch):
        def fake_run_workflow(**kwargs):
            raise RuntimeError("LLM down")

        monkeypatch.setattr(
            "companion_ai.backend.main.run_workflow", fake_run_workflow
        )
        response = client.post("/api/chat", json={
            "user_id": "api_test_user", "message": "你好",
        })
        assert response.status_code == 500

    def test_chat_missing_message_422(self):
        response = client.post("/api/chat", json={"user_id": "u"})
        assert response.status_code == 422


class TestFactsApi:
    """用户可控记忆：facts 查看/纠正/删除全链路。"""

    def _prepare_fact(self, user_id):
        vector_store.store_fact(user_id, "用户熟悉 Java", get_timestamp())
        return vector_store.get_facts_with_metadata(user_id)[0]["id"]

    def test_list_facts(self, unique_user):
        vector_store.store_fact(unique_user, "用户在准备秋招", get_timestamp())
        response = client.get(f"/api/memory/facts/{unique_user}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1
        assert data["facts"][0]["text"] == "用户在准备秋招"

    def test_update_fact(self, unique_user):
        fact_id = self._prepare_fact(unique_user)
        response = client.put(
            f"/api/memory/facts/{unique_user}/{fact_id}",
            json={"fact_text": "用户熟悉 Python"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert vector_store.get_all_facts(unique_user) == ["用户熟悉 Python"]

    def test_update_fact_empty_text_400(self, unique_user):
        fact_id = self._prepare_fact(unique_user)
        response = client.put(
            f"/api/memory/facts/{unique_user}/{fact_id}",
            json={"fact_text": "  "},
        )
        assert response.status_code == 400

    def test_delete_fact(self, unique_user):
        fact_id = self._prepare_fact(unique_user)
        response = client.delete(f"/api/memory/facts/{unique_user}/{fact_id}")
        assert response.status_code == 200
        assert vector_store.get_all_facts(unique_user) == []


class TestClassificationCorrectionApi:
    def test_correction(self):
        from companion_ai.memory.vector_store import vector_store as vs

        before = vs.classification_collection.count()
        response = client.post("/api/classification/correction", json={
            "user_id": "api_test_user",
            "message": "测试：这道动态规划题的时间复杂度怎么分析",
            "correct_category": "coding",
        })
        assert response.status_code == 200
        assert response.json()["success"] is True
        after = vs.classification_collection.count()
        assert after == before + 1
        vs.classification_collection.delete(ids=[response.json()["seed_id"]])

    def test_correction_empty_message_400(self):
        response = client.post("/api/classification/correction", json={
            "message": "   ", "correct_category": "coding",
        })
        assert response.status_code == 400


class TestMemoriesAndProfileApi:
    def test_retrieve_memories(self, unique_user):
        vector_store.store_conversation(
            user_id=unique_user, text="在学习 LangGraph",
            emotion="neutral", category="chitchat",
            timestamp=get_timestamp(), role="user", memory_type="episode",
        )
        response = client.get(
            f"/api/memories/{unique_user}", params={"query": "学习", "top_k": 3}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["memories"]) == 1

    def test_get_profile_default(self, unique_user):
        response = client.get(f"/api/profile/{unique_user}")
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_conversations_list(self):
        response = client.get("/api/conversations")
        assert response.status_code == 200
        assert response.json()["success"] is True
        assert "conversations" in response.json()

    def test_conversation_not_found_404(self):
        response = client.get("/api/conversations/nonexistent_conv_id_123")
        assert response.status_code == 404
