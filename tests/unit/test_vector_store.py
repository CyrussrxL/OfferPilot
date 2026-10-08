"""VectorStore（ChromaDB 封装）单元测试。

使用 conftest 注入的临时 ChromaDB 目录与本地随机向量，零外部调用；
每个测试使用唯一 user_id，fixture 自动清理。
"""

from datetime import datetime, timedelta

from companion_ai.memory.vector_store import vector_store
from companion_ai.utils.helpers import get_timestamp

BASE = datetime(2026, 1, 1)


def _ts(i: int) -> str:
    return (BASE + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S")


def _store_episode(user_id, i, role="user", category="chitchat"):
    vector_store.store_conversation(
        user_id=user_id,
        text=f"消息内容 {i}",
        emotion="neutral",
        category=category,
        timestamp=_ts(i),
        role=role,
        memory_type="episode",
    )


class TestStoreAndRetrieve:
    def test_store_conversation(self, unique_user):
        _store_episode(unique_user, 1)
        results = vector_store.conversation_collection.get(
            where={"user_id": unique_user}
        )
        assert len(results["ids"]) == 1
        assert results["metadatas"][0]["role"] == "user"
        assert results["metadatas"][0]["retrieval_count"] == "0"

    def test_retrieve_memories_user_isolation(self, unique_user):
        """retrieve_memories 只返回当前用户的记忆。"""
        other = unique_user + "_other"
        _store_episode(unique_user, 1)
        _store_episode(other, 2)
        memories = vector_store.retrieve_memories(unique_user, "消息内容", top_k=5)
        assert len(memories) == 1
        assert memories[0]["text"] == "消息内容 1"
        vector_store.conversation_collection.delete(where={"user_id": other})

    def test_retrieve_memories_excludes_facts(self, unique_user):
        """原文检索路径排除 fact 记忆（fact 走独立检索路径）。"""
        _store_episode(unique_user, 1)
        vector_store.store_fact(unique_user, "用户熟悉 LangGraph", _ts(2))
        memories = vector_store.retrieve_memories(unique_user, "消息", top_k=5)
        assert all(m["memory_type"] != "fact" for m in memories)
        assert len(memories) == 1

    def test_retrieve_memories_empty_for_new_user(self, unique_user):
        assert vector_store.retrieve_memories(unique_user, "任意查询") == []

    def test_retrieve_facts_only_returns_facts(self, unique_user):
        _store_episode(unique_user, 1)
        vector_store.store_fact(unique_user, "用户在准备秋招", _ts(2))
        facts = vector_store.retrieve_facts(unique_user, "秋招", top_k=5)
        assert len(facts) == 1
        assert facts[0]["memory_type"] == "fact"
        assert facts[0]["text"] == "用户在准备秋招"
        assert "final_score" in facts[0]


class TestFactsCrud:
    def test_get_all_facts(self, unique_user):
        vector_store.store_fact(unique_user, "事实A", _ts(1))
        vector_store.store_fact(unique_user, "事实B", _ts(2))
        assert set(vector_store.get_all_facts(unique_user)) == {"事实A", "事实B"}

    def test_get_facts_with_metadata_ids(self, unique_user):
        vector_store.store_fact(unique_user, "事实A", _ts(1))
        records = vector_store.get_facts_with_metadata(unique_user)
        assert len(records) == 1
        assert records[0]["id"]
        assert records[0]["text"] == "事实A"
        assert records[0]["timestamp"] == _ts(1)

    def test_update_fact(self, unique_user):
        vector_store.store_fact(unique_user, "用户熟悉 Java", _ts(1))
        fact_id = vector_store.get_facts_with_metadata(unique_user)[0]["id"]
        result = vector_store.update_fact(fact_id, "用户熟悉 Python")
        assert result["success"] is True
        assert vector_store.get_all_facts(unique_user) == ["用户熟悉 Python"]

    def test_update_fact_empty_text_rejected(self, unique_user):
        vector_store.store_fact(unique_user, "事实A", _ts(1))
        fact_id = vector_store.get_facts_with_metadata(unique_user)[0]["id"]
        assert vector_store.update_fact(fact_id, "  ")["success"] is False

    def test_delete_fact(self, unique_user):
        vector_store.store_fact(unique_user, "事实A", _ts(1))
        fact_id = vector_store.get_facts_with_metadata(unique_user)[0]["id"]
        result = vector_store.delete_fact(fact_id)
        assert result["success"] is True
        assert vector_store.get_all_facts(unique_user) == []


class TestUserProfile:
    def test_new_user_returns_default_profile(self, unique_user):
        """新用户返回默认画像模板并落库（upsert 保证）。"""
        profile = vector_store.get_user_profile(unique_user)
        assert profile["learning_goal"] == "找算法实习"
        assert profile["current_skill_level"] == "Python基础"
        assert profile["job_target"] == "AI开发"
        assert profile["emotional_trend"] == []
        # 默认画像已持久化，二次读取一致
        assert vector_store.get_user_profile(unique_user) == profile

    def test_save_new_user_profile_upsert(self, unique_user):
        """新用户直接保存应落库（upsert，chromadb update 静默忽略问题的回归测试）。"""
        profile = {"learning_goal": "拿到大厂 offer"}
        vector_store.save_user_profile(unique_user, profile)
        assert vector_store.get_user_profile(unique_user) == profile

    def test_save_updates_existing_profile(self, unique_user):
        vector_store.save_user_profile(unique_user, {"conversation_count": 1})
        vector_store.save_user_profile(unique_user, {"conversation_count": 2})
        assert vector_store.get_user_profile(unique_user)["conversation_count"] == 2

    def test_profiles_isolated_between_users(self, unique_user):
        other = unique_user + "_other"
        vector_store.save_user_profile(unique_user, {"goal": "A"})
        vector_store.save_user_profile(other, {"goal": "B"})
        assert vector_store.get_user_profile(unique_user)["goal"] == "A"
        assert vector_store.get_user_profile(other)["goal"] == "B"
        vector_store.profile_collection.delete(where={"user_id": other})

    def test_update_emotional_trend_appends(self, unique_user):
        vector_store.save_user_profile(unique_user, {})
        trend1 = vector_store.update_emotional_trend(unique_user, 0.6)
        trend2 = vector_store.update_emotional_trend(unique_user, 0.4)
        assert trend1 == [0.6]
        assert trend2 == [0.6, 0.4]
        # 趋势持久化到画像
        stored = vector_store.get_user_profile(unique_user)["emotional_trend"]
        assert stored == [0.6, 0.4]


class TestMemoryWeight:
    def test_time_decay_fresh_memory(self):
        assert vector_store._calculate_time_decay(get_timestamp()) == 1.0

    def test_time_decay_old_memory(self):
        old_ts = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")
        decay = vector_store._calculate_time_decay(old_ts)
        assert 0 < decay < 1.0
        # 14 天 ≈ 0.95^14 ≈ 0.49
        assert 0.4 <= decay <= 0.6

    def test_time_decay_invalid_timestamp(self):
        # 解析失败安全返回中等权重（不抛异常）
        assert vector_store._calculate_time_decay("not-a-timestamp") == 0.5
        assert vector_store._calculate_time_decay("") == 0.5

    def test_weight_increases_with_frequency(self, unique_user):
        """检索计数越高，权重越高（频率因子）。"""
        fresh = get_timestamp()
        low = vector_store._calculate_memory_weight({
            "similarity": 0.8, "timestamp": fresh,
            "retrieval_count": 0, "memory_type": "episode",
        })
        high = vector_store._calculate_memory_weight({
            "similarity": 0.8, "timestamp": fresh,
            "retrieval_count": 10, "memory_type": "episode",
        })
        assert high > low

    def test_fact_boost(self, unique_user):
        """fact 记忆检索得分有加成系数。"""
        fresh = get_timestamp()
        episode = vector_store._calculate_memory_weight({
            "similarity": 0.8, "timestamp": fresh,
            "retrieval_count": 0, "memory_type": "episode",
        })
        fact = vector_store._calculate_memory_weight({
            "similarity": 0.8, "timestamp": fresh,
            "retrieval_count": 0, "memory_type": "fact",
        })
        assert fact > episode


class TestCompressMemories:
    """记忆容量治理：episode 超量真实删除，fact 永不删除。"""

    def _seed(self, unique_user, n_episodes=251, n_facts=1):
        for i in range(n_facts):
            vector_store.store_fact(unique_user, f"用户事实 {i}", _ts(i))
        for i in range(n_episodes):
            _store_episode(
                unique_user, i,
                role="user" if i % 2 == 0 else "assistant",
            )

    def _count_by_type(self, unique_user, memory_type):
        where = (
            {"$and": [{"user_id": unique_user}, {"memory_type": memory_type}]}
            if memory_type
            else {"user_id": unique_user}
        )
        return len(
            vector_store.conversation_collection.get(where=where, limit=5000)["ids"]
        )

    def test_below_threshold_no_op(self, unique_user):
        self._seed(unique_user, n_episodes=10)
        result = vector_store.compress_memories(unique_user, 200, 100)
        assert result["status"] == "no_need"
        assert self._count_by_type(unique_user, None) == 11

    def test_compress_deletes_stale_episodes_keeps_facts(self, unique_user):
        self._seed(unique_user, n_episodes=251, n_facts=1)
        result = vector_store.compress_memories(unique_user, max_memories=200, keep_recent=100)
        assert result["status"] == "compressed"
        assert result["total_before"] == 251
        assert result["deleted"] == 151
        # episode 剩 100（timestamp 最大的 100 条），fact 保留
        assert self._count_by_type(unique_user, "episode") == 100
        assert self._count_by_type(unique_user, "fact") == 1
        # 保留的是最近时间戳
        eps = vector_store.conversation_collection.get(
            where={"$and": [
                {"user_id": unique_user}, {"memory_type": "episode"}
            ]},
            limit=5000,
        )
        kept_ts = sorted(m["timestamp"] for m in eps["metadatas"])
        assert kept_ts[0] == _ts(151) and kept_ts[-1] == _ts(250)
        # 压缩统计写入画像
        profile = vector_store.get_user_profile(unique_user)
        assert profile["memory_compression"]["deleted"] == 151

    def test_compress_covers_both_roles(self, unique_user):
        """删除统计覆盖 user/assistant 两种 role（每轮对话写 2 条 episode）。"""
        self._seed(unique_user, n_episodes=251, n_facts=0)
        result = vector_store.compress_memories(unique_user, 200, 100)
        assert result["deleted"] == 151
        remaining_roles = {
            m["role"] for m in vector_store.conversation_collection.get(
                where={"$and": [
                    {"user_id": unique_user}, {"memory_type": "episode"}
                ]},
                limit=5000,
            )["metadatas"]
        }
        assert remaining_roles == {"user", "assistant"}


class TestClassificationSeeds:
    def test_classify_returns_valid_category(self):
        category, confidence = vector_store.classify_message_with_confidence(
            "帮我优化这段代码的性能", top_k=3
        )
        assert category in ("coding", "career", "emotional", "chitchat")
        assert 0.0 <= confidence <= 1.0

    def test_seeds_initialized(self):
        assert vector_store.classification_collection.count() >= 100
