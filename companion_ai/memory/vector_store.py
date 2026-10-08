"""
记忆模块 —— VectorStore

职责：
  使用 ChromaDB 作为向量数据库，存储和检索用户的对话历史、用户画像和情绪轨迹。

设计理由：
  - 使用 chromadb.PersistentClient 实现数据持久化，重启后记忆不丢失。
  - 对话历史和用户画像分别存储在不同的 collection 中，便于独立管理和查询。
  - 每条对话记录包含 metadata（emotion, timestamp, category, user_id），
    支持按用户过滤和按类型检索。
  - 用户画像以 JSON 字符串形式存储，key 为 user_profile_{user_id}，
    包含学习目标、技能水平、求职目标、情绪趋势等。
  - 使用 OpenAI Embedding API 进行向量嵌入，支持语义检索和分类。
"""

import json
import os
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


class OpenAICompatibleEmbeddingFunction:
    """
    兼容 OpenAI 格式的 Embedding 函数类，支持阿里云 DashScope 等兼容 API。
    """
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._embedding_function = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base=base_url,
        )
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._embedding_function(input)

    def embed_documents(self, input) -> List[List[float]]:
        """ChromaDB add 路径（chromadb 1.x 以 input= 传参）。"""
        return self._embedding_function.embed_documents(input)

    def embed_query(self, input) -> List[float]:
        """ChromaDB query 路径（input 可为 str 或 List[str]）。"""
        return self._embedding_function.embed_query(input)

    def name(self) -> str:
        """ChromaDB >=0.5 需要 EF 提供 name()，委托给底层实现。"""
        return self._embedding_function.name()


def create_embedding_function():
    """
    创建 Embedding 函数实例。
    优先使用 OpenAI 兼容 API，若配置缺失则回退到简单随机向量。
    """
    api_key = settings.embedding_api_key
    base_url = settings.embedding_base_url
    model = settings.embedding_model
    
    if api_key and api_key != "your_embedding_api_key_here":
        logger.info(f"使用 OpenAI 兼容 Embedding API: {model} @ {base_url}")
        return OpenAICompatibleEmbeddingFunction(
            api_key=api_key,
            base_url=base_url,
            model_name=model,
        )
    else:
        logger.warning("未配置 Embedding API，使用简单随机向量（仅用于测试）")
        return SimpleEmbeddingFunction()


class SimpleEmbeddingFunction:
    """
    简单的本地嵌入函数类，满足 ChromaDB 的接口要求。
    返回固定维度的随机向量，用于避免从网络下载默认嵌入模型。
    """
    def __init__(self):
        pass

    def __call__(self, input):
        import numpy as np
        return [np.random.rand(384).tolist() for _ in input]

    def embed_documents(self, input) -> List[List[float]]:
        """ChromaDB add 路径。"""
        return self(input)

    def embed_query(self, input) -> List[List[float]]:
        """ChromaDB query 路径。

        chromadb 1.5 以 input=[text] 调用并期望返回 [embedding]
        （单元素嵌入列表），缺失此方法或返回单个向量都会导致
        本地降级模式检索静默失败。
        """
        texts = input if isinstance(input, list) else [input]
        return self(texts)

    @staticmethod
    def name():
        return "simple_local_embedding"


simple_embedding_function = SimpleEmbeddingFunction()


class VectorStore:
    """
    基于 ChromaDB 的向量存储，管理对话记忆和用户画像。
    """

    def __init__(self):
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
        
        self.embedding_fn = create_embedding_function()
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        
        self.conversation_collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_CONVERSATION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        self.profile_collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_PROFILE,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        self.classification_collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_CLASSIFICATION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_fn,
        )
        
        self._init_classification_seeds()
        
        logger.info(
            f"ChromaDB 初始化完成，对话记录数: {self.conversation_collection.count()}, "
            f"画像记录数: {self.profile_collection.count()}, "
            f"分类种子数: {self.classification_collection.count()}"
        )
    
    def _init_classification_seeds(self):
        """
        初始化分类种子数据到向量库。
        从 JSON 文件加载标注好的示例，用于向量相似度分类。
        """
        if self.classification_collection.count() > 0:
            logger.info("分类种子数据已存在，跳过初始化")
            return
        
        seeds_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "classification_seeds.json",
        )
        
        if not os.path.exists(seeds_file):
            logger.warning(f"分类种子文件不存在: {seeds_file}")
            return
        
        try:
            with open(seeds_file, "r", encoding="utf-8") as f:
                seeds = json.load(f)

            texts = [seed["text"] for seed in seeds]
            metadatas = [{"category": seed["category"]} for seed in seeds]
            ids = [f"seed_{i}" for i in range(len(seeds))]

            # DashScope Embedding API 限制单次批量 <= 10 条，分批插入
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                self.classification_collection.add(
                    documents=texts[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    ids=ids[i : i + batch_size],
                )
            logger.info(f"分类种子数据初始化完成，共 {len(seeds)} 条")
        except Exception as e:
            logger.error(f"加载分类种子数据失败: {e}")

    def store_conversation(
        self,
        user_id: str,
        text: str,
        emotion: str,
        category: str,
        timestamp: str,
        role: str = "user",
        memory_type: str = "episode",
    ) -> None:
        """
        存储一条对话记录到向量库。

        Args:
            user_id: 用户唯一标识
            text: 消息文本内容
            emotion: 情感标签（positive/negative/neutral）
            category: 消息类别（coding/career/emotional/chitchat）
            timestamp: 时间戳
            role: 角色（user/assistant）
            memory_type: 记忆类型（episode=原始对话 / fact=提取的结构化事实）
        """
        doc_id = f"{user_id}_{role}_{uuid.uuid4().hex[:8]}"
        metadata = {
            "user_id": user_id,
            "emotion": emotion,
            "category": category,
            "timestamp": timestamp,
            "role": role,
            "memory_type": memory_type,
            "retrieval_count": "0",  # 初始检索次数为 0
        }
        try:
            self.conversation_collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            )
            logger.info(
                f"对话已存储: user={user_id}, category={category}, "
                f"emotion={emotion}, memory_type={memory_type}"
            )
        except Exception as e:
            logger.error(f"存储对话失败: {e}")

    def store_fact(self, user_id: str, fact: str, timestamp: str) -> None:
        """
        存储一条从对话中提取的结构化用户事实（高权重记忆）。

        事实记忆与原始对话共用 conversations collection，
        通过 memory_type="fact" 区分，检索时享受加成系数。

        Args:
            user_id: 用户唯一标识
            fact: 事实文本（如"用户目标岗位是字节后端"）
            timestamp: 时间戳
        """
        self.store_conversation(
            user_id=user_id,
            text=fact,
            emotion="neutral",
            category="fact",
            timestamp=timestamp,
            role="user",
            memory_type="fact",
        )

    def retrieve_facts(
        self, user_id: str, query: str, top_k: int = 2
    ) -> List[Dict[str, Any]]:
        """
        事实记忆检索路径：只检索 memory_type="fact" 的记忆。

        与 retrieve_memories（原文路径）配合，实现"事实 + 原文"双路召回。

        Args:
            user_id: 用户唯一标识
            query: 查询文本
            top_k: 返回的最大事实条数

        Returns:
            事实记忆列表（含 type="fact" 标记）
        """
        try:
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=top_k * 3,
                where={"$and": [{"user_id": user_id}, {"memory_type": "fact"}]},
            )

            facts = []
            if results and results["documents"] and results["documents"][0]:
                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    similarity = 1.0 / (1.0 + distance)
                    memory = {
                        "text": doc,
                        "emotion": meta.get("emotion", "neutral"),
                        "timestamp": meta.get("timestamp", ""),
                        "category": meta.get("category", "fact"),
                        "role": meta.get("role", "user"),
                        "memory_type": "fact",
                        "similarity": round(similarity, 4),
                        "retrieval_count": int(meta.get("retrieval_count", "0")),
                    }
                    memory["final_score"] = self._calculate_memory_weight(memory)
                    facts.append(memory)
                    if len(facts) >= top_k:
                        break

            facts.sort(key=lambda x: x["final_score"], reverse=True)
            return facts
        except Exception as e:
            logger.error(f"事实记忆检索失败: {e}")
            return []

    def get_all_facts(self, user_id: str, limit: int = 50) -> List[str]:
        """
        获取用户全部事实记忆（供周期性反思使用）。

        Args:
            user_id: 用户唯一标识
            limit: 最大返回条数

        Returns:
            事实文本列表
        """
        try:
            results = self.conversation_collection.get(
                where={"$and": [{"user_id": user_id}, {"memory_type": "fact"}]},
                limit=limit,
            )
            if results and results["documents"]:
                return list(results["documents"])
            return []
        except Exception as e:
            logger.error(f"获取事实记忆失败: {e}")
            return []

    def get_facts_with_metadata(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取用户全部事实记忆（含 id / timestamp，供记忆管理面板使用）。

        与 get_all_facts 的区别：返回记录 id，用于删除/纠正时精确定位。

        Args:
            user_id: 用户唯一标识
            limit: 最大返回条数

        Returns:
            [{"id", "text", "timestamp"}] 列表
        """
        try:
            results = self.conversation_collection.get(
                where={"$and": [{"user_id": user_id}, {"memory_type": "fact"}]},
                limit=limit,
            )
            facts = []
            if results and results["documents"]:
                for doc_id, doc, meta in zip(
                    results["ids"], results["documents"], results["metadatas"]
                ):
                    facts.append(
                        {
                            "id": doc_id,
                            "text": doc,
                            "timestamp": meta.get("timestamp", ""),
                        }
                    )
            return facts
        except Exception as e:
            logger.error(f"获取事实记忆（含元数据）失败: {e}")
            return []

    def delete_fact(self, fact_id: str) -> Dict[str, Any]:
        """
        删除一条事实记忆（用户可控记忆：删除错误/过时事实）。

        Args:
            fact_id: 事实记录 id（来自 get_facts_with_metadata）

        Returns:
            {"success": bool, "message": str}
        """
        try:
            self.conversation_collection.delete(ids=[fact_id])
            logger.info(f"事实记忆已删除: {fact_id}")
            return {"success": True, "message": "事实已删除"}
        except Exception as e:
            logger.error(f"删除事实记忆失败: {e}")
            return {"success": False, "message": str(e)}

    def update_fact(self, fact_id: str, new_text: str) -> Dict[str, Any]:
        """
        纠正一条事实记忆（用户可控记忆：修正 AI 记错的表述）。

        直接更新对应记录的文本，时间戳与元数据保持不变。

        Args:
            fact_id: 事实记录 id
            new_text: 纠正后的事实文本

        Returns:
            {"success": bool, "message": str}
        """
        if not new_text or not new_text.strip():
            return {"success": False, "message": "事实文本不能为空"}
        try:
            # ChromaDB update 需同时提供 documents，metadata 不传则保留原值
            self.conversation_collection.update(
                ids=[fact_id],
                documents=[new_text.strip()],
            )
            logger.info(f"事实记忆已纠正: {fact_id} -> {new_text[:50]}")
            return {"success": True, "message": "事实已纠正"}
        except Exception as e:
            logger.error(f"纠正事实记忆失败: {e}")
            return {"success": False, "message": str(e)}

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        apply_decay: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        根据查询文本检索与当前消息最相关的历史对话。
        使用向量相似度检索，支持时间衰减和使用频率加权。

        Args:
            user_id: 用户唯一标识
            query: 查询文本
            top_k: 返回最相似的 top_k 条记录
            apply_decay: 是否应用权重衰减（默认 True）

        Returns:
            包含 text, emotion, timestamp, category, role, final_score 的字典列表
        """
        try:
            # 向量相似度检索：带 user_id + 排除 fact 的 where 过滤，
            # 避免多用户场景下其他用户记忆挤占 top_k 名额
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=top_k * 3,
                where={"$and": [
                    {"user_id": user_id},
                    {"memory_type": {"$ne": "fact"}},
                ]},
            )

            memories = []
            if results and results["documents"] and results["documents"][0]:
                for doc_id, doc, meta, distance in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    # 双保险过滤（where 已限定，此处兜底）
                    if meta.get("user_id") != user_id:
                        continue
                    if meta.get("memory_type") == "fact":
                        continue

                    # ChromaDB 返回的是距离，越小越相似，转换为相似度分数
                    similarity = 1.0 / (1.0 + distance)

                    memory = {
                        "text": doc,
                        "emotion": meta.get("emotion", "unknown"),
                        "timestamp": meta.get("timestamp", ""),
                        "category": meta.get("category", ""),
                        "role": meta.get("role", "user"),
                        "memory_type": meta.get("memory_type", "episode"),
                        "similarity": round(similarity, 4),
                        "retrieval_count": int(meta.get("retrieval_count", "0")),
                    }

                    # 应用权重衰减
                    if apply_decay:
                        memory["final_score"] = self._calculate_memory_weight(
                            memory
                        )
                    else:
                        memory["final_score"] = similarity

                    memories.append(memory)

                    # 更新检索次数（频率因子数据来源）
                    self._increment_retrieval_count(doc_id, meta)

                    if len(memories) >= top_k:
                        break
            
            # 按 final_score 降序排序
            memories.sort(key=lambda x: x["final_score"], reverse=True)
            
            logger.info(
                f"向量检索到 {len(memories)} 条相关记忆 (user={user_id}, decay={apply_decay})"
            )
            return memories
        except Exception as e:
            logger.error(f"向量检索记忆失败: {e}")
            return []

    def _calculate_memory_weight(self, memory: Dict[str, Any]) -> float:
        """
        计算记忆的权重（时间衰减 + 使用频率加权）。

        公式：final_score = similarity * time_weight * (1 + freq_weight)

        Args:
            memory: 记忆字典

        Returns:
            最终权重分数
        """
        similarity = memory.get("similarity", 0.5)
        timestamp = memory.get("timestamp", "")
        retrieval_count = memory.get("retrieval_count", 0)

        # 1. 时间衰减（遗忘曲线）
        time_weight = self._calculate_time_decay(timestamp)

        # 2. 使用频率加权
        freq_weight = min(retrieval_count * 0.1, 0.5)  # 最多增加 50%

        # 3. 综合权重（结构化事实记忆享受加成系数，优先于零散原文）
        fact_boost = (
            settings.MEMORY_FACT_BOOST
            if memory.get("memory_type") == "fact"
            else 1.0
        )
        final_score = similarity * time_weight * (1 + freq_weight) * fact_boost

        return round(final_score, 4)

    def _calculate_time_decay(self, timestamp: str, decay_rate: float = 0.95) -> float:
        """
        计算时间衰减权重（模拟遗忘曲线）。

        公式：time_weight = decay_rate ^ days_old

        Args:
            timestamp: 时间戳字符串（格式：YYYY-MM-DD HH:MM:SS）
            decay_rate: 衰减率（默认 0.95，表示每天衰减 5%）

        Returns:
            时间权重（0~1）
        """
        try:
            from datetime import datetime

            if not timestamp:
                return 0.5  # 没有时间戳，返回中等权重

            # 解析时间戳
            memory_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            days_old = (now - memory_time).days

            # 计算衰减权重
            time_weight = decay_rate ** days_old

            return max(time_weight, 0.1)  # 最低权重 0.1
        except Exception as e:
            logger.warning(f"计算时间衰减失败: {e}")
            return 0.5  # 解析失败，返回中等权重

    def _increment_retrieval_count(self, doc_id: str, meta: Dict[str, Any]) -> None:
        """
        增加记忆的检索次数并落库（三因子加权中"频率"因子的数据来源）。

        Args:
            doc_id: 记录 id（来自查询结果的 ids）
            meta: 元数据
        """
        try:
            if not doc_id:
                return

            current_count = int(meta.get("retrieval_count", "0"))
            new_count = current_count + 1

            # 更新 metadata（ChromaDB update 支持只改 metadata，文本保留原值）
            new_meta = meta.copy()
            new_meta["retrieval_count"] = str(new_count)
            self.conversation_collection.update(
                ids=[doc_id],
                metadatas=[new_meta],
            )
            logger.debug(f"记忆检索次数更新: {doc_id} ({current_count} -> {new_count})")
        except Exception as e:
            logger.warning(f"更新检索次数失败: {e}")

    def classify_message(self, text: str, top_k: int = 3) -> str:
        """
        基于向量相似度对消息进行分类。

        Args:
            text: 用户消息
            top_k: 检索最相似的 top_k 个示例

        Returns:
            分类结果（coding/career/emotional/chitchat）
        """
        category, _ = self.classify_message_with_confidence(text, top_k)
        return category

    def classify_message_with_confidence(
        self, text: str, top_k: int = 3
    ) -> tuple:
        """
        基于向量相似度对消息进行分类，并返回置信度。

        Args:
            text: 用户消息
            top_k: 检索最相似的 top_k 个示例

        Returns:
            (category, confidence) 分类结果和置信度
        """
        try:
            results = self.classification_collection.query(
                query_texts=[text],
                n_results=top_k,
            )
            
            if not results or not results["metadatas"] or not results["metadatas"][0]:
                logger.warning("向量分类未返回结果，回退到 chitchat")
                return "chitchat", 0.5
            
            # 统计类别投票
            category_votes = {}
            category_distances = {}
            
            for meta, distance in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                category = meta["category"]
                category_votes[category] = category_votes.get(category, 0) + 1
                
                # 记录该类别的最小距离（最相似的距离）
                if category not in category_distances:
                    category_distances[category] = distance
                else:
                    category_distances[category] = min(
                        category_distances[category], distance
                    )
            
            # 获取最高票数的类别
            best_category = max(category_votes, key=category_votes.get)
            best_votes = category_votes[best_category]
            
            # 计算置信度
            # 1. 投票比例（最高票数占总票数的比例）
            vote_ratio = best_votes / top_k
            
            # 2. 距离相似度（转换为 0-1 范围）
            best_distance = category_distances.get(best_category, 1.0)
            distance_similarity = 1.0 / (1.0 + best_distance)
            
            # 3. 综合置信度（投票权重 60% + 距离权重 40%）
            confidence = vote_ratio * 0.6 + distance_similarity * 0.4
            
            logger.info(
                f"向量分类结果: {best_category} "
                f"(confidence={confidence:.2f}, votes={category_votes})"
            )
            return best_category, min(confidence, 1.0)
        except Exception as e:
            logger.error(f"向量分类失败: {e}")
            return "chitchat", 0.5

    def get_similar_seeds(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        获取与文本最相似的分类种子（供 GuardAgent LLM 仲裁做 few-shot 示例）。

        Args:
            text: 用户消息
            top_k: 返回最相似的 top_k 个种子

        Returns:
            [{"text": ..., "category": ..., "distance": ...}] 列表
        """
        try:
            results = self.classification_collection.query(
                query_texts=[text],
                n_results=top_k,
            )
            seeds = []
            if results and results["documents"] and results["documents"][0]:
                for doc, meta, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    seeds.append(
                        {
                            "text": doc,
                            "category": meta.get("category", "chitchat"),
                            "distance": round(distance, 4),
                        }
                    )
            return seeds
        except Exception as e:
            logger.error(f"获取相似种子失败: {e}")
            return []

    def add_classification_seed(
        self, text: str, category: str, source: str = "feedback"
    ) -> Dict[str, Any]:
        """
        添加一条分类种子（错分反馈闭环的核心）。

        用户纠正错分后，将 (消息, 正确类别) 作为新种子写入向量库，
        后续同类消息的向量分类即可命中，实现分类系统自我改进。

        Args:
            text: 被错分的消息原文
            category: 用户纠正后的正确类别
            source: 种子来源（feedback=错分回流 / manual=手动补充）

        Returns:
            {"success": bool, "seed_id": str, "message": str}
        """
        if category not in ("coding", "career", "emotional", "chitchat"):
            return {
                "success": False,
                "seed_id": "",
                "message": f"非法类别: {category}",
            }

        seed_id = f"seed_{source}_{uuid.uuid4().hex[:8]}"
        try:
            self.classification_collection.add(
                documents=[text],
                metadatas=[{"category": category, "source": source}],
                ids=[seed_id],
            )
            total = self.classification_collection.count()
            logger.info(
                f"分类种子回流成功: {seed_id} -> {category} "
                f"(source={source}, 当前种子总数={total})"
            )
            return {
                "success": True,
                "seed_id": seed_id,
                "message": f"种子已回流，当前种子总数: {total}",
            }
        except Exception as e:
            logger.error(f"分类种子回流失败: {e}")
            return {"success": False, "seed_id": "", "message": str(e)}

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户画像。若不存在则返回默认画像。

        用户画像包含：
          - learning_goal: 学习目标
          - current_skill_level: 当前技能水平
          - job_target: 求职目标
          - emotional_trend: 最近5次情绪得分列表
        """
        profile_id = f"user_profile_{user_id}"
        try:
            results = self.profile_collection.get(ids=[profile_id])
            if results and results["documents"]:
                profile_data = json.loads(results["documents"][0])
                logger.info(f"获取用户画像成功: user={user_id}")
                return profile_data
        except Exception as e:
            logger.warning(f"获取用户画像失败: {e}")

        default_profile = {
            "learning_goal": "找算法实习",
            "current_skill_level": "Python基础",
            "job_target": "AI开发",
            "emotional_trend": [],
        }
        self.save_user_profile(user_id, default_profile)
        return default_profile

    def save_user_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        """
        保存或更新用户画像。
        """
        profile_id = f"user_profile_{user_id}"
        try:
            profile_json = json.dumps(profile, ensure_ascii=False)
            metadata = {"user_id": user_id, "type": "profile"}
            # upsert：存在则更新，不存在则插入。
            # 不能用 update + add 回退——chromadb 1.x 的 update 对不存在的
            # id 静默忽略（不抛异常），新用户画像将永远无法写入
            self.profile_collection.upsert(
                ids=[profile_id],
                documents=[profile_json],
                metadatas=[metadata],
            )
            logger.info(f"用户画像已保存: user={user_id}")
        except Exception as e:
            logger.error(f"保存用户画像失败: {e}")

    def update_emotional_trend(
        self, user_id: str, emotion_score: float, max_trend_length: int = 5
    ) -> List[float]:
        """
        更新用户画像中的情绪趋势记录。
        保留最近 max_trend_length 次的情绪得分。

        Args:
            user_id: 用户唯一标识
            emotion_score: 本次情绪得分（0~1）
            max_trend_length: 保留的最大趋势长度

        Returns:
            更新后的情绪趋势列表
        """
        profile = self.get_user_profile(user_id)
        trend = profile.get("emotional_trend", [])
        trend.append(emotion_score)
        trend = trend[-max_trend_length:]
        profile["emotional_trend"] = trend
        self.save_user_profile(user_id, profile)
        return trend

    def get_recent_emotions(
        self, user_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        获取用户最近的对话情绪记录，用于前端情绪仪表盘展示。

        Args:
            user_id: 用户唯一标识
            limit: 返回的最大记录数

        Returns:
            包含 emotion, timestamp, score 的字典列表
        """
        try:
            results = self.conversation_collection.get(
                where={"user_id": user_id, "role": "user"},
                limit=limit,
            )
            emotions = []
            if results and results["metadatas"]:
                for meta in reversed(results["metadatas"]):
                    emotions.append(
                        {
                            "emotion": meta.get("emotion", "neutral"),
                            "timestamp": meta.get("timestamp", ""),
                            "category": meta.get("category", ""),
                        }
                    )
            return emotions[:limit]
        except Exception as e:
            logger.error(f"获取近期情绪记录失败: {e}")
            return []

    def proactive_memory_retrieval(
        self,
        user_id: str,
        current_emotion: str,
        emotion_score: float,
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        主动记忆检索：根据用户当前情绪状态推送相关记忆。

        规则：
          - 情绪低落（negative, score < 0.4）→ 推送过去的成功经历
          - 情绪稳定（neutral）→ 推送学习笔记
          - 情绪积极（positive, score > 0.7）→ 推送进步记录

        Args:
            user_id: 用户唯一标识
            current_emotion: 当前情绪标签
            emotion_score: 当前情绪分数
            top_k: 返回的记忆数量

        Returns:
            主动推送的记忆列表
        """
        try:
            # 根据情绪状态选择检索关键词
            if current_emotion == "negative" and emotion_score < 0.4:
                # 情绪低落 → 推送成功经历
                query = "成功 完成 开心 解决 掌握 进步"
                logger.info(
                    f"主动记忆检索: 情绪低落，推送成功经历 (user={user_id})"
                )
            elif current_emotion == "positive" and emotion_score > 0.7:
                # 情绪积极 → 推送进步记录
                query = "进步 提升 学会 理解 突破"
                logger.info(
                    f"主动记忆检索: 情绪积极，推送进步记录 (user={user_id})"
                )
            else:
                # 情绪稳定 → 推送学习笔记
                query = "学习 笔记 总结 复习 知识点"
                logger.info(
                    f"主动记忆检索: 情绪稳定，推送学习笔记 (user={user_id})"
                )

            # 检索相关记忆
            memories = self.retrieve_memories(
                user_id=user_id,
                query=query,
                top_k=top_k,
                apply_decay=True,
            )

            # 添加主动推送标记
            for mem in memories:
                mem["proactive"] = True

            return memories
        except Exception as e:
            logger.error(f"主动记忆检索失败: {e}")
            return []

    def generate_summary(self, user_id: str, query: str = "学习情况总结") -> str:
        """
        深度加分项：长期记忆的总结能力。
        检索更多历史记录，生成可读的摘要文本供 LLM 进一步总结。

        Args:
            user_id: 用户唯一标识
            query: 用于检索的查询文本

        Returns:
            格式化的历史记录文本
        """
        memories = self.retrieve_memories(user_id, query, top_k=10)
        profile = self.get_user_profile(user_id)
        if not memories:
            return "暂无足够的历史记录生成摘要。"

        lines = [f"用户画像: {json.dumps(profile, ensure_ascii=False)}", "", "近期对话记录:"]
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"  {i}. [{mem.get('timestamp', '')}] "
                f"(情绪:{mem.get('emotion', '')}, 类别:{mem.get('category', '')}) "
                f"{mem.get('text', '')}"
            )
        return "\n".join(lines)

    def compress_memories(
        self,
        user_id: str,
        max_memories: int = 200,
        keep_recent: int = 100,
    ) -> Dict[str, Any]:
        """
        记忆容量治理：episode 记忆超量时真实删除陈旧条目。

        规则：
          1. 只统计 episode 记忆（用户消息 + AI 回复），fact 记忆永不删除
          2. episode 数量超过 max_memories 时触发压缩
          3. 按 timestamp 降序保留最近 keep_recent 条，其余从向量库真实删除
          4. 压缩统计写入用户画像 memory_compression 字段（保留审计痕迹）

        Args:
            user_id: 用户唯一标识
            max_memories: episode 记忆数量阈值（超过才触发压缩）
            keep_recent: 压缩后保留的最近记忆条数

        Returns:
            压缩统计信息字典
        """
        try:
            # 获取用户所有 episode 记忆（覆盖 user/assistant 两种 role）
            results = self.conversation_collection.get(
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"memory_type": {"$ne": "fact"}},
                    ]
                },
                limit=5000,
            )

            if not results or not results["documents"]:
                return {"status": "no_memories", "message": "没有记忆需要压缩"}

            total_memories = len(results["documents"])

            if total_memories <= max_memories:
                return {
                    "status": "no_need",
                    "message": f"记忆数量 ({total_memories}) 未超过阈值 ({max_memories})",
                }

            # 按时间降序排序（timestamp 为 "YYYY-MM-DD HH:MM:SS" 字符串，可直接比较）
            memories_with_meta = list(
                zip(results["documents"], results["metadatas"], results["ids"])
            )
            memories_with_meta.sort(
                key=lambda x: x[1].get("timestamp", ""), reverse=True
            )

            # 保留最近 keep_recent 条，其余真实删除
            recent_memories = memories_with_meta[:keep_recent]
            stale_memories = memories_with_meta[keep_recent:]
            stale_ids = [doc_id for _, _, doc_id in stale_memories]

            if stale_ids:
                self.conversation_collection.delete(ids=stale_ids)

            # 将被删记忆按类别分组，统计写入画像（保留压缩痕迹，不存原文）
            category_groups = {}
            for doc, meta, doc_id in stale_memories:
                category = meta.get("category", "unknown")
                category_groups[category] = category_groups.get(category, 0) + 1

            profile = self.get_user_profile(user_id)
            profile["memory_compression"] = {
                "last_compressed": self._get_current_timestamp(),
                "total_before": total_memories,
                "deleted": len(stale_ids),
                "kept_recent": len(recent_memories),
                "category_deleted": category_groups,
            }
            self.save_user_profile(user_id, profile)

            logger.info(
                f"记忆压缩完成: user={user_id}, "
                f"before={total_memories}, deleted={len(stale_ids)}, "
                f"kept={len(recent_memories)}, categories={category_groups}"
            )

            return {
                "status": "compressed",
                "total_before": total_memories,
                "deleted": len(stale_ids),
                "kept_recent": len(recent_memories),
                "category_deleted": category_groups,
            }
        except Exception as e:
            logger.error(f"记忆压缩失败: {e}")
            return {"status": "error", "message": str(e)}

    def _get_current_timestamp(self) -> str:
        """获取当前时间戳字符串"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


vector_store = VectorStore()
