"""
记忆召回评测 —— run_memory_eval.py

职责：
  构造带 ground truth 的记忆库，评测记忆检索质量：
    - Recall@k：目标记忆是否出现在 top-k 召回中
    - MRR：目标记忆的排名倒数均值
    - 双路召回：事实记忆路径 vs 原文记忆路径
    - 三因子加权效果：相似度 × 时间衰减 × 频率（含事实加成）

设计理由：
  - 用独立的评测用户（memory_eval_user）预写记忆，不污染真实用户数据；
  - 每条记忆带唯一语义锚点（如"Redis 分布式锁"），查询与锚点语义相关
    但表述不同（同义改写），评测语义检索而非字面匹配；
  - 时间衰减/频率权重用纯函数直接单测（可复现、不依赖网络）。

用法：
  python evaluation/run_memory_eval.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from companion_ai.utils.helpers import get_timestamp
from companion_ai.utils.logger import logger

EVAL_USER = "memory_eval_user"

# 预置记忆（episode=原文对话 / fact=结构化事实），每条含唯一语义锚点。
# 规模按真实用户对话密度设计（20 episode + 12 fact），未被查询命中的
# 记忆充当干扰项，考验排序而非仅有目标记忆的"空库检索"。
EPISODES = [
    ("用户问过如何用 Redis 实现分布式锁，讨论了 SETNX 和 RedLock 的区别", "coding"),
    ("用户提到在准备字节跳动后端开发实习的面试", "career"),
    ("用户说连续刷题刷到深夜，感觉身体有点吃不消", "emotional"),
    ("用户讨论过 TCP 拥塞控制的慢启动过程", "coding"),
    ("用户聊到喜欢在周末爬山放松", "chitchat"),
    ("用户询问过秋招简历上项目经历怎么写", "career"),
    ("用户说过和实验室师兄因为论文署名闹了矛盾", "emotional"),
    ("用户问过 Python asyncio 事件循环的原理", "coding"),
    ("用户讨论过 MySQL 索引失效的几种场景", "coding"),
    ("用户聊到毕业以后想去杭州工作", "career"),
    ("用户提到导师最近催论文进度，压力比较大", "emotional"),
    ("用户讨论过 Kafka 在哪些情况下会丢消息", "coding"),
    ("用户问过用前缀和优化区间求和的思路", "coding"),
    ("用户说习惯每天早上背英语单词", "chitchat"),
    ("用户聊过无领导小组讨论的面试技巧", "career"),
    ("用户担心自己的项目经历缺少亮点", "career"),
    ("用户讨论过 HTTP/2 多路复用与队头阻塞", "coding"),
    ("用户说过最近失眠，入睡比较困难", "emotional"),
    ("用户聊到平时喜欢听粤语老歌", "chitchat"),
    ("用户问过动态规划背包问题应该怎么入手", "coding"),
    ("用户问过快速排序最坏情况为什么会退化", "coding"),
    ("用户聊到过毕业旅行想去川西", "chitchat"),
    ("用户讨论过 Docker 镜像分层缓存的原理", "coding"),
    ("用户说过导师允许实习，但要求先完成小论文", "career"),
    ("用户问过线程池核心参数怎么设置", "coding"),
    ("用户提到焦虑的时候会长跑减压", "emotional"),
    ("用户聊过和对象因为异地产生过分歧", "emotional"),
    ("用户讨论过 B+ 树和哈希索引的选型", "coding"),
    ("用户问过秋招提前批什么时候开始", "career"),
    ("用户说过想养一只橘猫", "chitchat"),
    ("用户问过爬楼梯问题的递推解法", "coding"),
    ("用户聊到宿舍楼下新开了一家麻辣烫", "chitchat"),
]

FACTS = [
    "用户的目标岗位是字节跳动后端开发实习",
    "用户掌握了 Python、Redis 和消息队列",
    "用户正在准备 2026 届秋招",
    "用户每周刷 5 道 LeetCode 题",
    "用户的目标工作城市是杭州",
    "用户的研究方向是推荐系统",
    "用户的英语水平是 CET-6",
    "用户计划 9 月开始投递简历",
    "用户的项目经历包含一个推荐系统 demo",
    "用户偏好后端开发方向而非算法方向",
    "用户每天学习 3 小时",
    "用户的实习意向是大厂暑期实习",
    "用户参加过 ACM 校赛并拿了铜牌",
    "用户的实习期望日薪是 300 元",
    "用户在用王道的书复习操作系统",
    "用户的论文投的是 CCF-B 会议",
    "用户实习最长可接受 6 个月",
    "用户的目标公司还包括腾讯和阿里",
    "用户的 GitHub 上有 8 个仓库",
    "用户准备把推荐系统 demo 写进简历",
    "用户不接受单休的工作",
    "用户的论文答辩安排在 5 月",
    "用户习惯用 VSCode 写代码",
    "用户希望第一份工作有导师带教",
]

# 查询与期望命中的记忆（查询是语义改写，非字面匹配；24 条 = 12 事实型 + 12 原文型）
QUERIES = [
    # ---- 事实路径查询 ----
    {"query": "我想投的实习方向之前跟你说过的",
     "expect_fact": "用户的目标岗位是字节跳动后端开发实习"},
    {"query": "我刷题的频率你还记得吗",
     "expect_fact": "用户每周刷 5 道 LeetCode 题"},
    {"query": "我秋招的时间节点之前聊过",
     "expect_fact": "用户正在准备 2026 届秋招"},
    {"query": "我之前说过想去哪个城市工作来着",
     "expect_fact": "用户的目标工作城市是杭州"},
    {"query": "我的研究生方向是什么，之前提过",
     "expect_fact": "用户的研究方向是推荐系统"},
    {"query": "我的英语水平怎么样，之前说过的",
     "expect_fact": "用户的英语水平是 CET-6"},
    {"query": "简历什么时候开始投，我之前说过计划的",
     "expect_fact": "用户计划 9 月开始投递简历"},
    {"query": "我跟你说过我做过的项目有哪些吗",
     "expect_fact": "用户的项目经历包含一个推荐系统 demo"},
    {"query": "后端和算法之间我之前选的哪个方向",
     "expect_fact": "用户偏好后端开发方向而非算法方向"},
    {"query": "我一般每天花多少时间学习来着",
     "expect_fact": "用户每天学习 3 小时"},
    {"query": "暑期实习我是怎么打算的，之前聊过",
     "expect_fact": "用户的实习意向是大厂暑期实习"},
    {"query": "我掌握的技术栈你还记得吗",
     "expect_fact": "用户掌握了 Python、Redis 和消息队列"},
    # ---- 原文路径查询 ----
    {"query": "我之前问过怎么防止多台机器同时执行定时任务",
     "expect_episode": "用户问过如何用 Redis 实现分布式锁，讨论了 SETNX 和 RedLock 的区别"},
    {"query": "网络拥塞那块我以前好像没搞明白",
     "expect_episode": "用户讨论过 TCP 拥塞控制的慢启动过程"},
    {"query": "协程那块我之前请教过你问题",
     "expect_episode": "用户问过 Python asyncio 事件循环的原理"},
    {"query": "数据库里哪些情况索引会失效，我们聊过的",
     "expect_episode": "用户讨论过 MySQL 索引失效的几种场景"},
    {"query": "消息队列会不会丢消息，之前讨论过吗",
     "expect_episode": "用户讨论过 Kafka 在哪些情况下会丢消息"},
    {"query": "区间求和有没有快速的做法，我以前问过",
     "expect_episode": "用户问过用前缀和优化区间求和的思路"},
    {"query": "群面有什么技巧我之前问过你",
     "expect_episode": "用户聊过无领导小组讨论的面试技巧"},
    {"query": "HTTP2 和 HTTP1 的区别之前聊过",
     "expect_episode": "用户讨论过 HTTP/2 多路复用与队头阻塞"},
    {"query": "背包问题当时是怎么入门的，我们讨论过",
     "expect_episode": "用户问过动态规划背包问题应该怎么入手"},
    {"query": "我和师兄那次不愉快你还记得吗",
     "expect_episode": "用户说过和实验室师兄因为论文署名闹了矛盾"},
    {"query": "之前失眠的时候我跟你说过的",
     "expect_episode": "用户说过最近失眠，入睡比较困难"},
    {"query": "我以前提过毕业以后的打算",
     "expect_episode": "用户聊到毕业以后想去杭州工作"},
    # ---- 事实路径查询（扩充批） ----
    {"query": "我之前参加过什么编程比赛吗",
     "expect_fact": "用户参加过 ACM 校赛并拿了铜牌"},
    {"query": "实习工资我之前说的是多少来着",
     "expect_fact": "用户的实习期望日薪是 300 元"},
    {"query": "操作系统我是用什么资料复习的",
     "expect_fact": "用户在用王道的书复习操作系统"},
    {"query": "我的论文投的什么级别的会议",
     "expect_fact": "用户的论文投的是 CCF-B 会议"},
    {"query": "实习期限我最长能接受多久",
     "expect_fact": "用户实习最长可接受 6 个月"},
    {"query": "除了字节我还想去哪些公司",
     "expect_fact": "用户的目标公司还包括腾讯和阿里"},
    {"query": "我开源仓库有多少个来着",
     "expect_fact": "用户的 GitHub 上有 8 个仓库"},
    {"query": "简历上我打算突出哪个项目",
     "expect_fact": "用户准备把推荐系统 demo 写进简历"},
    {"query": "什么样的工作节奏我不会考虑",
     "expect_fact": "用户不接受单休的工作"},
    {"query": "我什么时候答辩来着",
     "expect_fact": "用户的论文答辩安排在 5 月"},
    {"query": "我平时用什么编辑器写代码",
     "expect_fact": "用户习惯用 VSCode 写代码"},
    {"query": "我希望第一份工作有什么条件",
     "expect_fact": "用户希望第一份工作有导师带教"},
    # ---- 原文路径查询（扩充批） ----
    {"query": "排序算法里快排最坏的情况我们讨论过",
     "expect_episode": "用户问过快速排序最坏情况为什么会退化"},
    {"query": "毕业旅行我之前说想去哪",
     "expect_episode": "用户聊到过毕业旅行想去川西"},
    {"query": "Docker 那个缓存加速的原理我们聊过",
     "expect_episode": "用户讨论过 Docker 镜像分层缓存的原理"},
    {"query": "导师对实习的态度我之前提过",
     "expect_episode": "用户说过导师允许实习，但要求先完成小论文"},
    {"query": "线程池的参数怎么配我请教过你",
     "expect_episode": "用户问过线程池核心参数怎么设置"},
    {"query": "心情不好时我之前说过怎么排解",
     "expect_episode": "用户提到焦虑的时候会长跑减压"},
    {"query": "我和对象之前因为什么闹过别扭",
     "expect_episode": "用户聊过和对象因为异地产生过分歧"},
    {"query": "数据库索引用树结构还是哈希我们讨论过",
     "expect_episode": "用户讨论过 B+ 树和哈希索引的选型"},
    {"query": "提前批的时间我之前问过你",
     "expect_episode": "用户问过秋招提前批什么时候开始"},
    {"query": "我想养的宠物你还记得吗",
     "expect_episode": "用户说过想养一只橘猫"},
    {"query": "上楼梯那道题我们之前讨论过思路",
     "expect_episode": "用户问过爬楼梯问题的递推解法"},
    {"query": "我之前说楼下开了家什么店",
     "expect_episode": "用户聊到宿舍楼下新开了一家麻辣烫"},
]


def setup():
    """写入评测记忆。"""
    from companion_ai.memory.vector_store import vector_store

    cleanup()
    ts = get_timestamp()
    for text, category in EPISODES:
        vector_store.store_conversation(
            user_id=EVAL_USER, text=text, emotion="neutral", category=category,
            timestamp=ts, role="user", memory_type="episode",
        )
    for fact in FACTS:
        vector_store.store_fact(user_id=EVAL_USER, fact=fact, timestamp=ts)
    logger.info(f"评测记忆写入完成: {len(EPISODES)} episodes + {len(FACTS)} facts")


def cleanup():
    from companion_ai.memory.vector_store import vector_store

    vector_store.conversation_collection.delete(where={"user_id": EVAL_USER})
    vector_store.profile_collection.delete(ids=[f"user_profile_{EVAL_USER}"])


def eval_retrieval():
    """检索评测：Recall@k 与 MRR（episode/fact 各自路径）。"""
    from companion_ai.memory.vector_store import vector_store

    episode_hits, fact_hits = 0, 0
    mrr_sum = 0.0
    details = []

    for q in QUERIES:
        # 事实路径（top-2，与线上 MemoryAgent 一致）
        facts = vector_store.retrieve_facts(EVAL_USER, q["query"], top_k=2)
        # 原文路径（top-3，与线上 MemoryAgent 一致）
        episodes = vector_store.retrieve_memories(EVAL_USER, q["query"], top_k=3)

        rank = None
        if "expect_fact" in q:
            fact_hits += any(f["text"] == q["expect_fact"] for f in facts)
            for i, f in enumerate(facts, 1):
                if f["text"] == q["expect_fact"]:
                    rank = i
                    break
        if "expect_episode" in q:
            episode_hits += any(m["text"] == q["expect_episode"] for m in episodes)
            for i, m in enumerate(episodes, 1):
                if m["text"] == q["expect_episode"]:
                    rank = i if rank is None else min(rank, i)
                    break
        if rank:
            mrr_sum += 1.0 / rank
        details.append({
            "query": q["query"][:30],
            "fact_hit": any(f["text"] == q.get("expect_fact", "") for f in facts),
            "episode_hit": any(m["text"] == q.get("expect_episode", "") for m in episodes),
            "rank": rank,
        })

    n = len(QUERIES)
    return {
        "queries": n,
        "fact_recall": round(fact_hits / sum(1 for q in QUERIES if "expect_fact" in q), 4),
        "episode_recall": round(episode_hits / sum(1 for q in QUERIES if "expect_episode" in q), 4),
        "mrr": round(mrr_sum / n, 4),
        "details": details,
    }


def eval_weighting():
    """三因子加权纯函数评测：时间衰减 / 频率 / 事实加成的方向性。"""
    from companion_ai.memory.vector_store import vector_store as vs

    now_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    old_ts = "2026-09-01 00:00:00"  # 30 天前

    def weight(similarity, ts, count, memory_type="episode"):
        return vs._calculate_memory_weight({
            "similarity": similarity, "timestamp": ts,
            "retrieval_count": count, "memory_type": memory_type,
        })

    checks = [
        # (名称, 期望: left > right)
        ("新记忆 > 30天前记忆", weight(0.8, now_ts, 0) > weight(0.8, old_ts, 0)),
        ("高频记忆 > 零频记忆", weight(0.8, now_ts, 5) > weight(0.8, now_ts, 0)),
        ("事实记忆 > 同条件原文", weight(0.8, now_ts, 0, "fact") > weight(0.8, now_ts, 0, "episode")),
    ]
    return {
        "checks": [{"name": n, "passed": p} for n, p in checks],
        "all_passed": all(p for _, p in checks),
    }


def write_report(retrieval, weighting):
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(report_dir, f"memory_report_{timestamp}.md")

    lines = [
        "# 记忆系统评测报告",
        "",
        "## 检索质量（双路召回，查询为语义改写非字面匹配）",
        "",
        f"- 查询数: {retrieval['queries']}",
        f"- 事实路径 Recall@2: **{retrieval['fact_recall']:.2%}**",
        f"- 原文路径 Recall@3: **{retrieval['episode_recall']:.2%}**",
        f"- MRR: **{retrieval['mrr']:.3f}**",
        "",
        "| 查询 | 事实命中 | 原文命中 | 排名 |",
        "|---|---|---|---|",
    ]
    for d in retrieval["details"]:
        lines.append(
            f"| {d['query']}... | {d['fact_hit']} | {d['episode_hit']} | {d['rank'] or '-'} |"
        )

    lines += [
        "",
        "## 三因子加权方向性（时间衰减 × 频率 × 事实加成）",
        "",
    ]
    for c in weighting["checks"]:
        lines.append(f"- {'✅' if c['passed'] else '❌'} {c['name']}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入: {filepath}")
    return filepath


if __name__ == "__main__":
    try:
        setup()
        retrieval = eval_retrieval()
        weighting = eval_weighting()
        write_report(retrieval, weighting)
        print(json.dumps({
            "retrieval": {k: v for k, v in retrieval.items() if k != "details"},
            "weighting_all_passed": weighting["all_passed"],
        }, ensure_ascii=False, indent=2))
        assert weighting["all_passed"], "加权方向性检查未全部通过"
    finally:
        cleanup()
        logger.info("评测数据已清理")
