"""
延迟与 Token 剖析 —— run_latency_profile.py

职责：
  用真实工作流跑代表性对话，输出节点级延迟与 LLM token 消耗画像：
    - 每个图节点的耗时（stream updates 模式逐节点打点）
    - 每轮对话的 LLM 调用次数与 prompt/completion/total tokens
      （collect_runs 收集全链路 LLM 调用，无需侵入 Agent 代码）

代表性场景：
  1. coding（贴代码问运行结果 → 触发 MCP 沙箱执行）
  2. career（岗位差距分析 → 触发 MCP 岗位要求查询，即技能 Gap 分析链路）
  3. chitchat（闲聊 → 纯对话路径）

用法：
  python evaluation/run_latency_profile.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from companion_ai.utils.logger import logger

EVAL_USER = "latency_eval_user"
THREAD = f"thread_{EVAL_USER}"

SCENARIOS = [
    {
        "name": "coding+MCP沙箱",
        "message": "这段代码输出什么：print([i*i for i in range(5)])",
    },
    {
        "name": "career+Gap分析",
        "message": "我想做AI Agent开发方向的岗位，帮我分析一下我还差哪些技能",
    },
    {
        "name": "chitchat纯对话",
        "message": "推荐几本好看的书",
    },
]


def profile_one(graph, message):
    """跑一轮工作流，返回节点延迟与 token 统计。"""
    from langchain_core.tracers.context import collect_runs

    node_times = {}
    input_state = {
        "user_id": EVAL_USER,
        "current_message": message,
        "interview_mode": False,
    }
    config = {"configurable": {"thread_id": THREAD}}

    llm_calls = 0
    prompt_tokens = 0
    completion_tokens = 0

    with collect_runs() as cb:
        t0 = time.time()
        for update in graph.stream(input_state, config=config, stream_mode="updates"):
            now = time.time()
            for node_name in update:
                node_times[node_name] = round(now - t0, 2)
            t0 = now
        # 从 run 树提取 LLM 调用的 token 用量（DashScope 兼容模式：
        # token_usage 位于 run.outputs["llm_output"]["token_usage"]）
        for run in cb.traced_runs:
            if run.outputs and "generations" in (run.outputs or {}):
                llm_calls += 1
                try:
                    usage = (run.outputs.get("llm_output") or {}).get("token_usage") or {}
                    prompt_tokens += usage.get("prompt_tokens", 0)
                    completion_tokens += usage.get("completion_tokens", 0)
                except Exception:
                    pass

    return {
        "node_times": node_times,
        "llm_calls": llm_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def main():
    from companion_ai.graph.workflow import compile_graph
    from companion_ai.memory.vector_store import vector_store

    graph = compile_graph()
    results = []

    for sc in SCENARIOS:
        logger.info(f"剖析场景: {sc['name']}")
        t0 = time.time()
        r = profile_one(graph, sc["message"])
        r["name"] = sc["name"]
        r["wall_time"] = round(time.time() - t0, 2)
        results.append(r)

    # 清理评测用户数据
    vector_store.conversation_collection.delete(where={"user_id": EVAL_USER})
    vector_store.profile_collection.delete(ids=[f"user_profile_{EVAL_USER}"])
    logger.info("评测数据已清理")

    # 汇总
    total_llm = sum(r["llm_calls"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    node_agg = {}
    for r in results:
        for n, t in r["node_times"].items():
            node_agg.setdefault(n, []).append(t)

    summary = {
        "scenarios": [
            {
                "name": r["name"],
                "wall_time": r["wall_time"],
                "llm_calls": r["llm_calls"],
                "total_tokens": r["total_tokens"],
                "node_times": r["node_times"],
            }
            for r in results
        ],
        "total_llm_calls": total_llm,
        "total_tokens": total_tokens,
        "avg_tokens_per_turn": round(total_tokens / len(results), 1),
        "node_avg_ms": {n: round(sum(ts) / len(ts) * 1000) for n, ts in node_agg.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 写报告
    report_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(report_dir, f"latency_token_report_{timestamp}.md")

    lines = [
        "# 延迟与 Token 剖析报告",
        "",
        f"- 场景数: {len(results)} | 总 LLM 调用: {total_llm} | 总 token: {total_tokens}",
        f"- 平均每轮 token: {summary['avg_tokens_per_turn']}",
        "",
        "| 场景 | 耗时(s) | LLM调用 | token |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {r['wall_time']} | {r['llm_calls']} | {r['total_tokens']} |")

    lines += ["", "## 节点平均耗时", "", "| 节点 | 平均耗时(ms) |", "|---|---|"]
    for n, ms in sorted(summary["node_avg_ms"].items(), key=lambda x: -x[1]):
        lines.append(f"| {n} | {ms} |")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入: {filepath}")
    return summary


if __name__ == "__main__":
    main()
