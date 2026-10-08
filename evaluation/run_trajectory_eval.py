"""
工具调用轨迹评测 —— run_trajectory_eval.py

职责：
  剧本化评测 Agent 的工具调用轨迹，分三层（由轻到重）：

  Layer 1 路由轨迹（零 LLM 成本，确定性）：
      消息 → GuardAgent 分类 → 条件路由 → 断言到达预期 Agent 节点
  Layer 2 工具选择轨迹（真实 LLM + 记录型工具替身）：
      复用 Agent 真实 prompt，把工具替换为同签名替身（只记录调用与参数、
      返回构造结果），断言 LLM 的工具选择与参数是否符合剧本预期
  Layer 3 工具执行契约（零 LLM 成本）：
      真实工具（本地降级路径）以剧本参数调用，断言输出契约字段

设计理由：
  - Function Calling 的质量 = 选对工具 + 传对参数 + 结果正确回填，
    传统单测 mock 掉 LLM 后无法覆盖"选没选对"这一层，本脚本补齐。
  - Layer 2 用替身而非真实工具：隔离 MCP/沙箱副作用，评测只关注
    LLM 的决策轨迹本身；工具执行正确性由 Layer 3 独立验证。
  - 评测数据只读不写（分类查询不落库），无需清理。

用法：
  python evaluation/run_trajectory_eval.py             # 全部三层
  python evaluation/run_trajectory_eval.py --skip-llm  # 跳过 Layer 2（零 API 成本）

报告输出至 evaluation/reports/，指标可复现。
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from companion_ai.agents.career_agent import _build_career_prompt
from companion_ai.agents.coding_agent import _build_coding_prompt
from companion_ai.agents.guard_agent import _classify_message
from companion_ai.graph.workflow import _route_by_category
from companion_ai.tools.career_tools import (
    evaluate_resume,
    get_interview_questions,
    get_job_requirements,
)
from companion_ai.tools.mcp_tools import execute_code_sandbox
from companion_ai.utils.helpers import invoke_llm_with_tools
from companion_ai.utils.logger import logger


# ============================================================
# 剧本定义
# ============================================================

# Layer 1：路由轨迹剧本（消息 → 预期节点）
ROUTING_SCENARIOS = [
    {"message": "帮我看看这段代码为什么报错：print(undefined_var)", "expected": "coding"},
    {"message": "帮我讲讲动态规划的核心思想", "expected": "coding"},
    {"message": "我想找 AI 算法岗的工作，帮我做下求职规划", "expected": "career"},
    {"message": "帮我看看这份简历怎么改：熟练掌握 Python 和 PyTorch", "expected": "career"},
    {"message": "今天心情有点低落，想找人聊聊", "expected": "general_chat"},
    {"message": "在吗，随便聊两句", "expected": "general_chat"},
    {"message": "这段代码抛了 IndexError，帮我看看哪里有问题", "expected": "coding"},
    {"message": "我想找一份机器学习实习，需要准备些什么", "expected": "career"},
    {"message": "我拿到字节跳动的 offer 了，太开心了！", "expected": "general_chat"},
    {"message": "帮我看看这道 LeetCode 两数之和怎么解", "expected": "coding"},
]

# Layer 2：工具选择轨迹剧本（真实 LLM 决策 + 替身记录）
# expected_tools: 预期调用的工具名列表；param_checks: 参数断言 {工具名: [必备参数]}
TOOL_SCENARIOS = [
    {
        "id": "code_exec",
        "desc": "贴代码问运行结果 → 应调用沙箱执行",
        "agent": "coding",
        "message": "帮我运行看看这段代码输出什么：\nprint(sum(range(101)))",
        "expected_tools": ["execute_code_sandbox"],
        "param_checks": {"execute_code_sandbox": ["code"]},
    },
    {
        "id": "code_concept",
        "desc": "纯概念讲解 → 不应调用任何工具",
        "agent": "coding",
        "message": "给我讲讲贪心算法和动态规划的区别，什么时候该用哪个？",
        "expected_tools": [],
        "param_checks": {},
    },
    {
        "id": "resume_eval",
        "desc": "贴简历求评估 → 应调用简历评分",
        "agent": "career",
        "message": (
            "帮我评估一下简历：本人熟练掌握 Python、PyTorch，做过推荐系统项目，"
            "在 XX 公司实习 6 个月，熟悉 A/B 测试与特征工程。"
        ),
        "expected_tools": ["evaluate_resume"],
        "param_checks": {"evaluate_resume": ["resume_text"]},
    },
    {
        "id": "gap_analysis",
        "desc": "岗位差距分析 → 应查询岗位要求",
        "agent": "career",
        "message": "我想做 Agent 开发工程师，帮我分析一下我现在还差什么技能？",
        "expected_tools": ["get_job_requirements"],
        "param_checks": {"get_job_requirements": ["position"]},
    },
    {
        "id": "interview_q",
        "desc": "求面试题 → 应调用题库工具",
        "agent": "career",
        "message": "给我出几道 AI 方向的面试题练练手",
        "expected_tools": ["get_interview_questions"],
        "param_checks": {"get_interview_questions": ["topic"]},
    },
    {
        "id": "career_generic",
        "desc": "一般求职咨询 → 不应调用任何工具",
        "agent": "career",
        "message": "秋招的时间线一般是怎么安排的？几月份开始投递比较合适？",
        "expected_tools": [],
        "param_checks": {},
    },
    {
        "id": "code_debug",
        "desc": "贴报错代码求调试 → 应调用沙箱复现",
        "agent": "coding",
        "message": "这段代码一直报 TypeError，帮我跑一下看看：\nx = [1,2,3]\nprint(x + 4)",
        "expected_tools": ["execute_code_sandbox"],
        "param_checks": {"execute_code_sandbox": ["code"]},
    },
    {
        "id": "code_concept2",
        "desc": "纯概念（泛化验证）→ 不应调用任何工具",
        "agent": "coding",
        "message": "什么是时间复杂度的大 O 表示法？为什么快排平均是 O(n log n)？",
        "expected_tools": [],
        "param_checks": {},
    },
    {
        "id": "career_interview_prep",
        "desc": "指定方向求面试题 → 应调用题库工具",
        "agent": "career",
        "message": "下周要面字节后端实习，帮我出几道操作系统方向的面试题练练",
        "expected_tools": ["get_interview_questions"],
        "param_checks": {"get_interview_questions": ["topic"]},
    },
    {
        "id": "career_requirements2",
        "desc": "问岗位技能要求 → 应查询岗位要求",
        "agent": "career",
        "message": "做机器学习方向的实习一般都需要会哪些技能？",
        "expected_tools": ["get_job_requirements"],
        "param_checks": {"get_job_requirements": ["position"]},
    },
    {
        "id": "career_resume2",
        "desc": "贴简历求评估（泛化验证）→ 应调用简历评分",
        "agent": "career",
        "message": "帮我看看这份简历怎么样：熟悉 Java 和 Spring，做过电商后端项目，实习过 3 个月",
        "expected_tools": ["evaluate_resume"],
        "param_checks": {"evaluate_resume": ["resume_text"]},
    },
    {
        "id": "career_generic2",
        "desc": "观点类咨询（泛化验证）→ 不应调用任何工具",
        "agent": "career",
        "message": "日常实习和秋招正式批，哪个途径更容易进大厂？",
        "expected_tools": [],
        "param_checks": {},
    },
]


# ============================================================
# Layer 2 记录型工具替身（与真实工具同签名，仅记录 + 返回构造结果）
# ============================================================

def _make_stubs():
    """创建 4 个替身工具与全局调用记录。"""

    calls = []

    def _record(name, args, result):
        calls.append({"tool": name, "args": args})
        return result

    from langchain_core.tools import tool

    @tool
    def execute_code_sandbox(code: str, language: str = "python") -> dict:
        """在安全沙箱中执行代码并返回运行结果"""
        return _record(
            "execute_code_sandbox",
            {"code": code, "language": language},
            {"success": True, "output": "5050", "source": "stub"},
        )

    @tool
    def evaluate_resume(resume_text: str) -> dict:
        """对简历进行 ATS 风格评分，返回分数、优点与改进建议"""
        return _record(
            "evaluate_resume",
            {"resume_text": resume_text},
            {"success": True, "score": 72, "strengths": ["Python"], "suggestions": ["补充量化成果"], "source": "stub"},
        )

    @tool
    def get_interview_questions(topic: str, count: int = 3) -> list:
        """获取指定主题的面试题（AI/算法/编程/系统设计/行为面试）"""
        return _record(
            "get_interview_questions",
            {"topic": topic, "count": count},
            ["问题一", "问题二", "问题三"],
        )

    @tool
    def get_job_requirements(position: str) -> dict:
        """查询目标岗位的技能要求（核心技能/加分项/考察重点）"""
        return _record(
            "get_job_requirements",
            {"position": position},
            {"success": True, "position": position, "core_skills": ["LangChain", "RAG"], "source": "stub"},
        )

    return {
        "execute_code_sandbox": execute_code_sandbox,
        "evaluate_resume": evaluate_resume,
        "get_interview_questions": get_interview_questions,
        "get_job_requirements": get_job_requirements,
    }, calls


# ============================================================
# 三层评测实现
# ============================================================

def eval_routing() -> list:
    """Layer 1：路由轨迹（消息 → 分类 → 条件路由 → 预期节点）。"""
    results = []
    for sc in ROUTING_SCENARIOS:
        category, confidence, _ = _classify_message(sc["message"], {})
        routed = _route_by_category(
            {"message_category": category, "interview_mode": False}
        )
        expected_node = (
            sc["expected"] if sc["expected"] != "general_chat" else "general_chat"
        )
        # emotional/chitchat 均路由到 general_chat 节点
        passed = routed == expected_node
        results.append({
            "layer": 1, "id": sc["message"][:20], "desc": sc["message"],
            "passed": passed,
            "detail": f"category={category}({confidence:.2f}) → {routed} (期望 {expected_node})",
        })
    return results


def eval_tool_selection(scenarios) -> list:
    """Layer 2：工具选择轨迹（真实 LLM 决策 + 替身记录）。"""
    from companion_ai.agents.career_agent import _get_llm as _career_llm
    from companion_ai.agents.coding_agent import _get_llm as _coding_llm

    results = []
    for sc in scenarios:
        stubs, calls = _make_stubs()
        state = {
            "user_id": "trajectory_eval",
            "current_message": sc["message"],
            "message_category": sc["agent"],
            "emotion_label": "neutral",
            "emotion_score": 0.5,
            "user_profile": {},
            "retrieved_memories": [],
        }
        try:
            if sc["agent"] == "coding":
                prompt = _build_coding_prompt(state)
                llm = _coding_llm()
                tools = [stubs["execute_code_sandbox"]]
            else:
                prompt = _build_career_prompt(state)
                llm = _career_llm()
                tools = [
                    stubs["evaluate_resume"],
                    stubs["get_interview_questions"],
                    stubs["get_job_requirements"],
                ]

            invoke_llm_with_tools(llm, prompt, tools)

            called = [c["tool"] for c in calls]
            # 工具选择断言（子集语义）：
            #   - 预期非空：预期工具必须被调用（允许 LLM 合理追加，如评估简历
            #     同时查询岗位要求以给针对性建议）
            #   - 预期为空：不应调用任何工具（过度调用浪费 token/延迟）
            if sc["expected_tools"]:
                select_ok = set(sc["expected_tools"]).issubset(set(called))
            else:
                select_ok = len(called) == 0

            # 参数断言：预期工具的必备参数均已传入
            param_fail = []
            for tool_name, required_keys in sc["param_checks"].items():
                tool_calls = [c for c in calls if c["tool"] == tool_name]
                if not tool_calls:
                    param_fail.append(f"{tool_name}:未调用")
                    continue
                for key in required_keys:
                    val = tool_calls[0]["args"].get(key)
                    if val is None or (isinstance(val, str) and not val.strip()):
                        param_fail.append(f"{tool_name}:缺参数{key}")

            passed = select_ok and not param_fail
            results.append({
                "layer": 2, "id": sc["id"], "desc": sc["desc"], "passed": passed,
                "detail": (
                    f"调用={called or '无'} (期望 {sc['expected_tools'] or '无'}) "
                    f"{'参数缺失:' + ';'.join(param_fail) if param_fail else '参数完整'}"
                ),
            })
        except Exception as e:
            results.append({
                "layer": 2, "id": sc["id"], "desc": sc["desc"], "passed": False,
                "detail": f"评测异常: {e}",
            })
    return results


def eval_tool_contract() -> list:
    """Layer 3：工具执行契约（真实工具本地路径 + 剧本参数）。"""
    checks = []

    def run(name, fn, detail_check):
        try:
            result = fn()
            ok, note = detail_check(result)
        except Exception as e:
            ok, note = False, f"异常: {e}"
        checks.append({"layer": 3, "id": name, "desc": f"{name} 输出契约", "passed": ok, "detail": note})

    run(
        "get_job_requirements",
        lambda: get_job_requirements.invoke({"position": "算法工程师"}),
        lambda r: (
            isinstance(r, dict) and bool(r.get("core_skills")),
            f"core_skills={len(r.get('core_skills', []))}项",
        ),
    )
    run(
        "evaluate_resume",
        lambda: evaluate_resume.invoke({
            "resume_text": "熟练掌握 Python PyTorch，做过推荐系统项目，实习6个月，熟悉A/B测试"
        }),
        lambda r: (
            isinstance(r, dict) and 0 <= float(r.get("score", -1)) <= 100,
            f"score={r.get('score')}",
        ),
    )
    run(
        "get_interview_questions",
        lambda: get_interview_questions.invoke({"topic": "AI", "count": 3}),
        lambda r: (
            isinstance(r, dict)
            and isinstance(r.get("questions"), list)
            and 1 <= len(r["questions"]) <= 3,
            f"返回{len(r.get('questions', []))}题",
        ),
    )
    run(
        "execute_code_sandbox",
        lambda: execute_code_sandbox.invoke({
            "code": "print(sum(range(101)))", "language": "python"
        }),
        lambda r: (
            isinstance(r, dict)
            and r.get("success")
            and "5050" in str(r.get("stdout", "")),
            f"stdout={str(r.get('stdout'))[:30]} source={r.get('source')}",
        ),
    )
    run(
        "execute_code_sandbox_error",
        lambda: execute_code_sandbox.invoke({
            "code": "print(1/0)", "language": "python"
        }),
        lambda r: (
            isinstance(r, dict)
            and r.get("success") is False
            and r.get("exit_code") != 0
            and "ZeroDivisionError" in str(r.get("stderr", "")),
            f"exit_code={r.get('exit_code')} stderr含错误信息",
        ),
    )
    run(
        "get_job_requirements_unknown",
        lambda: get_job_requirements.invoke({"position": "量子计算工程师"}),
        lambda r: (
            isinstance(r, dict)
            and r.get("matched") is False
            and isinstance(r.get("available_positions"), list)
            and len(r["available_positions"]) > 0,
            f"未知岗位返回可选列表: {len(r.get('available_positions', []))}个岗位",
        ),
    )
    return checks


# ============================================================
# 报告
# ============================================================

def write_report(results: list, llm_included: bool) -> str:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        f"# 工具调用轨迹评测报告",
        f"",
        f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模式: {'三层完整评测（含真实 LLM 工具选择）' if llm_included else '仅确定性层（--skip-llm）'}",
        f"- 总体: **{passed}/{total} 通过（{passed / total * 100:.0f}%）**",
        f"",
        f"| 层级 | 剧本 | 结果 | 详情 |",
        f"|---|---|---|---|",
    ]
    for r in results:
        mark = "✅" if r["passed"] else "❌"
        lines.append(f"| L{r['layer']} | {r['desc'][:30]} | {mark} | {r['detail']} |")

    # 分层汇总
    for layer in (1, 2, 3):
        layer_results = [r for r in results if r["layer"] == layer]
        if not layer_results:
            continue
        lp = sum(1 for r in layer_results if r["passed"])
        name = {1: "路由轨迹", 2: "工具选择轨迹", 3: "工具执行契约"}[layer]
        lines.append(f"\n**L{layer} {name}**: {lp}/{len(layer_results)} 通过")

    report_path = os.path.join(
        os.path.dirname(__file__), "reports", f"trajectory_eval_{ts}.md"
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="工具调用轨迹评测")
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="跳过 Layer 2（真实 LLM 工具选择），仅跑确定性层，零 API 成本",
    )
    args = parser.parse_args()

    logger.info("轨迹评测 | Layer 1 路由轨迹...")
    results = eval_routing()

    if not args.skip_llm:
        logger.info("轨迹评测 | Layer 2 工具选择轨迹（真实 LLM）...")
        results += eval_tool_selection(TOOL_SCENARIOS)
    else:
        logger.info("轨迹评测 | 已跳过 Layer 2（--skip-llm）")

    logger.info("轨迹评测 | Layer 3 工具执行契约...")
    results += eval_tool_contract()

    print("\n" + "=" * 60)
    print("工具调用轨迹评测")
    print("=" * 60)
    for r in results:
        mark = "✅" if r["passed"] else "❌"
        print(f"{mark} [L{r['layer']}] {r['desc'][:40]}")
        if not r["passed"]:
            print(f"    └─ {r['detail']}")

    passed = sum(1 for r in results if r["passed"])
    print("-" * 60)
    print(f"总体: {passed}/{len(results)} 通过（{passed / len(results) * 100:.0f}%）")

    report_path = write_report(results, llm_included=not args.skip_llm)
    print(f"\n报告已保存: {report_path}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
