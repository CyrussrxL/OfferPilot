"""
InterviewAgent —— 循环面试对话状态机

职责：
  循环面试状态机的三个节点，将一次性 interview_mode 升级为多轮陪练：
    - interview_evaluate: 决策节点。开启面试 / 评估用户最新回答 /
      判定 next（下一题）/ probe（动态追问）/ report（生成报告）
    - interview_ask: 出题节点。新题经 MCP Tool Server 的题库工具获取，
      并结合用户画像定制；probe 时基于回答薄弱点生成针对性追问
    - interview_report: 报告节点。汇总全部问答生成结构化面试报告
      （维度评分 / 优势 / 薄弱点 / 学习建议），薄弱点回流事实记忆

图结构（带条件回边）：
  memory →(interview_mode)→ interview_evaluate
  interview_evaluate →(ask)→  interview_ask  → response_composer
  interview_evaluate →(report)→ interview_report → response_composer

设计理由：
  - 面试会话状态（interview_session）经 LangGraph 检查点跨轮持久化，
    同一 thread_id 的多次 invoke 构成一场连续面试。
  - 动态追问（probe）是本状态机的核心亮点：不问固定题，而是基于
    用户回答的薄弱点实时追问，模拟真实面试官的深挖行为。
  - evaluate → ask 的条件回边 + 跨轮 awaiting_answer 标志共同构成
    "提问 → 回答 → 评估 → 追问/下一题/结束"的循环对话图。
  - 题目经自建 MCP Tool Server（get_interview_questions）获取，
    面试题库与 MCP 协议链路复用路线 3 的成果。
  - 面试报告的薄弱点写入事实记忆，后续对话的检索和画像反思
    都能感知"用户在面试中暴露的短板"，形成辅导闭环。
"""

import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from companion_ai.graph.state import State
from companion_ai.memory.vector_store import vector_store
from companion_ai.tools.career_tools import get_interview_questions
from companion_ai.tools.resume_parser import build_resume_brief
from companion_ai.utils.config import settings
from companion_ai.utils.helpers import format_user_profile, get_timestamp, safe_json_loads
from companion_ai.utils.logger import logger

# 用户主动结束面试的指令模式
_END_INTERVIEW_PATTERN = re.compile(
    r"(结束面试|停止面试|退出面试|不要面试|不面了|结束吧|到此为止)"
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0.7,
        timeout=120,
        max_retries=3,
    )


def _get_eval_llm() -> ChatOpenAI:
    """评估用 LLM（低温度保证评分稳定）。"""
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def _parse_json(content: str) -> Dict[str, Any]:
    """宽松解析 LLM 输出的 JSON（容忍 markdown 代码块包裹）。"""
    import json

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _infer_interview_topic(
    profile: Dict[str, Any], message: str, jd_text: str = ""
) -> str:
    """从用户画像、首条消息和 JD 推断面试主题（映射到题库主题）。

    JD 优先级最高：有 JD 时主要从 JD 关键词推断主题（岗位描述天然带方向）。
    """
    text = f"{profile.get('job_target', '')} {profile.get('learning_goal', '')} {message} {jd_text}"
    topic_map = {
        "AI": ["ai", "算法工程师", "机器学习", "深度学习", "nlp", "大模型", "llm"],
        "算法": ["算法", "数据结构", "刷题", "leetcode"],
        "编程": ["后端", "开发", "python", "java", "软件工程"],
        "系统设计": ["系统设计", "架构", "分布式", "高并发"],
        "行为面试": ["行为", "hr", "软技能"],
    }
    text_lower = text.lower()
    for topic, keywords in topic_map.items():
        if any(kw in text_lower for kw in keywords):
            return topic
    return "AI"


def _new_session(
    profile: Dict[str, Any],
    message: str,
    resume_summary: str = "",
    jd_text: str = "",
) -> Dict[str, Any]:
    """创建一场新的面试会话状态。

    resume_summary / jd_text 为定向面试上下文：有则出题/追问/评估全程注入，
    无则按通识面试（向后兼容）。
    """
    return {
        "active": True,
        "topic": _infer_interview_topic(profile, message, jd_text),
        "questions": [],       # [{"question", "dimension", "answer", "evaluation"}]
        "asked_count": 0,
        "max_questions": settings.INTERVIEW_MAX_QUESTIONS,
        "probe_count": 0,      # 当前题已追问次数（每题最多 1 次）
        "awaiting_answer": False,
        "started_at": get_timestamp(),
        "resume_summary": resume_summary,  # 简历结构化摘要（定向出题用）
        "jd_text": jd_text,              # 目标岗位 JD 全文（定向考察用）
    }


def interview_evaluate(state: State) -> Dict:
    """
    面试状态机决策节点。

    每轮面试消息都经过此节点，决策逻辑：
      1. 无活跃会话 → 创建新会话，转 ask 出第一题
      2. 有会话但非等待回答状态 → 转 ask 出题
      3. 等待回答 → 评估用户最新回答：
         - 用户主动结束 → 转 report
         - 已问满 max_questions → 转 report
         - 得分低于阈值且未追问过 → probe（回边到 ask 追问）
         - 否则 → next（ask 出下一题）

    Returns:
        更新 interview_session（含本轮 decision 字段，供条件边路由）
    """
    user_id = state.get("user_id", "default_user")
    message = state.get("current_message", "")
    profile = state.get("user_profile", {})
    session = state.get("interview_session") or {}
    # 定向面试上下文（前端传入；无则空串，按通识面试）
    resume_summary = state.get("resume_summary", "") or ""
    jd_text = state.get("jd_text", "") or ""

    # 1. 无活跃会话 → 开启新面试（注入简历/JD 上下文）
    if not session.get("active"):
        session = _new_session(profile, message, resume_summary, jd_text)
        session["decision"] = "ask"
        directed = "定向" if (resume_summary or jd_text) else "通识"
        logger.info(
            f"InterviewEvaluate | user={user_id} | 开启新面试 topic={session['topic']} | {directed}"
        )
        return {"interview_session": session}

    # 2. 非等待回答（理论上仅首轮），直接出题
    if not session.get("awaiting_answer"):
        session["decision"] = "ask"
        return {"interview_session": session}

    # 3. 用户主动结束面试
    if _END_INTERVIEW_PATTERN.search(message):
        session["decision"] = "report"
        session["end_reason"] = "user_request"
        logger.info(f"InterviewEvaluate | user={user_id} | 用户主动结束面试")
        return {"interview_session": session}

    # 4. 评估用户对当前题目的回答
    current_q = session["questions"][-1] if session["questions"] else {}
    jd_hint = (
        f"\n目标岗位 JD 摘要（评估时对照岗位要求，含岗位匹配度维度）：\n{session.get('jd_text', '')[:600]}"
        if session.get("jd_text")
        else ""
    )
    prompt = f"""你是一位资深的技术面试官，正在评估候选人对面试题的回答。

面试主题：{session.get("topic", "AI")}
面试题目：{current_q.get("question", "")}
考察维度：{current_q.get("dimension", "综合能力")}{jd_hint}

候选人回答：
「{message}」

请从面试官视角评估该回答，只输出 JSON：
{{
  "score": 0到100的整数,
  "strengths": ["回答中的亮点", ...],
  "weaknesses": ["回答暴露的薄弱点", ...],
  "dimension_feedback": "一句话总体点评"
}}"""

    try:
        response = _get_eval_llm().invoke(prompt)
        evaluation = _parse_json(response.content.strip())
        score = int(evaluation.get("score", 50))
    except Exception as e:
        logger.warning(f"InterviewEvaluate | 评估失败，按中等处理: {e}")
        evaluation = {
            "score": 50,
            "strengths": [],
            "weaknesses": [],
            "dimension_feedback": "评估服务暂时不可用",
        }
        score = 50

    # 回填本轮回答与评估到会话状态
    current_q["answer"] = message
    current_q["evaluation"] = evaluation
    logger.info(
        f"InterviewEvaluate | user={user_id} | "
        f"第{session['asked_count']}题得分={score}"
    )

    # 5. 决策：问满 → report；低分且未追问 → probe；否则 → next
    if session["asked_count"] >= session["max_questions"]:
        session["decision"] = "report"
        session["end_reason"] = "questions_completed"
    elif (
        score < settings.INTERVIEW_PROBE_SCORE_THRESHOLD
        and session.get("probe_count", 0) < 1
        and evaluation.get("weaknesses")
    ):
        session["decision"] = "probe"
        session["probe_count"] = session.get("probe_count", 0) + 1
    else:
        session["decision"] = "next"

    return {"interview_session": session}


def interview_ask(state: State) -> Dict:
    """
    面试出题/追问节点。

    根据 decision 分支：
      - probe: 基于评估薄弱点生成针对性追问（动态深挖）
      - ask/next: 经 MCP 题库工具获取新题 + LLM 结合画像定制

    Returns:
        interview_response（面试官话语）+ 更新 interview_session
    """
    user_id = state.get("user_id", "default_user")
    message = state.get("current_message", "")
    profile = state.get("user_profile", {})
    session = state.get("interview_session") or {}
    decision = session.get("decision", "ask")
    profile_text = format_user_profile(profile)

    llm = _get_llm()

    if decision == "probe":
        # 动态追问：基于薄弱点深挖
        current_q = session["questions"][-1] if session["questions"] else {}
        evaluation = current_q.get("evaluation", {})
        # resume_summary 已是构建好的摘要字符串，直接注入
        resume_hint = session.get("resume_summary", "")

        prompt = f"""你是一位资深的技术面试官。候选人刚刚的回答暴露了一些薄弱点，
请基于薄弱点进行一次针对性追问，深挖候选人的理解深度。

面试题目：{current_q.get("question", "")}
候选人回答：{current_q.get("answer", "")}
薄弱点：{", ".join(evaluation.get("weaknesses", []))}
{f"候选人简历要点（可针对其中项目细节追问）：{chr(10)}{resume_hint}" if resume_hint else ""}

要求：
- 追问要具体、直接指向薄弱点，不要泛泛而谈
- 语气保持面试官的专业与礼貌
- 只输出追问内容本身（1-3 句话），不要评分不要点评"""

        response = llm.invoke(prompt)
        interview_response = (
            f"🔎 **追问**：{response.content.strip()}\n\n"
            f"_(请针对上面的追问继续作答)_"
        )
        session["awaiting_answer"] = True
        logger.info(
            f"InterviewAsk | user={user_id} | 动态追问 "
            f"(第{session['asked_count']}题, probe_count={session['probe_count']})"
        )
        return {"interview_response": interview_response, "interview_session": session}

    # 出新题（ask=第一题 / next=下一题）
    topic = session.get("topic", "AI")

    # 题目经 MCP Tool Server 题库工具获取（内部 MCP 优先 + 本地降级）
    try:
        bank_result = get_interview_questions.invoke(
            {"topic": topic, "count": session.get("max_questions", 3)}
        )
        bank_questions = bank_result.get("questions", [])
        source = bank_result.get("source", "unknown")
    except Exception as e:
        logger.warning(f"InterviewAsk | 题库获取失败: {e}")
        bank_questions, source = [], "unavailable"

    # 已问过的题不再重复
    asked = [q.get("question", "") for q in session.get("questions", [])]
    fresh = [q for q in bank_questions if q not in asked]

    q_num = session.get("asked_count", 0) + 1
    # 定向面试上下文（有简历/JD 时优先围绕简历项目深挖 + JD 技能考察）
    resume_hint = session.get("resume_summary", "")
    jd_hint = session.get("jd_text", "")
    directed_block = ""
    if resume_hint or jd_hint:
        directed_block = "\n【定向面试模式】\n"
        if resume_hint:
            directed_block += (
                f"候选人简历要点（出题优先围绕其中的项目经历深挖）：\n{resume_hint}\n\n"
            )
        if jd_hint:
            directed_block += (
                f"目标岗位 JD（出题优先考察 JD 中的核心技能与职责）：\n{jd_hint[:600]}\n\n"
            )
        directed_block += (
            "出题策略：优先从简历项目经历和 JD 技能要求中找考察点，"
            "让题目贴合候选人真实背景与目标岗位；题库仅作兜底参考。\n"
        )

    prompt = f"""你是一位资深的技术面试官，正在对候选人进行「{topic}」方向的模拟面试，
现在是第 {q_num} / {session.get("max_questions", 3)} 题。

候选人画像：
{profile_text}{directed_block}
题库候选题目（兜底参考，定向模式下优先自创贴合题）：
{chr(10).join(f'- {q}' for q in fresh) if fresh else '（题库不可用，请自行出一道贴合画像的题）'}

之前的面试题（不要重复）：
{chr(10).join(f'- {q}' for q in asked) if asked else '（无）'}

要求：
- 出一道题，定向模式优先结合简历项目经历和 JD 技能要求，通识模式参考题库改编
- 指定一个考察维度（如：基础知识 / 项目深度 / 表达能力 / 岗位匹配 / 系统设计）
- 开头简短承接上一轮（如"好的，下一题"），面试官口吻

只输出 JSON：
{{
  "question": "题目内容",
  "dimension": "考察维度"
}}"""

    try:
        response = llm.invoke(prompt)
        parsed = _parse_json(response.content.strip())
        question = parsed.get("question", "").strip()
        dimension = parsed.get("dimension", "综合能力").strip()
    except Exception as e:
        logger.warning(f"InterviewAsk | 出题失败: {e}")
        question = ""
        dimension = "综合能力"

    # LLM 出题失败时降级用题库原题
    if not question:
        question = fresh[0] if fresh else f"请介绍一个你最有把握的{topic}相关项目"
        dimension = "项目深度"

    session["questions"].append(
        {"question": question, "dimension": dimension, "answer": "", "evaluation": {}}
    )
    session["asked_count"] = q_num
    session["probe_count"] = 0
    session["awaiting_answer"] = True

    interview_response = (
        f"🎯 **第 {q_num} / {session['max_questions']} 题**（{dimension}）\n\n"
        f"{question}\n\n"
        f"_(回答后我会评估你的作答；随时说「结束面试」可生成面试报告)_"
    )
    logger.info(
        f"InterviewAsk | user={user_id} | 出题 {q_num}/{session['max_questions']} "
        f"(dimension={dimension}, 题库source={source})"
    )
    return {"interview_response": interview_response, "interview_session": session}


def interview_report(state: State) -> Dict:
    """
    面试报告生成节点。

    汇总整场面试的问答与评估，生成结构化报告：
      - 总分与各题得分
      - 优势 / 薄弱点清单
      - 针对性学习建议
    并将薄弱点写入事实记忆（回流长期记忆，供后续辅导检索）。

    Returns:
        interview_response（报告文本）+ interview_report + 更新画像
    """
    user_id = state.get("user_id", "default_user")
    profile = state.get("user_profile", {})
    session = state.get("interview_session") or {}
    questions = [q for q in session.get("questions", []) if q.get("answer")]

    # 没有任何有效回答时直接结束
    if not questions:
        session["active"] = False
        session["awaiting_answer"] = False
        return {
            "interview_response": "本场面试没有有效作答，已结束。想再练一场随时告诉我！",
            "interview_session": session,
            "interview_report": {"success": False, "reason": "no_answers"},
        }

    qa_text = ""
    for i, q in enumerate(questions, 1):
        ev = q.get("evaluation", {})
        qa_text += (
            f"第{i}题（{q.get('dimension', '')}）：{q.get('question', '')}\n"
            f"回答：{q.get('answer', '')}\n"
            f"得分：{ev.get('score', 0)} | 亮点：{ev.get('strengths', [])} | "
            f"薄弱点：{ev.get('weaknesses', [])}\n\n"
        )

    avg_score = sum(
        q.get("evaluation", {}).get("score", 0) for q in questions
    ) / len(questions)

    # 定向面试：注入简历与 JD，报告增加岗位匹配度分析
    resume_hint = session.get("resume_summary", "")
    jd_hint = session.get("jd_text", "")
    directed_context = ""
    directed_report_section = ""
    if resume_hint or jd_hint:
        directed_context = "\n【定向面试上下文】\n"
        if resume_hint:
            directed_context += f"候选人简历要点：\n{resume_hint}\n\n"
        if jd_hint:
            directed_context += f"目标岗位 JD：\n{jd_hint[:600]}\n\n"
        directed_report_section = (
            "\n6. 岗位匹配度分析（结合 JD 要求与简历项目，指出匹配与差距）"
        )

    prompt = f"""你是一位资深的技术面试官，刚结束一场「{session.get("topic", "AI")}」方向的模拟面试。
请基于以下完整面试记录，生成一份结构化面试报告。

候选人画像：
{format_user_profile(profile)}{directed_context}
面试记录：
{qa_text}

报告要求（Markdown 格式）：
1. 总体评价（2-3 句，含本场平均分 {avg_score:.0f}/100）
2. 各题得分一览（表格：题号/维度/得分/一句话点评）
3. 核心优势（列表）
4. 薄弱点清单（列表，具体到知识点）
5. 针对性学习建议（结合薄弱点，给出可执行的行动项）{directed_report_section}

语气：专业、建设性，像真实面试后的复盘反馈。"""

    try:
        llm = _get_llm()
        report_text = llm.invoke(prompt).content.strip()
    except Exception as e:
        logger.error(f"InterviewReport | 报告生成失败: {e}")
        report_text = f"面试报告生成失败，本场平均分 {avg_score:.0f}/100。"

    # 薄弱点回流事实记忆（后续检索与画像反思可感知面试短板）
    all_weaknesses = []
    for q in questions:
        all_weaknesses.extend(q.get("evaluation", {}).get("weaknesses", []))
    if all_weaknesses:
        weakness_fact = (
            f"用户完成一次{session.get('topic', 'AI')}方向模拟面试，"
            f"平均得分{avg_score:.0f}，薄弱点：{'；'.join(all_weaknesses[:5])}"
        )
        vector_store.store_fact(
            user_id=user_id, fact=weakness_fact, timestamp=get_timestamp()
        )

    # 面试报告摘要写入画像的求职进度
    job_progress = profile.get("job_progress", {})
    reports = job_progress.get("interview_reports", [])
    reports.append(
        {
            "date": get_timestamp(),
            "topic": session.get("topic", ""),
            "avg_score": round(avg_score, 1),
            "question_count": len(questions),
            "weaknesses": all_weaknesses[:5],
        }
    )
    job_progress["interview_reports"] = reports[-5:]
    profile["job_progress"] = job_progress
    vector_store.save_user_profile(user_id, profile)

    # 关闭会话
    session["active"] = False
    session["awaiting_answer"] = False

    report_data = {
        "success": True,
        "topic": session.get("topic", ""),
        "avg_score": round(avg_score, 1),
        "question_count": len(questions),
        "weaknesses": all_weaknesses[:5],
        "end_reason": session.get("end_reason", "questions_completed"),
    }

    interview_response = (
        f"📋 **模拟面试报告**\n\n{report_text}\n\n"
        f"_(面试已结束，薄弱点已记入长期记忆，后续辅导会针对性跟进)_"
    )
    logger.info(
        f"InterviewReport | user={user_id} | 报告生成完成 "
        f"(avg={avg_score:.0f}, questions={len(questions)}, "
        f"weaknesses={len(all_weaknesses)})"
    )
    return {
        "interview_response": interview_response,
        "interview_session": session,
        "interview_report": report_data,
        "interview_score": avg_score,
    }
