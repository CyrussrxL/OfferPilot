"""
简历解析模块 —— 提取文本 + LLM 结构化摘要

职责：
  1. 从上传文件（PDF/MD/TXT）或粘贴文本中提取纯文本
  2. 调用 LLM 将简历结构化为摘要（项目经历 / 核心技能 / 亮点 / 目标岗位），
     存入 user_profile.resume_summary 供模拟面试定向出题与长期辅导复用

设计理由：
  - PDF 解析用 pypdf（纯 Python，无系统依赖），MD/TXT 直接读取
  - LLM 结构化只做一次（上传时），结果持久化到画像，后续面试/辅导直接复用，
    避免每轮面试重复调用 LLM 解析简历
  - 解析失败时返回原始文本截断作为摘要，保证链路不中断
"""

import os
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


def _get_llm() -> ChatOpenAI:
    """简历解析用 LLM（低温度保证结构稳定）。"""
    return ChatOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """从 PDF 二进制提取纯文本（pypdf）。"""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_file(content: bytes, filename: str) -> str:
    """按文件扩展名分发提取文本。"""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(content)
    if ext in (".md", ".txt"):
        return content.decode("utf-8", errors="ignore").strip()
    # 未知扩展名按文本尝试
    try:
        return content.decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def parse_resume(resume_text: str) -> Dict[str, Any]:
    """
    将简历文本结构化为摘要。

    调用 LLM 提取：目标岗位 / 核心技能 / 项目经历 / 亮点 / 潜在薄弱点。
    解析失败时退化为原始文本前 800 字的摘要，保证链路不中断。

    Returns:
        {
            "success": bool,
            "summary": str,          # 简历一句话摘要
            "target_role": str,      # 推断目标岗位
            "skills": List[str],     # 核心技能
            "projects": List[str],   # 项目经历要点
            "highlights": List[str], # 亮点
            "raw_excerpt": str,      # 原文截断（兜底与校验用）
        }
    """
    if not resume_text or not resume_text.strip():
        return {"success": False, "error": "简历文本为空", "raw_excerpt": "", "brief": ""}

    raw_excerpt = resume_text[:800]

    prompt = f"""你是一位资深的技术招聘官，正在解析候选人的简历。请从以下简历文本中提取结构化信息。

简历文本：
{resume_text[:3000]}

只输出 JSON，不要任何额外说明：
{{
  "summary": "一句话概括候选人画像（如：3年经验的Python后端，有推荐系统项目经验）",
  "target_role": "推断的目标岗位（如：算法工程师/后端开发/数据分析师）",
  "skills": ["核心技能1", "核心技能2", "..."],
  "projects": ["项目1名称与要点", "项目2名称与要点"],
  "highlights": ["亮点1", "亮点2"]
}}"""

    try:
        response = _get_llm().invoke(prompt)
        import json
        import re

        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if not match:
            raise ValueError("LLM 输出无 JSON")
        data = json.loads(match.group(0))
        result = {
            "success": True,
            "summary": data.get("summary", ""),
            "target_role": data.get("target_role", ""),
            "skills": data.get("skills", []),
            "projects": data.get("projects", []),
            "highlights": data.get("highlights", []),
            "raw_excerpt": raw_excerpt,
        }
        # 预构建给面试 prompt 用的摘要字符串，前端透传给 run_workflow
        result["brief"] = build_resume_brief(result)
        return result
    except Exception as e:
        logger.warning(f"简历解析 LLM 失败，退化为原文摘要: {e}")
        fallback = {
            "success": False,
            "error": str(e),
            "summary": "",
            "target_role": "",
            "skills": [],
            "projects": [],
            "highlights": [],
            "raw_excerpt": raw_excerpt,
        }
        fallback["brief"] = build_resume_brief(fallback)
        return fallback


def build_resume_brief(summary: Dict[str, Any]) -> str:
    """
    将结构化简历摘要拼成给面试 prompt 用的简短文本。

    供 interview_agent 出题/追问/评估 prompt 注入，控制在 500 字以内，
    避免占用过多 token。
    """
    if not summary or not summary.get("success"):
        return summary.get("raw_excerpt", "") if summary else ""

    parts = []
    if summary.get("summary"):
        parts.append(f"画像：{summary['summary']}")
    if summary.get("target_role"):
        parts.append(f"目标岗位：{summary['target_role']}")
    if summary.get("skills"):
        parts.append("核心技能：" + "、".join(summary["skills"]))
    if summary.get("projects"):
        parts.append("项目经历：\n" + "\n".join(f"- {p}" for p in summary["projects"]))
    if summary.get("highlights"):
        parts.append("亮点：" + "；".join(summary["highlights"]))
    brief = "\n".join(parts)
    return brief[:500]
