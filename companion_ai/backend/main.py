"""
FastAPI 后端服务 —— main.py

职责：
  提供 RESTful API 接口，将 OfferPilot 的核心功能暴露为 Web 服务：
    - /api/chat: 处理用户消息，返回 AI 回复
    - /api/chat/stream: 流式返回 AI 回复
    - /api/conversations: 获取对话历史列表
    - /api/conversations/{id}: 获取/保存/删除特定对话
    - /api/profile: 获取/更新用户画像
    - /api/memories: 检索相关记忆
    - /api/report: 生成日报

设计理由：
  - 采用前后端分离架构，后端提供 API，前端负责展示
  - 使用 FastAPI 高性能异步框架，支持流式输出
  - 自动生成 OpenAPI 文档，方便前端开发和调试
  - CORS 配置支持跨域请求
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from companion_ai.graph.workflow import run_workflow, run_daily_report
from companion_ai.memory.vector_store import vector_store
from companion_ai.tools.resume_parser import extract_text_from_file, parse_resume
from companion_ai.utils.config import settings
from companion_ai.utils.logger import logger


app = FastAPI(
    title="OfferPilot API",
    description="多 Agent 智能求职领航系统 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    user_id: str = Field(default=settings.DEFAULT_USER_ID, description="用户 ID")
    message: str = Field(..., description="用户消息")
    thread_id: Optional[str] = Field(None, description="对话线程 ID")
    interview_mode: bool = Field(default=False, description="是否开启模拟面试模式")
    resume_summary: Optional[str] = Field(
        default=None, description="简历结构化摘要（定向面试用，无则按通识面试）"
    )
    jd_text: Optional[str] = Field(
        default=None, description="目标岗位 JD 全文（定向面试用，无则按通识面试）"
    )


class ChatResponse(BaseModel):
    success: bool
    final_response: str
    emotion_label: str
    emotion_score: float
    message_category: str


class UserProfile(BaseModel):
    learning_goal: Optional[str] = None
    current_skill_level: Optional[str] = None
    job_target: Optional[str] = None
    custom_fields: Optional[Dict[str, Any]] = None


class ConversationItem(BaseModel):
    id: str
    name: str
    timestamp: str
    message_count: int
    user_id: str


def get_conversations_dir():
    return os.path.join(os.path.dirname(__file__), "..", "..", "conversations")


def load_saved_conversations(user_id: Optional[str] = None) -> List[ConversationItem]:
    save_dir = get_conversations_dir()
    if not os.path.exists(save_dir):
        return []
    
    conversations = []
    for filename in os.listdir(save_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(save_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conv_user_id = data.get("user_id", "")
                    if user_id and conv_user_id != user_id:
                        continue
                    conversations.append(ConversationItem(
                        id=filename.replace(".json", ""),
                        name=data.get("name", filename),
                        timestamp=data.get("timestamp", ""),
                        message_count=len(data.get("chat_history", [])),
                        user_id=conv_user_id,
                    ))
            except Exception as e:
                logger.error(f"加载对话失败: {e}")
    
    conversations.sort(key=lambda x: x.timestamp, reverse=True)
    return conversations


@app.get("/")
async def root():
    return {
        "service": "OfferPilot",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/resume/upload")
async def upload_resume(user_id: str = settings.DEFAULT_USER_ID, file: UploadFile = File(default=None)):
    """上传简历文件（PDF/MD/TXT）→ 提取文本 → LLM 结构化摘要 → 存入用户画像。

    返回结构化摘要；失败时返回原始文本截断与错误信息，链路不中断。
    """
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="未提供简历文件")
        content = await file.read()
        text = extract_text_from_file(content, file.filename or "resume.txt")
        if not text:
            raise HTTPException(status_code=400, detail="无法从文件提取文本")

        summary = parse_resume(text)
        # 存入用户画像，供模拟面试与长期辅导复用
        profile = vector_store.get_user_profile(user_id)
        profile["resume_summary"] = summary
        vector_store.save_user_profile(user_id, profile)

        logger.info(
            f"API /api/resume/upload | user={user_id} | "
            f"解析成功={summary.get('success')} | file={file.filename}"
        )
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API /api/resume/upload | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resume/text")
async def parse_resume_text(payload: Dict[str, Any]):
    """粘贴简历文本 → LLM 结构化摘要 → 存入用户画像。"""
    try:
        user_id = payload.get("user_id", settings.DEFAULT_USER_ID)
        text = payload.get("text", "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="简历文本为空")

        summary = parse_resume(text)
        profile = vector_store.get_user_profile(user_id)
        profile["resume_summary"] = summary
        vector_store.save_user_profile(user_id, profile)

        logger.info(
            f"API /api/resume/text | user={user_id} | 解析成功={summary.get('success')}"
        )
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API /api/resume/text | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理用户消息，返回 AI 回复"""
    try:
        logger.info(f"API /api/chat | user={request.user_id} | 收到请求")

        result = run_workflow(
            user_id=request.user_id,
            message=request.message,
            thread_id=request.thread_id or f"thread_{request.user_id}",
            interview_mode=request.interview_mode,
            resume_summary=request.resume_summary,
            jd_text=request.jd_text,
        )
        
        response = ChatResponse(
            success=True,
            final_response=result.get("final_response", ""),
            emotion_label=result.get("emotion_label", "neutral"),
            emotion_score=result.get("emotion_score", 0.5),
            message_category=result.get("message_category", ""),
        )
        
        logger.info(f"API /api/chat | user={request.user_id} | 响应成功")
        return response
    except Exception as e:
        logger.error(f"API /api/chat | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式返回 AI 回复"""
    try:
        logger.info(f"API /api/chat/stream | user={request.user_id} | 收到请求")
        
        result = run_workflow(
            user_id=request.user_id,
            message=request.message,
            thread_id=request.thread_id or f"thread_{request.user_id}",
            interview_mode=request.interview_mode,
            resume_summary=request.resume_summary,
            jd_text=request.jd_text,
        )

        final_response = result.get("final_response", "")
        emotion_label = result.get("emotion_label", "neutral")
        emotion_score = result.get("emotion_score", 0.5)
        message_category = result.get("message_category", "")
        
        async def generate():
            for char in final_response:
                yield char
                import asyncio
                await asyncio.sleep(0.01)
            
            yield f"\n\n__META__:{json.dumps({'emotion': emotion_label, 'category': message_category}, ensure_ascii=False)}"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )
    except Exception as e:
        logger.error(f"API /api/chat/stream | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations")
async def get_conversations(user_id: Optional[str] = None):
    """获取对话列表"""
    try:
        conversations = load_saved_conversations(user_id)
        return {
            "success": True,
            "count": len(conversations),
            "conversations": conversations,
        }
    except Exception as e:
        logger.error(f"API /api/conversations | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取特定对话"""
    save_dir = get_conversations_dir()
    filepath = os.path.join(save_dir, f"{conversation_id}.json")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="对话不存在")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"API /api/conversations/{conversation_id} | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conversations/{conversation_id}")
async def save_conversation(conversation_id: str, request: Request):
    """保存对话"""
    save_dir = get_conversations_dir()
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{conversation_id}.json")
    
    try:
        data = await request.json()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return {"success": True, "id": conversation_id}
    except Exception as e:
        logger.error(f"API /api/conversations/{conversation_id} | 保存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除对话"""
    save_dir = get_conversations_dir()
    filepath = os.path.join(save_dir, f"{conversation_id}.json")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="对话不存在")
    
    try:
        os.remove(filepath)
        return {"success": True}
    except Exception as e:
        logger.error(f"API /api/conversations/{conversation_id} | 删除失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    """获取用户画像"""
    try:
        profile = vector_store.get_user_profile(user_id)
        return {"success": True, "user_id": user_id, "profile": profile}
    except Exception as e:
        logger.error(f"API /api/profile/{user_id} | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/profile/{user_id}")
async def save_profile(user_id: str, profile: UserProfile):
    """保存用户画像"""
    try:
        existing = vector_store.get_user_profile(user_id)
        
        if profile.learning_goal:
            existing["learning_goal"] = profile.learning_goal
        if profile.current_skill_level:
            existing["current_skill_level"] = profile.current_skill_level
        if profile.job_target:
            existing["job_target"] = profile.job_target
        if profile.custom_fields:
            existing.update(profile.custom_fields)
        
        vector_store.save_user_profile(user_id, existing)
        return {"success": True, "user_id": user_id, "profile": existing}
    except Exception as e:
        logger.error(f"API /api/profile/{user_id} | 保存失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories/{user_id}")
async def retrieve_memories(user_id: str, query: str, top_k: int = 3):
    """检索相关记忆"""
    try:
        memories = vector_store.retrieve_memories(user_id, query, top_k=top_k)
        return {"success": True, "user_id": user_id, "memories": memories}
    except Exception as e:
        logger.error(f"API /api/memories/{user_id} | 检索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/{user_id}")
async def generate_report(user_id: str):
    """生成日报"""
    try:
        report = run_daily_report(user_id)
        return {"success": True, "user_id": user_id, "report": report}
    except Exception as e:
        logger.error(f"API /api/report/{user_id} | 生成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ClassificationCorrectionRequest(BaseModel):
    user_id: str = Field(default=settings.DEFAULT_USER_ID, description="用户 ID")
    message: str = Field(..., description="被错分的消息原文")
    correct_category: str = Field(
        ..., description="正确的类别（coding/career/emotional/chitchat）"
    )


class FactCorrectionRequest(BaseModel):
    fact_text: str = Field(..., description="纠正后的事实文本")


@app.get("/api/memory/facts/{user_id}")
async def list_facts(user_id: str):
    """列出用户全部事实记忆（含记录 id，供删除/纠正定位）"""
    try:
        facts = vector_store.get_facts_with_metadata(user_id)
        return {"success": True, "user_id": user_id, "count": len(facts), "facts": facts}
    except Exception as e:
        logger.error(f"API /api/memory/facts/{user_id} | 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/memory/facts/{user_id}/{fact_id}")
async def delete_fact(user_id: str, fact_id: str):
    """删除一条事实记忆（用户可控记忆）"""
    try:
        result = vector_store.delete_fact(fact_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "删除失败"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API /api/memory/facts/{user_id}/{fact_id} DELETE | 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/memory/facts/{user_id}/{fact_id}")
async def update_fact(user_id: str, fact_id: str, request: FactCorrectionRequest):
    """纠正一条事实记忆（用户可控记忆）"""
    try:
        result = vector_store.update_fact(fact_id, request.fact_text)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "纠正失败"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API /api/memory/facts/{user_id}/{fact_id} PUT | 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classification/correction")
async def correct_classification(request: ClassificationCorrectionRequest):
    """
    分类错分纠正（GuardAgent 反馈闭环入口）。

    用户发现路由错分后提交 (消息, 正确类别)，系统将其回流为新的
    分类种子，后续同类消息的向量分类即可命中，实现自我改进。
    """
    try:
        from companion_ai.agents.guard_agent import record_classification_correction

        result = record_classification_correction(
            user_id=request.user_id,
            message=request.message,
            correct_category=request.correct_category,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "纠正失败"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API /api/classification/correction | 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_backend(host: str = "0.0.0.0", port: int = 8000):
    """启动 FastAPI 后端服务"""
    import uvicorn
    logger.info(f"启动 FastAPI 后端服务 | host={host}, port={port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_backend()
