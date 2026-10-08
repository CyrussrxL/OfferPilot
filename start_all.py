"""
OfferPilot 启动脚本 —— start_all.py

功能：
  1. 启动 FastAPI 后端服务（后台运行）
  2. 启动 Streamlit 前端服务
  3. 提供统一的启动入口
"""

import os
import sys
import subprocess
import time
import signal


def start_backend():
    """启动 FastAPI 后端服务"""
    print("🚀 启动 FastAPI 后端服务 (端口 8000)...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "companion_ai.backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ]
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    time.sleep(3)
    return backend_process


def start_frontend():
    """启动 Streamlit 前端服务"""
    print("🎨 启动 Streamlit 前端服务 (端口 8501)...")
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "companion_ai/frontend/streamlit_app.py",
        "--server.port", "8501",
    ]
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return frontend_process


def main():
    """主函数：启动后端和前端"""
    print("=" * 60)
    print("🧭 OfferPilot - 多 Agent 智能求职领航系统")
    print("=" * 60)

    backend_process = None
    frontend_process = None

    try:
        backend_process = start_backend()
        frontend_process = start_frontend()

        print("\n✅ 服务启动成功！")
        print("📱 前端地址: http://localhost:8501")
        print("🔧 后端地址: http://localhost:8000")
        print("📚 API 文档: http://localhost:8000/docs")
        print("\n按 Ctrl+C 停止服务...\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 正在停止服务...")
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            backend_process.wait(timeout=5)
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            frontend_process.wait(timeout=5)
        print("✅ 服务已停止")


if __name__ == "__main__":
    main()
