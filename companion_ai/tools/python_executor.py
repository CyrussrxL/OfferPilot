"""
Python 代码执行工具模块

包含代码执行等编程相关工具。
"""

import os
import subprocess
import sys
import tempfile
from typing import Dict, Any
from langchain_core.tools import tool


@tool
def execute_python_code(code: str) -> Dict[str, Any]:
    """
    执行 Python 代码并返回结果。
    
    Args:
        code: 要执行的 Python 代码字符串
        
    Returns:
        包含执行结果或错误信息的字典
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file_path = f.name
        
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            stdin=subprocess.DEVNULL,  # 断开继承的 stdin（MCP stdio 管道环境下孙进程会挂起）
        )
        
        os.unlink(temp_file_path)
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "执行超时（30秒）",
            "stdout": "",
            "stderr": "代码执行超时，请检查是否有无限循环或耗时操作"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e)
        }
