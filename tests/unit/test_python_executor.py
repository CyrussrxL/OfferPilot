"""Python 代码沙箱执行器单元测试。"""

import subprocess

from companion_ai.tools.python_executor import execute_python_code


class TestExecutePythonCode:
    def test_normal_execution(self):
        result = execute_python_code.invoke({"code": "print('hello sandbox')"})
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "hello sandbox" in result["stdout"]

    def test_computation_result(self):
        result = execute_python_code.invoke({"code": "print(sum(range(101)))"})
        assert result["success"] is True
        assert "5050" in result["stdout"]

    def test_exception_captured(self):
        result = execute_python_code.invoke({"code": "raise ValueError('boom')"})
        assert result["success"] is False
        assert result["exit_code"] != 0
        assert "ValueError" in result["stderr"]

    def test_syntax_error_captured(self):
        result = execute_python_code.invoke({"code": "def broken(:"})
        assert result["success"] is False
        assert result["exit_code"] != 0

    def test_no_stdout_leak_from_stdin(self):
        # stdin=DEVNULL：依赖 input() 的代码应快速失败而非挂起
        result = execute_python_code.invoke({"code": "input()" })
        assert result["success"] is False

    def test_timeout_returns_error(self, monkeypatch):
        """30s 硬超时不适合单测，mock subprocess.run 抛 TimeoutExpired 验证分支。"""

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="python", timeout=30)

        monkeypatch.setattr(
            "companion_ai.tools.python_executor.subprocess.run", fake_run
        )
        result = execute_python_code.invoke({"code": "while True: pass"})
        assert result["success"] is False
        assert "超时" in result["error"]
        assert result["stdout"] == ""

    def test_generic_exception_branch(self, monkeypatch):
        def fake_run(*args, **kwargs):
            raise OSError("spawn failed")

        monkeypatch.setattr(
            "companion_ai.tools.python_executor.subprocess.run", fake_run
        )
        result = execute_python_code.invoke({"code": "print(1)"})
        assert result["success"] is False
        assert "spawn failed" in result["error"]
