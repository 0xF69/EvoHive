"""Tests for evohive.engine.sandbox"""

import pytest
from evohive.engine.sandbox import (
    ExecutionResult,
    execute_python,
    execute_with_test_cases,
    extract_code_blocks,
)


# ── execute_python ────────────────────────────────────────────────


def test_execute_simple_print():
    result = execute_python('print("hello")')
    assert result.success is True
    assert result.stdout == "hello\n"
    assert result.exit_code == 0
    assert result.timeout is False


def test_execute_with_error():
    result = execute_python("1/0")
    assert result.success is False
    assert "ZeroDivisionError" in result.stderr


def test_execute_timeout():
    result = execute_python("while True: pass", timeout=2.0)
    assert result.success is False
    assert result.timeout is True


def test_execute_memory_limit():
    """Verify the function doesn't crash when a memory bomb is run."""
    code = "x = ' ' * (512 * 1024 * 1024)"  # 512 MB string
    result = execute_python(code, max_memory_mb=64, timeout=5.0)
    # Either fails cleanly or succeeds on generous systems — should not hang
    assert isinstance(result, ExecutionResult)
    # On most Linux systems the memory limit will trigger a MemoryError
    # but we don't strictly assert failure since some CI environments
    # overcommit memory.


def test_execute_blocked_imports():
    result = execute_python("import subprocess")
    assert result.success is False
    assert "not allowed" in result.stderr.lower() or "ImportError" in result.stderr


def test_execute_blocked_imports_socket():
    result = execute_python("import socket")
    assert result.success is False
    assert "not allowed" in result.stderr.lower() or "ImportError" in result.stderr


def test_execute_stdin():
    code = "import sys; data = sys.stdin.read(); print(data.upper())"
    result = execute_python(code, stdin_data="hello world")
    assert result.success is True
    assert result.stdout.strip() == "HELLO WORLD"


def test_execute_duration_tracked():
    result = execute_python("import time; time.sleep(0.1); print('done')")
    assert result.success is True
    assert result.duration_ms >= 50  # at least some time passed


def test_execute_env_keys_stripped():
    """Sensitive env vars should not be visible in the subprocess."""
    import os

    os.environ["MY_SECRET_KEY"] = "supersecret"
    os.environ["API_TOKEN"] = "tok123"
    try:
        code = (
            "import os\n"
            "print(os.environ.get('MY_SECRET_KEY', 'MISSING'))\n"
            "print(os.environ.get('API_TOKEN', 'MISSING'))\n"
        )
        result = execute_python(code)
        assert result.success is True
        lines = result.stdout.strip().split("\n")
        assert lines[0] == "MISSING"
        assert lines[1] == "MISSING"
    finally:
        os.environ.pop("MY_SECRET_KEY", None)
        os.environ.pop("API_TOKEN", None)


# ── extract_code_blocks ──────────────────────────────────────────


def test_extract_code_blocks_markdown():
    text = (
        "Here is some code:\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
        "And more:\n"
        "```js\n"
        "console.log('hi')\n"
        "```\n"
    )
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]["language"] == "python"
    assert blocks[0]["code"] == "print('hello')"
    assert blocks[1]["language"] == "js"


def test_extract_code_blocks_bare():
    text = "def foo():\n    return 42\n\nprint(foo())"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["language"] == "python"
    assert "def foo():" in blocks[0]["code"]


def test_extract_code_blocks_no_code():
    blocks = extract_code_blocks("Just some plain English text about nothing.")
    assert blocks == []


# ── execute_with_test_cases ──────────────────────────────────────


def test_execute_with_test_cases_all_pass():
    code = "import sys; nums = sys.stdin.read().split(); print(int(nums[0]) + int(nums[1]))"
    test_cases = [
        {"input": "1 2", "expected": "3"},
        {"input": "10 20", "expected": "30"},
        {"input": "0 0", "expected": "0"},
    ]
    result = execute_with_test_cases(code, test_cases, timeout_per_case=5.0)
    assert result["total"] == 3
    assert result["passed"] == 3
    assert result["failed"] == 0
    assert result["score"] == 1.0


def test_execute_with_test_cases_partial():
    code = "import sys; n = int(sys.stdin.read()); print(n * 2)"
    test_cases = [
        {"input": "3", "expected": "6"},
        {"input": "5", "expected": "10"},
        {"input": "4", "expected": "999"},  # will fail
    ]
    result = execute_with_test_cases(code, test_cases, timeout_per_case=5.0)
    assert result["total"] == 3
    assert result["passed"] == 2
    assert result["failed"] == 1
    assert abs(result["score"] - 0.667) < 0.01


def test_execute_with_test_cases_empty():
    result = execute_with_test_cases("print('hi')", [])
    assert result["total"] == 0
    assert result["score"] == 1.0
    assert result["results"] == []


def test_execute_with_test_cases_error_in_code():
    code = "raise ValueError('boom')"
    test_cases = [{"input": "", "expected": "anything"}]
    result = execute_with_test_cases(code, test_cases)
    assert result["total"] == 1
    assert result["passed"] == 0
    assert result["results"][0]["passed"] is False
    assert result["results"][0]["error"] is not None
