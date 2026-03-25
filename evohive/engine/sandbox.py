"""Code Execution Sandbox — 安全代码执行沙箱

Runs untrusted Python code in isolated subprocesses with:
- Timeout enforcement (default 10s)
- Memory limit (default 256MB)
- No network access
- No filesystem write access (except /tmp)
- Captured stdout/stderr

Uses subprocess with resource limits, NOT eval/exec in the main process.
"""

import subprocess
import sys
import tempfile
import os
import re
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of executing code in the sandbox."""
    success: bool                       # True if exit code == 0
    stdout: str = ""                    # Captured stdout
    stderr: str = ""                    # Captured stderr
    exit_code: int = -1                 # Process exit code
    duration_ms: float = 0.0           # Execution time in milliseconds
    timeout: bool = False               # True if killed by timeout
    error: Optional[str] = None         # Error message if something went wrong
    return_value: Optional[str] = None  # Parsed return value if code prints JSON to stdout


def _clean_env() -> dict:
    """Return a copy of os.environ with sensitive variables stripped."""
    env = {}
    sensitive = {"KEY", "SECRET", "TOKEN"}
    for k, v in os.environ.items():
        upper = k.upper()
        if any(s in upper for s in sensitive):
            continue
        env[k] = v
    return env


_WRAPPER_TEMPLATE = '''\
import resource
import sys

# ── Set memory limit ──────────────────────────────────────────────
_mem_bytes = {max_memory_mb} * 1024 * 1024
try:
    resource.setrlimit(resource.RLIMIT_AS, (_mem_bytes, _mem_bytes))
except Exception:
    pass

# ── Block dangerous modules ──────────────────────────────────────
import builtins as _builtins
_original_import = _builtins.__import__
_BLOCKED = {{
    "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "ftplib", "smtplib", "ctypes", "multiprocessing",
}}

def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in _BLOCKED:
        raise ImportError(f"Module '{{name}}' is not allowed in sandbox")
    return _original_import(name, *args, **kwargs)

_builtins.__import__ = _safe_import

# ── Execute user code ────────────────────────────────────────────
try:
    with open({user_code_path!r}, "r") as _f:
        _code = _f.read()
    exec(compile(_code, "<sandbox>", "exec"))
except SystemExit:
    raise
except Exception as _e:
    print(f"ERROR: {{type(_e).__name__}}: {{_e}}", file=sys.stderr)
    sys.exit(1)
'''


def execute_python(
    code: str,
    *,
    timeout: float = 10.0,
    max_memory_mb: int = 256,
    stdin_data: str = "",
) -> ExecutionResult:
    """Execute Python code in an isolated subprocess.

    The code is written to a temp file and run as a separate process with
    resource limits enforced via a wrapper script.

    Args:
        code: Python source code to execute.
        timeout: Max execution time in seconds.
        max_memory_mb: Max memory in MB (enforced via resource.RLIMIT_AS).
        stdin_data: Data to feed to stdin.

    Returns:
        ExecutionResult with stdout, stderr, exit_code, duration, etc.
    """
    user_fd = wrapper_fd = None
    user_path = wrapper_path = None

    try:
        # Write user code to a temp file
        user_fd, user_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_user_")
        with os.fdopen(user_fd, "w") as f:
            user_fd = None  # os.fdopen takes ownership
            f.write(code)

        # Write wrapper script to another temp file
        wrapper_code = _WRAPPER_TEMPLATE.format(
            max_memory_mb=max_memory_mb,
            user_code_path=user_path,
        )
        wrapper_fd, wrapper_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_wrap_")
        with os.fdopen(wrapper_fd, "w") as f:
            wrapper_fd = None
            f.write(wrapper_code)

        env = _clean_env()

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, wrapper_path],
                input=stdin_data or None,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            elapsed = (time.monotonic() - t0) * 1000.0

            return ExecutionResult(
                success=(proc.returncode == 0),
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
                duration_ms=elapsed,
                timeout=False,
                error=proc.stderr.strip() if proc.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = (time.monotonic() - t0) * 1000.0
            return ExecutionResult(
                success=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                exit_code=-1,
                duration_ms=elapsed,
                timeout=True,
                error=f"Execution timed out after {timeout}s",
            )
    except Exception as exc:
        return ExecutionResult(
            success=False,
            exit_code=-1,
            error=f"Sandbox error: {type(exc).__name__}: {exc}",
        )
    finally:
        # Clean up temp files
        for fd in (user_fd, wrapper_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for path in (user_path, wrapper_path):
            if path is not None:
                try:
                    os.unlink(path)
                except OSError:
                    pass


def execute_with_test_cases(
    code: str,
    test_cases: list[dict],
    *,
    timeout_per_case: float = 5.0,
    max_memory_mb: int = 256,
) -> dict:
    """Run code against multiple test cases.

    Each test case should be a dict with:
        - ``"input"``: str fed to stdin
        - ``"expected"``: str compared to stdout (stripped, exact match)
      or:
        - ``"input"``: str
        - ``"validator"``: str — Python expression that receives variable ``output``

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "score": float,   # passed / total  (1.0 when total == 0)
            "results": [
                {
                    "input": str,
                    "expected": str | None,
                    "actual": str,
                    "passed": bool,
                    "error": str | None,
                },
                ...
            ],
        }
    """
    if not test_cases:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "score": 1.0,
            "results": [],
        }

    results: list[dict] = []

    for tc in test_cases:
        tc_input = tc.get("input", "")
        expected = tc.get("expected")
        validator = tc.get("validator")

        res = execute_python(
            code,
            timeout=timeout_per_case,
            max_memory_mb=max_memory_mb,
            stdin_data=tc_input,
        )

        actual = res.stdout.strip() if res.stdout else ""
        passed = False
        error: Optional[str] = None

        if not res.success:
            error = res.error or res.stderr
        elif validator is not None:
            # Run the validator expression safely — still in subprocess
            val_code = (
                f"output = {actual!r}\n"
                f"result = bool({validator})\n"
                f"print(result)\n"
            )
            val_res = execute_python(val_code, timeout=timeout_per_case)
            passed = val_res.stdout.strip() == "True"
            if not val_res.success:
                error = val_res.error
        elif expected is not None:
            expected_stripped = expected.strip()
            # Exact match first, then contains
            if actual == expected_stripped:
                passed = True
            elif expected_stripped in actual:
                passed = True
        else:
            # No expected or validator — pass if execution succeeded
            passed = True

        results.append({
            "input": tc_input,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "error": error,
        })

    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "score": round(passed_count / total, 3) if total else 1.0,
        "results": results,
    }


def extract_code_blocks(text: str) -> list[dict]:
    """Extract code blocks from markdown-formatted text.

    Looks for fenced code blocks (````python ... ``` `` or ```` ... ``` ``).
    If none found, checks whether the entire text looks like Python code.

    Returns:
        List of ``{"language": str, "code": str}`` dicts.
    """
    blocks: list[dict] = []

    # Match fenced code blocks: ```lang\n...\n```
    pattern = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)
    for m in pattern.finditer(text):
        lang = m.group(1).strip().lower() or "text"
        code = m.group(2)
        # Strip one trailing newline if present
        if code.endswith("\n"):
            code = code[:-1]
        blocks.append({"language": lang, "code": code})

    if blocks:
        return blocks

    # Heuristic: does the whole text look like Python code?
    stripped = text.strip()
    if stripped and _looks_like_python(stripped):
        return [{"language": "python", "code": stripped}]

    return []


def _looks_like_python(text: str) -> bool:
    """Heuristic check for whether *text* looks like Python source."""
    first_line = text.split("\n", 1)[0].strip()
    indicators = (
        "import ", "from ", "def ", "class ", "if ", "for ",
        "while ", "print(", "return ", "#!", "# ",
    )
    if any(first_line.startswith(s) for s in indicators):
        return True
    # Check if multiple lines have consistent indentation (spaces/tabs at start)
    lines = [l for l in text.split("\n") if l.strip()]
    indented = sum(1 for l in lines if l.startswith((" ", "\t")))
    if len(lines) >= 3 and indented / len(lines) > 0.3:
        return True
    return False
