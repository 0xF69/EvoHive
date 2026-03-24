"""LLM统一调用接口，基于LiteLLM

支持 20+ Provider 自动检测:
  - OpenAI / Anthropic / Gemini / DeepSeek / Groq / Mistral / xAI
  - Together AI / Fireworks / Cohere / Perplexity
  - ZhipuAI / SiliconFlow / Moonshot / Baichuan / Yi / DashScope / Volcengine / Minimax

自定义 Provider (需要 api_base) 通过 CUSTOM_PROVIDERS 注册。
详见 model_registry.py 获取完整模型列表和自动分配逻辑。
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import litellm

# ═══ 彻底关闭LiteLLM的所有日志/打印输出 ═══
litellm.set_verbose = False
litellm.suppress_debug_info = True
litellm.turn_off_message_logging = True

# 关闭所有LiteLLM相关logger
for _logger_name in [
    "LiteLLM", "litellm", "LiteLLM Router", "LiteLLM Proxy",
    "litellm.utils", "litellm.main", "litellm.cost_calculator",
]:
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(_logger_name).handlers = []
    logging.getLogger(_logger_name).propagate = False
logging.getLogger("httpx").setLevel(logging.WARNING)

# 猴子补丁: LiteLLM用print()直接输出到stderr，logging压不住
_noop = lambda *args, **kwargs: None
litellm.print_verbose = _noop
try:
    import litellm.litellm_core_utils.litellm_logging as _ll_logging
    _ll_logging.print_verbose = _noop
except Exception:
    pass
try:
    import litellm.utils as _ll_utils
    _ll_utils.print_verbose = _noop
except Exception:
    pass
try:
    import litellm.main as _ll_main
    _ll_main.print_verbose = _noop
except Exception:
    pass
try:
    import litellm.litellm_core_utils as _ll_core
    _ll_core.print_verbose = _noop
except Exception:
    pass

# 终极补丁: 拦截builtins.print，过滤LiteLLM的直接print输出
import builtins
_original_print = builtins.print

def _filtered_print(*args, **kwargs):
    """拦截全局print，过滤LiteLLM的垃圾输出"""
    text = " ".join(str(a) for a in args)
    _litellm_keywords = [
        "LiteLLM", "litellm", "Give Feedback", "BerriAI",
        "Provider List:", "get_llm_provider", "completion()",
        "\x1b[92m", "\x1b[1m",
    ]
    if any(kw in text for kw in _litellm_keywords):
        return  # 静默吞掉
    return _original_print(*args, **kwargs)

builtins.print = _filtered_print

# ── EvoHive structured logger (lazy init to avoid circular import) ──
_dm_logger = None
_log_event_fn = None

def _get_dm_logger():
    global _dm_logger, _log_event_fn
    if _dm_logger is None:
        from evohive.engine.logger import get_logger, log_event as _le
        _dm_logger = get_logger("evohive.llm.provider")
        _log_event_fn = _le
    return _dm_logger

def log_event(logger, *args, **kwargs):
    global _log_event_fn
    if _log_event_fn is None:
        _get_dm_logger()
    _log_event_fn(logger, *args, **kwargs)

# ── 自定义Provider注册表 ──
# 对于LiteLLM不原生支持或支持不稳定的Provider，在此定义api_base和api_key映射
CUSTOM_PROVIDERS = {
    "siliconflow": {
        "api_base": "https://api.siliconflow.cn/v1",
        "env_var": "SILICONFLOW_API_KEY",
        "litellm_prefix": "openai",
    },
    "zhipuai": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "env_var": "ZHIPUAI_API_KEY",
        "litellm_prefix": "openai",
    },
    "moonshot": {
        "api_base": "https://api.moonshot.cn/v1",
        "env_var": "MOONSHOT_API_KEY",
        "litellm_prefix": "openai",
    },
    "baichuan": {
        "api_base": "https://api.baichuan-ai.com/v1",
        "env_var": "BAICHUAN_API_KEY",
        "litellm_prefix": "openai",
    },
    "yi": {
        "api_base": "https://api.lingyiwanwu.com/v1",
        "env_var": "YI_API_KEY",
        "litellm_prefix": "openai",
    },
    "dashscope": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_var": "DASHSCOPE_API_KEY",
        "litellm_prefix": "openai",
    },
    "volcengine": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "env_var": "VOLCENGINE_API_KEY",
        "litellm_prefix": "openai",
    },
    "minimax": {
        "api_base": "https://api.minimax.chat/v1",
        "env_var": "MINIMAX_API_KEY",
        "litellm_prefix": "openai",
    },
}


# ═══ Error Classification ═══

class _ErrorType:
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    SERVER = "server"
    TIMEOUT = "timeout"
    OTHER = "other"


def _classify_error(error: Exception) -> str:
    """Classify an LLM error by inspecting status codes and error messages."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        # Some litellm exceptions wrap the status code differently
        original = getattr(error, "original_exception", None)
        if original is not None:
            status_code = getattr(original, "status_code", None)

    if status_code == 429:
        return _ErrorType.RATE_LIMIT
    if status_code in (401, 403):
        return _ErrorType.AUTH
    if status_code in (500, 502, 503):
        return _ErrorType.SERVER

    error_str = str(error).lower()
    if "timeout" in error_str or "timed out" in error_str:
        return _ErrorType.TIMEOUT
    if "rate" in error_str and "limit" in error_str:
        return _ErrorType.RATE_LIMIT
    if "auth" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
        return _ErrorType.AUTH

    return _ErrorType.OTHER


# ═══ Circuit Breaker (per-provider) ═══

class CircuitBreaker:
    """Per-provider circuit breaker with closed / open / half-open states."""

    FAILURE_THRESHOLD = 3
    RECOVERY_TIMEOUT = 60  # seconds

    def __init__(self, provider: str):
        self.provider = provider
        self.state = "closed"  # closed | open | half-open
        self.consecutive_failures = 0
        self.last_failure_time: float = 0.0
        self._lock = asyncio.Lock()

    async def allow_request(self) -> bool:
        """Return True if a request is allowed, False to reject immediately."""
        async with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                elapsed = time.monotonic() - self.last_failure_time
                if elapsed >= self.RECOVERY_TIMEOUT:
                    log_event(_get_dm_logger(), "circuit_breaker_state_change",
                              provider=self.provider,
                              old_state="open", new_state="half-open")
                    self.state = "half-open"
                    return True
                return False
            # half-open: allow one probe
            return True

    async def record_success(self):
        async with self._lock:
            old_state = self.state
            self.consecutive_failures = 0
            self.state = "closed"
            if old_state != "closed":
                log_event(_get_dm_logger(), "circuit_breaker_state_change",
                          provider=self.provider,
                          old_state=old_state, new_state="closed")

    async def record_failure(self):
        async with self._lock:
            self.consecutive_failures += 1
            self.last_failure_time = time.monotonic()
            old_state = self.state
            if self.state == "half-open":
                # probe failed – re-open
                self.state = "open"
            elif self.consecutive_failures >= self.FAILURE_THRESHOLD:
                self.state = "open"
            if self.state != old_state:
                log_event(_get_dm_logger(), "circuit_breaker_state_change",
                          provider=self.provider,
                          old_state=old_state, new_state=self.state,
                          consecutive_failures=self.consecutive_failures)

    def status(self) -> dict:
        return {
            "provider": self.provider,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_time": self.last_failure_time,
        }


# Module-level circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def _get_circuit_breaker(provider: str) -> CircuitBreaker:
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker(provider)
    return _circuit_breakers[provider]


# ═══ Per-provider Concurrency Control ═══

_PROVIDER_RATE_LIMITS: dict[str, int] = {
    "groq": 5,
    "deepseek": 15,
    "gemini": 15,
    "openai": 20,
    "anthropic": 15,
}
_DEFAULT_RATE_LIMIT = 10


class ProviderRateLimiter:
    """Manages a per-provider asyncio.Semaphore for concurrency control."""

    def __init__(self, provider: str, max_concurrent: int):
        self.provider = provider
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)


# Module-level rate limiter registry
_provider_limiters: dict[str, ProviderRateLimiter] = {}


def _get_provider_limiter(provider: str) -> ProviderRateLimiter:
    if provider not in _provider_limiters:
        limit = _PROVIDER_RATE_LIMITS.get(provider, _DEFAULT_RATE_LIMIT)
        _provider_limiters[provider] = ProviderRateLimiter(provider, limit)
    return _provider_limiters[provider]


def _extract_provider(model: str) -> str:
    """Extract the provider name from a model identifier like 'deepseek/deepseek-chat'."""
    if "/" in model:
        return model.split("/")[0]
    return "default"


def get_provider_status() -> dict:
    """Return the status of all circuit breakers for CLI display."""
    return {
        name: cb.status()
        for name, cb in _circuit_breakers.items()
    }


def _resolve_model(model: str) -> dict:
    """解析模型标识符，返回litellm.acompletion所需的额外参数。

    对于标准Provider (deepseek/groq/gemini/zhipuai)，LiteLLM原生支持，
    直接透传即可。
    对于自定义Provider (如siliconflow)，需要替换前缀并注入api_base/api_key。
    """
    extra_kwargs = {}
    resolved_model = model

    # 检查是否命中自定义Provider
    prefix = model.split("/")[0] if "/" in model else ""
    if prefix in CUSTOM_PROVIDERS:
        cfg = CUSTOM_PROVIDERS[prefix]
        # 去掉自定义前缀，换成litellm识别的前缀
        actual_model_name = model[len(prefix) + 1:]  # e.g. "Qwen/Qwen2.5-7B-Instruct"
        resolved_model = f"{cfg['litellm_prefix']}/{actual_model_name}"
        extra_kwargs["api_base"] = cfg["api_base"]
        api_key = os.environ.get(cfg["env_var"], "")
        if api_key:
            extra_kwargs["api_key"] = api_key

    return {"model": resolved_model, **extra_kwargs}


# ── Fallback model registry (populated at runtime by evolution engine) ──
_fallback_models: list[str] = []

def set_fallback_models(models: list[str]):
    """Set the list of fallback models to try when primary model fails.
    Called by the evolution engine after pre-flight check."""
    global _fallback_models
    _fallback_models = list(models)


async def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
) -> str:
    """统一的LLM调用接口（含自动重试 + 自动降级）

    Args:
        model: 模型标识符，如 "deepseek/deepseek-chat" 或 "siliconflow/Qwen/Qwen2.5-7B-Instruct"
        api_base: 可选，手动指定api_base（优先于自动解析）
        api_key: 可选，手动指定api_key（优先于自动解析）
        max_retries: 失败后重试次数（默认2次，总共最多3次尝试）
    """
    try:
        return await _call_llm_single(
            model, system_prompt, user_prompt,
            temperature, max_tokens, json_mode,
            api_base, api_key, max_retries,
        )
    except RuntimeError as e:
        # Try fallback models if primary fails
        primary_error = e
        provider = _extract_provider(model)
        for fallback in _fallback_models:
            if fallback == model:
                continue
            # Skip models from the same broken provider
            fb_provider = _extract_provider(fallback)
            if fb_provider == provider:
                continue
            # Check circuit breaker for fallback
            fb_cb = _get_circuit_breaker(fb_provider)
            if not await fb_cb.allow_request():
                continue
            try:
                log_event(_get_dm_logger(), "fallback_attempt",
                          original=model, fallback=fallback,
                          reason=str(primary_error)[:100])
                return await _call_llm_single(
                    fallback, system_prompt, user_prompt,
                    temperature, max_tokens, json_mode,
                    None, None, 1,  # fewer retries for fallback
                )
            except Exception:
                continue
        # All fallbacks failed too
        raise primary_error


async def _call_llm_single(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
    api_base: str | None = None,
    api_key: str | None = None,
    max_retries: int = 2,
) -> str:
    """Single-model LLM call with retries (no fallback)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 自动解析Provider配置
    resolved = _resolve_model(model)

    kwargs = {
        **resolved,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # 手动传入的api_base/api_key优先
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_error = None
    provider = _extract_provider(model)
    cb = _get_circuit_breaker(provider)
    limiter = _get_provider_limiter(provider)

    for attempt in range(max_retries + 1):
        # Circuit breaker check
        if not await cb.allow_request():
            raise RuntimeError(
                f"Circuit breaker open for provider '{provider}' — "
                f"skipping call to {model}. Recent failures: {cb.consecutive_failures}"
            )

        try:
            async with limiter.semaphore:
                response = await litellm.acompletion(**kwargs)
            await cb.record_success()
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            error_type = _classify_error(e)

            # Auth errors: don't retry, mark provider unavailable
            if error_type == _ErrorType.AUTH:
                await cb.record_failure()
                log_event(_get_dm_logger(), "auth_failure",
                          provider=provider, model=model, error=str(e))
                raise RuntimeError(
                    f"Auth error for provider '{provider}' ({model}): {e}. "
                    f"Check your API key / permissions."
                )

            await cb.record_failure()

            if error_type == _ErrorType.RATE_LIMIT:
                log_event(_get_dm_logger(), "rate_limit_hit",
                          provider=provider, model=model,
                          attempt=attempt + 1)

            if attempt >= max_retries:
                break

            # Compute backoff based on error type
            if error_type == _ErrorType.RATE_LIMIT:
                wait = min(2 ** (attempt + 1) * 2, 60)  # 4s, 8s, 16s...
            elif error_type == _ErrorType.SERVER:
                wait = 2 ** attempt * 1.5  # 1.5s, 3s, 6s
            elif error_type == _ErrorType.TIMEOUT:
                wait = 2 ** attempt  # 1s, 2s, 4s
                # Increase timeout for next attempt
                kwargs["timeout"] = kwargs.get("timeout", 60) * 1.5
            else:
                wait = 1 * (attempt + 1)  # 1s, 2s, 3s (original behavior)

            log_event(_get_dm_logger(), "retry_attempt",
                      provider=provider, model=model,
                      attempt=attempt + 1, max_retries=max_retries,
                      error_type=error_type, backoff_s=round(wait, 2))
            await asyncio.sleep(wait)

    raise RuntimeError(f"LLM调用失败 ({model}，重试{max_retries}次后仍失败): {last_error}")


async def preflight_check(models: list[str], timeout: float = 10.0) -> dict:
    """Pre-flight check: validate that each model is reachable.

    Sends a minimal request (max_tokens=1) to each unique model.

    Returns:
        {"ok": [...models that work], "failed": [{"model": str, "error": str}, ...]}
    """
    unique_models = list(dict.fromkeys(models))  # deduplicate, preserve order

    async def _check_one(model: str) -> tuple[str, Exception | None]:
        resolved = _resolve_model(model)
        kwargs = {
            **resolved,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "timeout": timeout,
        }
        try:
            await litellm.acompletion(**kwargs)
            return (model, None)
        except Exception as e:
            return (model, e)

    results = await asyncio.gather(
        *[_check_one(m) for m in unique_models],
        return_exceptions=True,
    )

    ok: list[str] = []
    failed: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            # gather itself caught an unexpected error
            failed.append({"model": "unknown", "error": str(r)})
        else:
            model_name, err = r
            if err is None:
                ok.append(model_name)
            else:
                failed.append({"model": model_name, "error": str(err)})

    return {"ok": ok, "failed": failed}


async def call_llm_batch(
    calls: list[dict],
    max_concurrent: int = 20,
) -> list[str]:
    """批量并发调用LLM"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_call(params):
        async with semaphore:
            return await call_llm(**params)

    tasks = [limited_call(c) for c in calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 把异常转为错误字符串，避免整体崩溃
    processed = []
    for r in results:
        if isinstance(r, Exception):
            processed.append(f"ERROR: {r}")
        else:
            processed.append(r)
    return processed


def extract_json(text: str) -> dict | list | None:
    """从LLM回复中提取JSON，处理markdown代码块等情况"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 ```json ... ``` 中的内容
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 或 [ 到最后一个 } 或 ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    return None
