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
import time
from contextvars import ContextVar
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

# ── EvoHive structured logger (lazy init to avoid circular import) ──
_dm_logger = None
_log_event_fn = None

_SESSION_API_KEYS: ContextVar[dict[str, str]] = ContextVar("evohive_session_api_keys", default={})
_SESSION_COST_TRACKER: ContextVar[object | None] = ContextVar("evohive_session_cost_tracker", default=None)
_SESSION_COST_PHASE: ContextVar[str] = ContextVar("evohive_session_cost_phase", default="llm")
_SESSION_TOKEN_BUDGET: ContextVar[dict | None] = ContextVar("evohive_session_token_budget", default=None)
_PROVIDER_ALIASES = {
    "together_ai": "together",
    "fireworks_ai": "fireworks",
}


def set_session_api_keys(api_keys: dict[str, str]):
    """Attach per-run API keys to the current async context."""
    normalized = {
        _PROVIDER_ALIASES.get(str(provider).strip().lower(), str(provider).strip().lower()): str(secret).strip()
        for provider, secret in (api_keys or {}).items()
        if str(provider).strip() and str(secret).strip()
    }
    return _SESSION_API_KEYS.set(normalized)


def reset_session_api_keys(token) -> None:
    _SESSION_API_KEYS.reset(token)


def _session_api_key_for_provider(provider: str) -> str:
    normalized = _PROVIDER_ALIASES.get((provider or "").strip().lower(), (provider or "").strip().lower())
    return _SESSION_API_KEYS.get().get(normalized, "")


def set_session_cost_tracker(cost_tracker, phase: str = "llm"):
    """Attach a CostTracker to the current async context."""
    tracker_token = _SESSION_COST_TRACKER.set(cost_tracker)
    phase_token = _SESSION_COST_PHASE.set(phase or "llm")
    return tracker_token, phase_token


def set_session_cost_phase(phase: str) -> None:
    """Tag subsequent LLM calls in this async context with a cost phase."""
    _SESSION_COST_PHASE.set(phase or "llm")


def reset_session_cost_tracker(tokens) -> None:
    tracker_token, phase_token = tokens
    _SESSION_COST_PHASE.reset(phase_token)
    _SESSION_COST_TRACKER.reset(tracker_token)


def clear_session_cost_tracker() -> None:
    """Clear cost tracking from the current async context."""
    _SESSION_COST_PHASE.set("llm")
    _SESSION_COST_TRACKER.set(None)


def set_session_token_budget(plan: dict, *, enabled: bool = True, min_output_tokens: int = 1):
    """Attach a token budget plan to the current async context.

    The provider uses this as a last-mile guard: it clips per-call max_tokens
    using the current cost phase and the tokens already recorded by CostTracker.
    """
    state = {
        "plan": plan or {},
        "enabled": bool(enabled),
        "min_output_tokens": max(1, int(min_output_tokens)),
        "events": [],
    }
    return _SESSION_TOKEN_BUDGET.set(state)


def reset_session_token_budget(token) -> None:
    _SESSION_TOKEN_BUDGET.reset(token)


def clear_session_token_budget() -> None:
    _SESSION_TOKEN_BUDGET.set(None)


def get_session_token_budget_events() -> list[dict]:
    state = _SESSION_TOKEN_BUDGET.get()
    if not state:
        return []
    return list(state.get("events", []))


def _usage_value(usage, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _record_session_cost(model: str, response) -> None:
    tracker = _SESSION_COST_TRACKER.get()
    if tracker is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return
    input_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    if input_tokens <= 0 and output_tokens <= 0:
        total_tokens = _usage_value(usage, "total_tokens")
        input_tokens = total_tokens
    tracker.record_call(model, input_tokens, output_tokens, phase=_SESSION_COST_PHASE.get())


def _estimate_prompt_tokens(messages: list[dict]) -> int:
    """Cheap token estimate for pre-call budget clipping.

    We intentionally avoid provider-specific tokenizers here; this runs before
    every LLM call, so a conservative chars/4 estimate is good enough to keep
    runaway outputs from blowing up a phase budget.
    """
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get("role", ""))) + len(str(message.get("content", ""))) + 4
    return max(1, total_chars // 4)


def _apply_session_token_budget(max_tokens: int, messages: list[dict]) -> int:
    state = _SESSION_TOKEN_BUDGET.get()
    tracker = _SESSION_COST_TRACKER.get()
    if not state or not state.get("enabled") or tracker is None:
        return max_tokens

    plan = state.get("plan") or {}
    total_budget = int(plan.get("total_token_budget") or 0)
    if total_budget <= 0:
        return max_tokens

    phase = _SESSION_COST_PHASE.get()
    phase_budgets = plan.get("phase_budgets") or {}
    phase_budget = int(phase_budgets.get(phase) or 0)
    policy = plan.get("policy") or {}
    hard_stop_at = float(policy.get("hard_stop_at", 1.15))

    snapshot = tracker.snapshot()
    total_used = int(snapshot.get("total_input_tokens", 0)) + int(snapshot.get("total_output_tokens", 0))
    total_limit = max(1, int(total_budget * hard_stop_at))
    total_room = total_limit - total_used

    effective_room = total_room
    phase_room = None
    if phase_budget > 0:
        phase_data = (snapshot.get("phases") or {}).get(phase, {})
        phase_used = int(phase_data.get("input_tokens", 0)) + int(phase_data.get("output_tokens", 0))
        phase_limit = max(1, int(phase_budget * hard_stop_at))
        phase_room = phase_limit - phase_used
        effective_room = min(total_room, phase_room)

    prompt_estimate = _estimate_prompt_tokens(messages)
    available_output = effective_room - prompt_estimate
    if available_output >= max_tokens:
        return max_tokens

    capped = max(1, min(max_tokens, available_output))
    capped = max(capped, min(int(state.get("min_output_tokens", 1)), max_tokens))
    if capped >= max_tokens:
        return max_tokens

    event = {
        "version": "llm-token-budget-cap.v1",
        "phase": phase,
        "requested_max_tokens": max_tokens,
        "capped_max_tokens": capped,
        "estimated_prompt_tokens": prompt_estimate,
        "total_budget_tokens": total_budget,
        "total_tokens_used_before_call": total_used,
        "total_room_tokens": total_room,
        "phase_budget_tokens": phase_budget,
        "phase_room_tokens": phase_room,
        "reason": "prompt_exceeds_remaining" if available_output <= 0 else "output_capped",
    }
    state.setdefault("events", []).append(event)
    log_event(
        _get_dm_logger(),
        "llm_token_budget_cap",
        phase=phase,
        requested=max_tokens,
        capped=capped,
        reason=event["reason"],
    )
    return capped

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
    "groq": 15,
    "deepseek": 25,
    "gemini": 25,
    "openai": 30,
    "anthropic": 20,
}
_DEFAULT_RATE_LIMIT = 20


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
    session_api_key = _session_api_key_for_provider(prefix)
    if prefix in CUSTOM_PROVIDERS:
        cfg = CUSTOM_PROVIDERS[prefix]
        # 去掉自定义前缀，换成litellm识别的前缀
        actual_model_name = model[len(prefix) + 1:]  # e.g. "Qwen/Qwen2.5-7B-Instruct"
        resolved_model = f"{cfg['litellm_prefix']}/{actual_model_name}"
        extra_kwargs["api_base"] = cfg["api_base"]
        api_key = session_api_key or os.environ.get(cfg["env_var"], "")
        if api_key:
            extra_kwargs["api_key"] = api_key
    elif session_api_key:
        extra_kwargs["api_key"] = session_api_key

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
        fallback_errors: list[str] = []
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
            except Exception as fallback_error:
                fallback_errors.append(f"{fallback}: {fallback_error}")
                log_event(_get_dm_logger(), "fallback_failed",
                          original=model, fallback=fallback,
                          error=str(fallback_error)[:200])
                continue
        # All fallbacks failed too
        if fallback_errors:
            joined = " | ".join(fallback_errors[:3])
            raise RuntimeError(f"{primary_error}; fallback failures: {joined}") from primary_error
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
    requested_max_tokens = max_tokens
    max_tokens = _apply_session_token_budget(max_tokens, messages)
    if max_tokens < requested_max_tokens:
        budget_note = (
            "\n\nToken budget notice: produce a concise but complete response "
            f"within {max_tokens} output tokens. Prefer a short finished answer "
            "over a long answer that stops mid-sentence."
        )
        if json_mode:
            budget_note += " Return compact valid JSON only."
        else:
            budget_note += " Use short bullets or short paragraphs, and end with a complete final sentence."
        messages[0]["content"] = f"{messages[0]['content']}{budget_note}"

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
            _record_session_cost(model, response)
            return response.choices[0].message.content
        except Exception as e:
            if e.__class__.__name__ == "BudgetExceededError":
                raise
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
