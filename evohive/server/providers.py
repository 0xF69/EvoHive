try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from evohive.llm.model_registry import CUSTOM_PROVIDER_CONFIGS, MODEL_REGISTRY

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "together": "TOGETHERAI_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "cohere": "COHERE_API_KEY",
    "zhipuai": "ZHIPUAI_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "baichuan": "BAICHUAN_API_KEY",
    "yi": "YI_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "volcengine": "VOLCENGINE_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}
SEARCH_ENV_VARS = {
    "tavily": "TAVILY_API_KEY",
    "serper": "SERPER_API_KEY",
}
LIVE_DISCOVERY_BASES = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "xai": "https://api.x.ai/v1",
    "together": "https://api.together.xyz/v1",
}


def normalize_api_keys(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {}
    normalized = {}
    for provider, value in raw.items():
        if provider not in PROVIDER_ENV_VARS:
            continue
        key = (value or "").strip()
        if key:
            normalized[provider] = key
    return normalized


def normalize_search_api_keys(raw: dict | None) -> dict:
    if not isinstance(raw, dict):
        return {}
    normalized = {}
    for provider, value in raw.items():
        if provider not in SEARCH_ENV_VARS:
            continue
        key = (value or "").strip()
        if key:
            normalized[provider] = key
    return normalized


def redact_config(config: dict) -> dict:
    safe = dict(config or {})
    if "api_keys" in safe:
        safe["api_keys"] = {provider: "[REDACTED]" for provider in normalize_api_keys(safe.get("api_keys")).keys()}
    if "search_api_keys" in safe:
        safe["search_api_keys"] = {
            provider: "[REDACTED]"
            for provider in normalize_search_api_keys(safe.get("search_api_keys")).keys()
        }
    return safe


def registry_models_for_provider(provider: str) -> list[str]:
    aliases = {
        "together": "together_ai",
        "fireworks": "fireworks_ai",
    }
    normalized = aliases.get(provider, provider)
    models = [entry.id.split("/", 1)[1] for entry in MODEL_REGISTRY if entry.provider == normalized]
    seen = set()
    ordered = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    return ordered


def api_base_for_provider(provider: str) -> str | None:
    if provider in LIVE_DISCOVERY_BASES:
        return LIVE_DISCOVERY_BASES[provider]
    custom = CUSTOM_PROVIDER_CONFIGS.get(provider)
    if custom:
        return custom.get("api_base")
    return None


async def discover_openai_compatible_models(provider: str, api_key: str) -> list[str]:
    api_base = api_base_for_provider(provider)
    if not api_base or not HAS_HTTPX:
        return []
    url = f"{api_base.rstrip('/')}/models"
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        payload = resp.json()
    models = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            models.append(model_id.strip())
    return sorted(set(models))


async def discover_anthropic_models(api_key: str) -> list[str]:
    if not HAS_HTTPX:
        return []
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        resp.raise_for_status()
        payload = resp.json()
    models = []
    for item in payload.get("data", []):
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            models.append(model_id.strip())
    return sorted(set(models))


async def discover_gemini_models(api_key: str) -> list[str]:
    if not HAS_HTTPX:
        return []
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"key": api_key},
        )
        resp.raise_for_status()
        payload = resp.json()
    models = []
    for item in payload.get("models", []):
        name = item.get("name", "")
        if not isinstance(name, str):
            continue
        model_id = name.split("/", 1)[-1].strip()
        methods = item.get("supportedGenerationMethods") or []
        if model_id and "generateContent" in methods:
            models.append(model_id)
    return sorted(set(models))


async def discover_provider_models_live(provider: str, api_key: str) -> list[str]:
    provider = (provider or "").strip().lower()
    secret = (api_key or "").strip()
    if not provider or not secret:
        return []
    if provider == "anthropic":
        return await discover_anthropic_models(secret)
    if provider == "gemini":
        return await discover_gemini_models(secret)
    return await discover_openai_compatible_models(provider, secret)


async def discover_provider_models_with_source(provider: str, api_key: str) -> tuple[list[str], str]:
    provider = (provider or "").strip().lower()
    secret = (api_key or "").strip()
    if not provider or not secret:
        return [], "fallback"
    try:
        models = await discover_provider_models_live(provider, secret)
        if models:
            return models, "live"
        return registry_models_for_provider(provider), "fallback"
    except Exception:
        return registry_models_for_provider(provider), "fallback"


async def discover_provider_models(provider: str, api_key: str) -> list[str]:
    models, _source = await discover_provider_models_with_source(provider, api_key)
    return models


async def probe_provider_access(provider: str, api_key: str) -> dict:
    provider = (provider or "").strip().lower()
    secret = (api_key or "").strip()
    if not provider or not secret:
        return {
            "provider": provider,
            "ok": False,
            "error_code": "missing_api_key",
            "message": "API key is required for this provider.",
            "models": [],
        }
    http_status_error = getattr(httpx, "HTTPStatusError", None) if HAS_HTTPX else None
    try:
        models = await discover_provider_models_live(provider, secret)
        return {
            "provider": provider,
            "ok": bool(models),
            "error_code": None if models else "model_unavailable",
            "message": "Provider is reachable." if models else "Provider is reachable but no models were discovered.",
            "models": models,
            "source": "live",
        }
    except Exception as exc:
        if http_status_error and isinstance(exc, http_status_error):
            status = getattr(exc.response, "status_code", None)
            detail = str(exc)
            if status in {401, 403}:
                return {
                    "provider": provider,
                    "ok": False,
                    "error_code": "auth_failed",
                    "message": detail,
                    "models": [],
                }
            if status in {402, 429}:
                return {
                    "provider": provider,
                    "ok": False,
                    "error_code": "quota_exhausted",
                    "message": detail,
                    "models": [],
                }
            return {
                "provider": provider,
                "ok": False,
                "error_code": "provider_unreachable",
                "message": detail,
                "models": [],
            }
        detail = str(exc)
        lowered = detail.lower()
        if any(word in lowered for word in ["quota", "billing", "insufficient", "credit", "429"]):
            code = "quota_exhausted"
        elif any(word in lowered for word in ["auth", "unauthorized", "forbidden", "permission", "401", "403"]):
            code = "auth_failed"
        else:
            code = "provider_unreachable"
        return {
            "provider": provider,
            "ok": False,
            "error_code": code,
            "message": detail or "Provider access probe failed.",
            "models": [],
        }


async def probe_manual_models(provider: str, api_key: str, model_ids: list[str]) -> dict:
    normalized = sorted({str(model).strip() for model in (model_ids or []) if str(model).strip()})
    if not normalized:
        return {
            "provider": provider,
            "ok": True,
            "mode": "none",
            "source": "none",
            "valid": [],
            "invalid": [],
            "unchecked": [],
        }
    probe = await probe_provider_access(provider, api_key)
    if not probe.get("ok"):
        return {
            "provider": provider,
            "ok": False,
            "mode": "provider_error",
            "source": probe.get("source", "fallback"),
            "valid": [],
            "invalid": normalized,
            "unchecked": [],
            "error_code": probe.get("error_code"),
            "message": probe.get("message"),
        }
    source = probe.get("source", "fallback")
    discovered = {str(model).strip() for model in (probe.get("models") or []) if str(model).strip()}
    if source == "live":
        valid = [model for model in normalized if model in discovered]
        invalid = [model for model in normalized if model not in discovered]
        return {
            "provider": provider,
            "ok": not invalid,
            "mode": "verified",
            "source": source,
            "valid": valid,
            "invalid": invalid,
            "unchecked": [],
        }
    return {
        "provider": provider,
        "ok": True,
        "mode": "fallback",
        "source": source,
        "valid": [],
        "invalid": [],
        "unchecked": normalized,
    }
