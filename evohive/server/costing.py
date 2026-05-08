import math
import re

from evohive.server.catalog import PROVIDERS


def estimate_model_unit_rate(model_name: str = "") -> float:
    name = (model_name or "").lower()
    if re.search(r"opus|gpt-5|o1|gemini-2\.5-pro|claude-sonnet-4", name):
        return 0.055
    if re.search(r"gpt-4\.1|gpt-4o|reasoner|grok-3|mistral-large|command-r\+|glm-4-plus|qwen-max", name):
        return 0.028
    if re.search(r"gpt-4o-mini|gpt-4\.1-mini|gpt-4\.1-nano|haiku|flash|small|8b|9b|instant|turbo|command-r|glm-4-flash|qwen-plus|qwen-turbo", name):
        return 0.008
    if re.search(r"deepseek-chat|deepseek-reasoner|gemma|mini", name):
        return 0.012
    return 0.018


def provider_models_for_config(config: dict | None) -> dict[str, list[str]]:
    cfg = config or {}
    explicit = cfg.get("provider_models") or {}
    providers = cfg.get("providers") or []
    result: dict[str, list[str]] = {}
    for provider in providers:
        models = explicit.get(provider)
        if isinstance(models, list) and models:
            cleaned = [str(model).strip() for model in models if isinstance(model, str) and str(model).strip()]
            if cleaned:
                result[provider] = cleaned
                continue
        provider_entry = PROVIDERS.get(provider)
        if provider_entry:
            result[provider] = list(provider_entry.get("models") or [])
    return result


def estimate_run_cost(config: dict | None) -> dict:
    cfg = config or {}
    total = max(10, int(cfg.get("total", 60) or 60))
    gens = max(1, int(cfg.get("gens", 3) or 3))
    mode = cfg.get("mode", "fast")
    budget = max(0.1, float(cfg.get("budget", 0.5) or 0.5))
    enable_search = bool(cfg.get("enable_search", False))
    providers = list(cfg.get("providers") or [])
    provider_models = provider_models_for_config(cfg)
    all_models: list[str] = []
    for provider in providers:
        all_models.extend(provider_models.get(provider) or [])
    representative_models = all_models or ["deepseek-chat"]
    avg_rate = max(
        0.006,
        sum(estimate_model_unit_rate(model) for model in representative_models) / len(representative_models),
    )
    gen_calls = total
    duel_calls = round(total * max(1, math.log2(max(2, total))) * gens * 0.45)
    refine_calls = (gens * 6 if mode == "deep" else gens * 4 if mode == "balanced" else gens * 3) + max(1, round(total * 0.08))
    search_calls = max(2, gens * 2) if enable_search else 0
    base_tokens = (
        gen_calls * 0.0024
        + duel_calls * 0.0009
        + refine_calls * 0.0032
        + search_calls * 0.0011
    )
    provider_factor = 1 + max(0, len(providers) - 1) * 0.08
    low = max(0.02, base_tokens * avg_rate * 0.72 * provider_factor)
    mid = max(low, base_tokens * avg_rate * provider_factor)
    high = max(mid, base_tokens * avg_rate * 1.35 * provider_factor + (0.08 if enable_search else 0))
    risk = "hard" if mid > budget * 1.05 else "soft" if high > budget else "safe"
    return {
        "low": round(low, 2),
        "mid": round(mid, 2),
        "high": round(high, 2),
        "risk": risk,
        "budget": round(budget, 2),
        "duel_estimate": duel_calls,
    }
