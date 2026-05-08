"""Reasoning effort presets for EvoHive runs."""

from __future__ import annotations

from copy import deepcopy

from evohive.models import EvolutionConfig


EFFORT_PRESETS: dict[str, dict] = {
    "quick": {
        "mode": "fast",
        "population_size_max": 8,
        "generations_max": 1,
        "judge_rounds": 1,
        "enable_debate": False,
        "enable_pressure_test": False,
        "enable_red_team": False,
        "enable_elimination_memory": False,
        "enable_fresh_blood": False,
        "enable_swarm": False,
        "token_budget_multiplier": 0.45,
    },
    "balanced": {
        "mode": "fast",
        "population_size_min": 8,
        "population_size_max": 20,
        "generations_min": 2,
        "generations_max": 3,
        "judge_rounds": 1,
        "enable_debate": False,
        "enable_pressure_test": False,
        "enable_red_team": True,
        "enable_elimination_memory": True,
        "enable_fresh_blood": True,
        "enable_swarm": False,
        "token_budget_multiplier": 1.0,
    },
    "max": {
        "mode": "deep",
        "population_size_min": 20,
        "population_size_max": 80,
        "generations_min": 3,
        "generations_max": 8,
        "judge_rounds": 2,
        "enable_debate": True,
        "enable_pressure_test": True,
        "enable_red_team": True,
        "enable_elimination_memory": True,
        "enable_fresh_blood": True,
        "enable_swarm": True,
        "token_budget_multiplier": 2.5,
    },
}

MODE_TO_EFFORT = {
    "fast": "quick",
    "standard": "balanced",
    "balanced": "balanced",
    "deep": "max",
}

TOKEN_BUDGET_CONTROL_ALIASES = {
    "": "off",
    "false": "off",
    "none": "off",
    "off": "off",
    "disabled": "off",
    "disable": "off",
    "true": "auto",
    "on": "auto",
    "enabled": "auto",
    "enable": "auto",
    "auto": "auto",
    "balanced": "auto",
    "relaxed": "relaxed",
    "loose": "relaxed",
    "quality": "relaxed",
    "strict": "strict",
    "economy": "strict",
    "cheap": "strict",
    "fast": "strict",
}

TOKEN_BUDGET_CONTROL_FACTORS = {
    "auto": 1.0,
    "relaxed": 1.75,
    "strict": 0.65,
}


def normalize_reasoning_effort(value: str | None, mode: str | None = None) -> str:
    raw = (value or "").strip().lower()
    if raw in EFFORT_PRESETS:
        return raw
    return MODE_TO_EFFORT.get((mode or "").strip().lower(), "balanced")


def normalize_token_budget_control(value: str | None, enabled: bool = False) -> str:
    raw = (value or "").strip().lower()
    if not raw and enabled:
        return "auto"
    return TOKEN_BUDGET_CONTROL_ALIASES.get(raw, "auto" if enabled else "off")


def apply_reasoning_effort(config: EvolutionConfig) -> EvolutionConfig:
    """Return a copy of config constrained by its reasoning effort preset."""
    raw_effort = (config.reasoning_effort or "").strip().lower()
    explicit_effort = raw_effort in EFFORT_PRESETS or raw_effort in MODE_TO_EFFORT
    effort = normalize_reasoning_effort(raw_effort, config.mode)
    preset = EFFORT_PRESETS[effort]
    updated = deepcopy(config)
    updated.reasoning_effort = effort

    if not explicit_effort:
        _apply_token_budget_control(config, updated, preset)
        return updated

    updated.mode = preset["mode"]

    if "population_size_min" in preset:
        updated.population_size = max(updated.population_size, preset["population_size_min"])
    if "population_size_max" in preset:
        updated.population_size = min(updated.population_size, preset["population_size_max"])
    if "generations_min" in preset:
        updated.generations = max(updated.generations, preset["generations_min"])
    if "generations_max" in preset:
        updated.generations = min(updated.generations, preset["generations_max"])

    for key in (
        "judge_rounds",
        "enable_debate",
        "enable_pressure_test",
        "enable_red_team",
        "enable_elimination_memory",
        "enable_fresh_blood",
        "enable_swarm",
    ):
        setattr(updated, key, preset[key])

    updated.token_budget_multiplier = preset["token_budget_multiplier"]
    _apply_token_budget_control(config, updated, preset)
    return updated


def _apply_token_budget_control(config: EvolutionConfig, updated: EvolutionConfig, preset: dict) -> None:
    control = normalize_token_budget_control(config.token_budget_control, config.enable_token_budget_control)
    updated.token_budget_control = control

    if control == "off":
        updated.enable_token_budget_control = bool(config.enable_token_budget_control)
        if config.enable_token_budget_control and config.token_budget_multiplier != 1.0:
            updated.token_budget_multiplier = max(0.1, config.token_budget_multiplier)
        return

    updated.enable_token_budget_control = True
    if config.token_budget_multiplier != 1.0:
        updated.token_budget_multiplier = max(0.1, config.token_budget_multiplier)
    else:
        updated.token_budget_multiplier = max(
            0.1,
            preset["token_budget_multiplier"] * TOKEN_BUDGET_CONTROL_FACTORS[control],
        )


def effort_metadata(config: EvolutionConfig) -> dict:
    effort = normalize_reasoning_effort(config.reasoning_effort, config.mode)
    preset = EFFORT_PRESETS[effort]
    return {
        "reasoning_effort": effort,
        "mode": preset["mode"],
        "token_budget_multiplier": preset["token_budget_multiplier"],
        "description": {
            "quick": "lowest latency and cost; minimal verification",
            "balanced": "default answer evolution with verification and red-team checks",
            "max": "deeper search, debate, pressure testing, and larger populations",
        }[effort],
    }
