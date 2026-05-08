"""Token budget planning and post-run recommendations."""

from __future__ import annotations

from evohive.models import EvolutionConfig


PHASE_WEIGHTS = {
    "baseline": 0.04,
    "initialization": 0.12,
    "swarm": 0.14,
    "evaluation": 0.18,
    "elo_tournament": 0.18,
    "elimination_memory": 0.06,
    "crossover": 0.12,
    "mutation": 0.05,
    "fresh_blood": 0.04,
    "red_team": 0.04,
    "debate": 0.08,
    "pressure_test": 0.05,
    "final_evaluation": 0.06,
    "refinement": 0.08,
    "quality_comparison": 0.02,
}

BASE_TOKEN_BUDGETS = {
    "quick": 18_000,
    "balanced": 60_000,
    "max": 180_000,
}


def estimate_token_budget(config: EvolutionConfig) -> int:
    effort = config.reasoning_effort or "balanced"
    base = BASE_TOKEN_BUDGETS.get(effort, BASE_TOKEN_BUDGETS["balanced"])
    scale = max(config.population_size, 1) / 20
    generation_scale = max(config.generations, 1) / 3
    multiplier = max(config.token_budget_multiplier or 1.0, 0.1)
    return max(4_000, round(base * max(scale, 0.35) * max(generation_scale, 0.35) * multiplier))


def build_token_budget_plan(config: EvolutionConfig) -> dict:
    total_budget = estimate_token_budget(config)
    reserve_budget = max(1, round(total_budget * 0.05))
    allocatable_budget = max(1, total_budget - reserve_budget)
    active_weights = {
        phase: weight
        for phase, weight in PHASE_WEIGHTS.items()
        if _phase_enabled(config, phase)
    }
    active_weight_sum = sum(active_weights.values()) or 1.0
    phase_budgets = {
        phase: max(1, round(allocatable_budget * weight / active_weight_sum))
        for phase, weight in active_weights.items()
    }
    allocated = sum(phase_budgets.values())
    phase_budgets["reserve"] = max(0, total_budget - allocated)
    return {
        "version": "token-budget-plan.v1",
        "reasoning_effort": config.reasoning_effort or "balanced",
        "total_token_budget": total_budget,
        "phase_budgets": phase_budgets,
        "policy": {
            "soft_limit": True,
            "recommend_stop_at": 0.9,
            "hard_stop_at": 1.15,
            "reserve_ratio": 0.05,
        },
    }


def _phase_enabled(config: EvolutionConfig, phase: str) -> bool:
    if phase == "swarm":
        return config.enable_swarm
    if phase == "debate":
        return config.enable_debate and config.mode != "fast"
    if phase == "pressure_test":
        return config.enable_pressure_test and config.mode != "fast"
    if phase == "red_team":
        return config.enable_red_team
    if phase == "elimination_memory":
        return config.enable_elimination_memory
    if phase == "fresh_blood":
        return config.enable_fresh_blood
    if phase == "mutation":
        return config.mutation_rate > 0
    if phase == "refinement":
        return config.mode != "fast"
    return True


def build_token_budget_report(*, plan: dict, resource_report: dict) -> dict:
    phase_budgets = plan.get("phase_budgets", {})
    phase_usage = resource_report.get("phases", {})
    phases = {}
    for phase, budget in phase_budgets.items():
        used = phase_usage.get(phase, {}).get("total_tokens", 0)
        ratio = used / budget if budget else 0.0
        if ratio >= 1.15:
            action = "cut_or_skip_next_run"
        elif ratio >= 0.9:
            action = "watch_closely"
        elif used == 0 and phase != "reserve":
            action = "unused"
        else:
            action = "ok"
        phases[phase] = {
            "budget_tokens": budget,
            "used_tokens": used,
            "usage_ratio": round(ratio, 3),
            "remaining_tokens": budget - used,
            "recommended_action": action,
        }

    total_budget = plan.get("total_token_budget", 0)
    total_used = resource_report.get("total_tokens", 0)
    total_ratio = total_used / total_budget if total_budget else 0.0
    over_budget_phases = [
        phase for phase, data in phases.items()
        if data["recommended_action"] == "cut_or_skip_next_run"
    ]
    return {
        "version": "token-budget-report.v1",
        "total_token_budget": total_budget,
        "total_tokens_used": total_used,
        "usage_ratio": round(total_ratio, 3),
        "status": "over_budget" if total_ratio > 1.15 else "near_budget" if total_ratio > 0.9 else "within_budget",
        "over_budget_phases": over_budget_phases,
        "phase_reports": phases,
        "next_run_recommendations": _recommendations(total_ratio, over_budget_phases),
    }


def assess_live_token_budget(*, plan: dict, cost_snapshot: dict, checkpoint: str = "") -> dict:
    """Assess live token usage against a budget plan during a run."""
    phase_budgets = plan.get("phase_budgets", {})
    phase_usage = cost_snapshot.get("phases", {})
    policy = plan.get("policy", {})
    soft_limit = float(policy.get("recommend_stop_at", 0.9))
    hard_limit = float(policy.get("hard_stop_at", 1.15))
    total_budget = plan.get("total_token_budget", 0)
    total_used = cost_snapshot.get("total_input_tokens", 0) + cost_snapshot.get("total_output_tokens", 0)
    total_ratio = total_used / total_budget if total_budget else 0.0

    phase_decisions = {}
    for phase, budget in phase_budgets.items():
        data = phase_usage.get(phase, {})
        used = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        ratio = used / budget if budget else 0.0
        phase_decisions[phase] = {
            "budget_tokens": budget,
            "used_tokens": used,
            "usage_ratio": round(ratio, 3),
            "over_soft_limit": ratio >= soft_limit,
            "over_hard_limit": ratio >= hard_limit,
        }

    over_hard_phases = [
        phase for phase, data in phase_decisions.items()
        if data["over_hard_limit"]
    ]
    over_soft_phases = [
        phase for phase, data in phase_decisions.items()
        if data["over_soft_limit"]
    ]
    should_stop = total_ratio >= hard_limit
    should_skip_optional = total_ratio >= soft_limit or bool(over_hard_phases)

    if should_stop:
        status = "hard_stop"
    elif should_skip_optional:
        status = "soft_limit"
    else:
        status = "within_budget"

    return {
        "version": "live-token-budget-assessment.v1",
        "checkpoint": checkpoint,
        "status": status,
        "total_token_budget": total_budget,
        "total_tokens_used": total_used,
        "usage_ratio": round(total_ratio, 3),
        "over_soft_phases": over_soft_phases,
        "over_hard_phases": over_hard_phases,
        "should_skip_optional": should_skip_optional,
        "should_stop": should_stop,
        "phase_decisions": phase_decisions,
    }


def _recommendations(total_ratio: float, over_budget_phases: list[str]) -> list[str]:
    recs = []
    if total_ratio > 1.15:
        recs.append("lower reasoning_effort or reduce population_size")
    if "elo_tournament" in over_budget_phases:
        recs.append("switch to swiss_tournament or reduce judge_models")
    if "crossover" in over_budget_phases:
        recs.append("increase survival_rate to generate fewer children")
    if "refinement" in over_budget_phases:
        recs.append("skip deep refinement unless champion confidence is low")
    return recs or ["keep current budget profile"]
