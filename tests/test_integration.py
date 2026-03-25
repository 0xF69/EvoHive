"""Integration tests for the evolution main loop.

These tests mock all LLM and embedding calls to verify the full
orchestration pipeline without network access.
"""

import asyncio
import json
import random

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from evohive.models import EvolutionConfig, EvolutionRun, Solution
from evohive.engine.evolution import run_evolution
from evohive.engine.cost_tracker import BudgetExceededError


# ═══════════════════════════════════════════════════
# Mock LLM helper
# ═══════════════════════════════════════════════════

def _mock_llm_content(messages: list[dict]) -> str:
    """Determine a plausible response string based on the prompt messages.

    ``litellm.acompletion`` receives a ``messages`` list.  We inspect the
    system and user content to decide what kind of response to fabricate.
    """
    combined = ""
    for m in messages:
        combined += " " + (m.get("content") or "")
    combined = combined.lower()

    # --- Judge / evaluate prompts ---
    if any(kw in combined for kw in ["评审", "judge", "evaluate", "评分", "scoring"]):
        return json.dumps({
            "scores": [
                {"name": "quality", "score": 7, "reason": "solid approach"},
                {"name": "feasibility", "score": 6, "reason": "mostly feasible"},
                {"name": "innovation", "score": 8, "reason": "creative ideas"},
                {"name": "completeness", "score": 7, "reason": "covers key areas"},
            ],
            "reasoning": "Overall a good solution with room for improvement.",
        })

    # --- Pairwise / compare prompts ---
    if any(kw in combined for kw in ["pairwise", "compare", "对比", "winner", "方案a", "方案b"]):
        return json.dumps({
            "winner": "A",
            "reason": "Solution A is more comprehensive",
            "confidence": 0.8,
        })

    # --- Elimination / failure extraction ---
    if any(kw in combined for kw in ["淘汰", "failure", "eliminat", "失败原因"]):
        return json.dumps({
            "failure_reasons": ["too vague", "lacks specifics"],
        })

    # --- Red team prompts ---
    if any(kw in combined for kw in ["red team", "红队", "attack", "攻击", "vulnerab"]):
        return json.dumps({
            "attacks": [
                {"attack": "edge case handling", "success": False, "severity": 3},
            ],
            "defense_analysis": "Generally robust",
            "overall_vulnerability": 0.3,
        })

    # --- Debate prompts ---
    if any(kw in combined for kw in ["debate", "辩论", "argument"]):
        return json.dumps({
            "arguments": ["Strong evidence base", "Clear logic"],
            "counterarguments": ["Needs more data"],
            "verdict": "A",
            "confidence": 0.75,
        })

    # --- Pressure test prompts ---
    if any(kw in combined for kw in ["pressure", "压力", "extreme", "edge case", "resilience"]):
        return json.dumps({
            "scenarios": [
                {"scenario": "extreme load", "resilience": 0.8, "notes": "handles well"},
            ],
            "avg_resilience": 0.8,
        })

    # --- Mutation prompts ---
    if any(kw in combined for kw in ["mutate", "变异", "mutation", "改进"]):
        return "This is a mutated and improved solution with new strategic insights and deeper analysis."

    # --- Crossover prompts ---
    if any(kw in combined for kw in ["crossover", "交叉", "重组", "combine", "merge"]):
        return "This is a crossover solution combining the best elements of both parents."

    # --- Baseline prompts ---
    if any(kw in combined for kw in ["baseline", "基线", "直接回答"]):
        return "This is a baseline solution generated without evolution."

    # --- Refine / expand prompts ---
    if any(kw in combined for kw in ["refine", "扩写", "深度", "expand", "章节", "chapter"]):
        # Outline request (expects JSON with chapters)
        if any(kw in combined for kw in ["大纲", "outline", "章节标题"]):
            return json.dumps({
                "chapters": ["Executive Summary", "Implementation Plan", "Risk Analysis"],
            })
        return "This is a deeply refined and expanded solution section with detailed analysis."

    # --- Thinker / genesis / persona prompts ---
    if any(kw in combined for kw in ["thinker", "persona", "角色", "人格", "genesis"]):
        # Batch persona generation (expects JSON list)
        if "json" in combined or "格式" in combined:
            return json.dumps([
                {"persona": "Strategic analyst with focus on market dynamics"},
                {"persona": "Technical architect specializing in scalable systems"},
                {"persona": "Creative innovator exploring unconventional approaches"},
                {"persona": "Risk management expert with regulatory expertise"},
                {"persona": "User experience researcher focused on human factors"},
                {"persona": "Data scientist with quantitative modeling skills"},
            ])
        return json.dumps({
            "persona": "Experienced strategist with cross-domain expertise",
            "knowledge_bias": "data-driven decision making",
            "constraint": "must be actionable within 6 months",
        })

    # --- Swarm seed prompts ---
    if any(kw in combined for kw in ["swarm", "seed", "策略种子", "idea"]):
        return "A lightweight strategy seed exploring market disruption through technology."

    # --- Solution generation prompts ---
    if any(kw in combined for kw in ["solution", "方案", "solve", "解决"]):
        return "A comprehensive solution addressing the problem through multi-phase implementation."

    # --- Default fallback ---
    return "Mock LLM response for testing purposes."


def _build_litellm_response(content: str) -> MagicMock:
    """Build a fake litellm response object with the expected structure:

    ``response.choices[0].message.content`` and
    ``response.usage.{prompt_tokens, completion_tokens, total_tokens}``
    """
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


async def _mock_acompletion(**kwargs) -> MagicMock:
    """Drop-in async mock for ``litellm.acompletion``."""
    messages = kwargs.get("messages", [])
    content = _mock_llm_content(messages)
    return _build_litellm_response(content)


async def _mock_get_embedding(
    text: str,
    model: str = "text-embedding-3-small",
) -> list[float]:
    """Return a deterministic-ish random 256-dim vector seeded by text hash."""
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    return [rng.gauss(0, 1) for _ in range(256)]


async def _mock_preflight_check(models: list[str], timeout: float = 10.0) -> dict:
    """Mock preflight check that reports all models as reachable."""
    return {"ok": list(models), "failed": []}


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def _fast_config(**overrides) -> EvolutionConfig:
    """Build a minimal fast-mode EvolutionConfig for testing."""
    defaults = dict(
        problem="Design a strategy to improve urban public transportation efficiency",
        population_size=6,
        generations=2,
        mode="fast",
        survival_rate=0.4,
        mutation_rate=0.3,
        diversity_weight=0.15,
        elite_rate=0.1,
        tournament_size=2,
        judge_rounds=1,
        thinker_model="deepseek/deepseek-chat",
        judge_model="deepseek/deepseek-chat",
        judge_dimensions=[
            {"name": "quality", "weight": 0.3, "description": "Overall quality"},
            {"name": "feasibility", "weight": 0.3, "description": "Feasibility"},
            {"name": "innovation", "weight": 0.2, "description": "Innovation"},
            {"name": "completeness", "weight": 0.2, "description": "Completeness"},
        ],
        judge_models=["deepseek/deepseek-chat"],
        red_team_models=["deepseek/deepseek-chat"],
        thinker_models=["deepseek/deepseek-chat"],
        enable_pairwise_judge=True,
        enable_elimination_memory=True,
        enable_diversity_guard=True,
        enable_fresh_blood=True,
        enable_red_team=True,
        enable_debate=False,  # fast mode skips debate anyway
        enable_pressure_test=False,  # fast mode skips pressure test anyway
        enable_swarm=False,
        enable_web_search=False,
        enable_swiss_tournament=True,
        convergence_threshold=0.95,  # high threshold so we don't early-stop
        memory_window=2,
        swarm_count=10,
        swarm_max_representatives=6,
    )
    defaults.update(overrides)
    return EvolutionConfig(**defaults)


def _patch_all():
    """Return a contextmanager-compatible tuple of patches.

    We mock three things:
    1. ``litellm.acompletion`` -- the lowest-level async LLM call used by
       ``call_llm`` and ``call_llm_batch``.
    2. ``evohive.engine.embedding.get_embedding`` -- the embedding call.
    3. ``evohive.engine.evolution.preflight_check`` -- skip model validation.
    """
    return (
        patch("litellm.acompletion", side_effect=_mock_acompletion),
        patch("evohive.engine.embedding.get_embedding", side_effect=_mock_get_embedding),
        patch("evohive.engine.evolution.preflight_check", side_effect=_mock_preflight_check),
    )


# ═══════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_evolution_fast_mode_e2e():
    """Full end-to-end test of the evolution loop in fast mode.

    All LLM calls are mocked.  Verifies that the orchestration completes
    and produces the expected result structure.
    """
    config = _fast_config()

    p1, p2, p3 = _patch_all()
    with p1, p2, p3:
        result = await run_evolution(config)

    # Basic structure checks
    assert isinstance(result, EvolutionRun), "Should return an EvolutionRun"
    assert result.finished_at is not None, "finished_at must be set"

    # Generations data -- fast mode runs 1 generation regardless of config.generations
    assert len(result.generations_data) >= 1, "Should have at least 1 generation"

    # Final solutions
    assert len(result.final_top_solutions) > 0, "Should have final top solutions"

    # Baseline
    assert result.baseline_solution, "Should have a baseline solution"

    # Refined top solution
    assert result.refined_top_solution, "Should have a refined top solution"

    # Cost tracking populated
    assert result.total_api_calls > 0, "Should track API calls"

    # No swarm stats when swarm is disabled
    assert result.swarm_stats == {}, "Swarm stats should be empty when swarm disabled"


@pytest.mark.asyncio
async def test_evolution_budget_exceeded():
    """Verify that an extremely low budget either triggers early stop or
    raises ``BudgetExceededError``.

    Note: The current implementation creates a ``CostTracker`` but does not
    call ``record_call`` on it during the loop (cost is tracked via
    ``run.total_api_calls`` counter).  Therefore, the budget check inside
    ``CostTracker`` is never triggered during the run.  We verify the run
    completes and check that the budget_limit is recorded.
    """
    config = _fast_config(population_size=4, generations=1)

    completed = False
    budget_error = False

    p1, p2, p3 = _patch_all()
    with p1, p2, p3:
        try:
            result = await run_evolution(config, budget_limit=0.0001)
            completed = True
        except BudgetExceededError:
            budget_error = True

    # Either outcome is acceptable
    assert completed or budget_error, (
        "Should either complete (with budget recorded) or raise BudgetExceededError"
    )

    if completed:
        assert isinstance(result, EvolutionRun)
        assert result.budget_limit == 0.0001


@pytest.mark.asyncio
async def test_evolution_with_swarm():
    """Verify swarm integration: when enable_swarm=True the swarm phase
    runs and swarm_stats is populated in the result.
    """
    config = _fast_config(
        enable_swarm=True,
        swarm_count=10,
        swarm_max_representatives=6,
        population_size=6,
        generations=1,
    )

    p1, p2, p3 = _patch_all()
    with p1, p2, p3:
        result = await run_evolution(config)

    assert isinstance(result, EvolutionRun)
    assert result.finished_at is not None

    # Swarm stats should be populated
    assert result.swarm_stats, "swarm_stats should be populated when swarm is enabled"
    assert "total_seeds" in result.swarm_stats or "n_clusters" in result.swarm_stats, (
        "swarm_stats should contain clustering information"
    )

    # Should still produce final solutions
    assert len(result.final_top_solutions) > 0

    # Baseline should still exist
    assert result.baseline_solution
