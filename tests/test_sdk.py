"""Tests for the EvoHive Python SDK (sdk.py)."""

import asyncio
import json
import random
from datetime import datetime, timedelta

import pytest
from unittest.mock import patch, MagicMock

from evohive.sdk import evolve, evolve_sync, EvolutionResult
from evohive.models import EvolutionConfig, EvolutionRun, GenerationStats


# ═══════════════════════════════════════════════════
# Fixtures / helpers
# ═══════════════════════════════════════════════════

def _make_run(**overrides) -> EvolutionRun:
    """Build a realistic EvolutionRun for testing EvolutionResult."""
    now = datetime.now()
    defaults = dict(
        id="test_run_001",
        config=EvolutionConfig(problem="Test problem"),
        started_at=now - timedelta(seconds=42.5),
        finished_at=now,
        generations_data=[
            GenerationStats(
                generation=1,
                best_fitness=0.85,
                avg_fitness=0.62,
                worst_fitness=0.31,
                alive_count=6,
                eliminated_count=4,
            ),
            GenerationStats(
                generation=2,
                best_fitness=0.91,
                avg_fitness=0.74,
                worst_fitness=0.55,
                alive_count=6,
                eliminated_count=4,
            ),
        ],
        final_top_solutions=[
            {"content": "Top solution alpha", "fitness": 0.91, "id": "aaa"},
            {"content": "Top solution beta", "fitness": 0.87, "id": "bbb"},
            {"content": "Top solution gamma", "fitness": 0.82, "id": "ccc"},
        ],
        baseline_solution="A simple baseline answer.",
        refined_top_solution="A deeply refined top solution with details.",
        total_api_calls=150,
        estimated_cost=0.0345,
    )
    defaults.update(overrides)
    return EvolutionRun(**defaults)


# ═══════════════════════════════════════════════════
# Mock LLM (reuses pattern from test_integration.py)
# ═══════════════════════════════════════════════════

def _mock_llm_content(messages: list[dict]) -> str:
    combined = ""
    for m in messages:
        combined += " " + (m.get("content") or "")
    combined = combined.lower()

    if any(kw in combined for kw in ["judge", "evaluate", "scoring"]):
        return json.dumps({
            "scores": [
                {"name": "quality", "score": 7, "reason": "ok"},
                {"name": "feasibility", "score": 6, "reason": "ok"},
            ],
            "reasoning": "Good.",
        })
    if any(kw in combined for kw in ["pairwise", "compare", "winner"]):
        return json.dumps({"winner": "A", "reason": "Better", "confidence": 0.8})
    if any(kw in combined for kw in ["failure", "eliminat"]):
        return json.dumps({"failure_reasons": ["too vague"]})
    if any(kw in combined for kw in ["red team", "attack", "vulnerab"]):
        return json.dumps({
            "attacks": [{"attack": "edge case", "success": False, "severity": 3}],
            "defense_analysis": "Robust",
            "overall_vulnerability": 0.3,
        })
    if any(kw in combined for kw in ["mutate", "mutation"]):
        return "Mutated solution with improvements."
    if any(kw in combined for kw in ["crossover", "combine", "merge"]):
        return "Crossover solution combining parents."
    if any(kw in combined for kw in ["baseline"]):
        return "Baseline solution."
    if any(kw in combined for kw in ["refine", "expand", "chapter"]):
        if any(kw in combined for kw in ["outline", "chapter"]):
            return json.dumps({"chapters": ["Summary", "Plan"]})
        return "Refined section."
    if any(kw in combined for kw in ["thinker", "persona"]):
        if "json" in combined:
            return json.dumps([
                {"persona": "Analyst"},
                {"persona": "Architect"},
                {"persona": "Innovator"},
                {"persona": "Risk expert"},
                {"persona": "UX researcher"},
                {"persona": "Data scientist"},
            ])
        return json.dumps({"persona": "Strategist"})
    if any(kw in combined for kw in ["solution"]):
        return "A comprehensive solution."
    return "Mock response."


def _build_litellm_response(content: str) -> MagicMock:
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
    messages = kwargs.get("messages", [])
    return _build_litellm_response(_mock_llm_content(messages))


async def _mock_get_embedding(text, model="text-embedding-3-small"):
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    return [rng.gauss(0, 1) for _ in range(256)]


async def _mock_preflight_check(models, timeout=10.0):
    return {"ok": list(models), "failed": []}


def _patch_all():
    return (
        patch("litellm.acompletion", side_effect=_mock_acompletion),
        patch("evohive.engine.embedding.get_embedding", side_effect=_mock_get_embedding),
        patch("evohive.engine.evolution.preflight_check", side_effect=_mock_preflight_check),
    )


# ═══════════════════════════════════════════════════
# Tests: EvolutionResult accessors
# ═══════════════════════════════════════════════════

class TestEvolutionResultAccessors:
    """Verify all EvolutionResult property accessors."""

    def test_best_solution_returns_refined(self):
        run = _make_run()
        result = EvolutionResult(run)
        assert result.best_solution == "A deeply refined top solution with details."

    def test_best_solution_falls_back_to_top_solution(self):
        run = _make_run(refined_top_solution="")
        result = EvolutionResult(run)
        assert result.best_solution == "Top solution alpha"

    def test_best_solution_empty_when_no_solutions(self):
        run = _make_run(refined_top_solution="", final_top_solutions=[])
        result = EvolutionResult(run)
        assert result.best_solution == ""

    def test_fitness(self):
        run = _make_run()
        result = EvolutionResult(run)
        assert result.fitness == pytest.approx(0.91)

    def test_fitness_zero_when_no_solutions(self):
        run = _make_run(final_top_solutions=[])
        result = EvolutionResult(run)
        assert result.fitness == 0.0

    def test_top_solutions(self):
        run = _make_run()
        result = EvolutionResult(run)
        top = result.top_solutions
        assert len(top) == 3
        assert top[0]["rank"] == 1
        assert top[0]["fitness"] == pytest.approx(0.91)
        assert top[0]["content"] == "Top solution alpha"
        assert top[2]["rank"] == 3

    def test_cost(self):
        run = _make_run()
        result = EvolutionResult(run)
        assert result.cost == pytest.approx(0.0345)

    def test_duration_seconds(self):
        run = _make_run()
        result = EvolutionResult(run)
        assert result.duration_seconds == pytest.approx(42.5, abs=0.1)

    def test_duration_zero_when_not_finished(self):
        run = _make_run(finished_at=None)
        result = EvolutionResult(run)
        assert result.duration_seconds == 0.0

    def test_generations_data(self):
        run = _make_run()
        result = EvolutionResult(run)
        assert len(result.generations_data) == 2
        assert result.generations_data[0].generation == 1
        assert result.generations_data[1].best_fitness == pytest.approx(0.91)

    def test_raw(self):
        run = _make_run()
        result = EvolutionResult(run)
        assert result.raw is run


# ═══════════════════════════════════════════════════
# Tests: EvolutionResult __str__
# ═══════════════════════════════════════════════════

class TestEvolutionResultStr:
    """Verify __str__ produces readable output."""

    def test_str_contains_key_info(self):
        run = _make_run()
        result = EvolutionResult(run)
        text = str(result)
        assert "EvoHive Evolution Result" in text
        assert "0.91" in text  # fitness
        assert "42.5" in text  # duration
        assert "150" in text  # api calls
        assert "$0.0345" in text  # cost
        assert "Best Solution" in text
        assert "deeply refined" in text

    def test_str_truncates_long_solution(self):
        run = _make_run(refined_top_solution="x" * 1000)
        result = EvolutionResult(run)
        text = str(result)
        assert text.endswith("...")
        # Should be truncated around 500 chars + "..."
        assert len(text.split("--- Best Solution ---\n")[-1]) <= 510

    def test_str_hides_zero_cost(self):
        run = _make_run(estimated_cost=0.0)
        result = EvolutionResult(run)
        text = str(result)
        assert "cost" not in text.lower()


# ═══════════════════════════════════════════════════
# Tests: evolve_sync end-to-end with mocked LLM
# ═══════════════════════════════════════════════════

class TestEvolveSyncBasic:
    """Test evolve_sync with fully mocked LLM calls."""

    def test_evolve_sync_returns_evolution_result(self):
        p1, p2, p3 = _patch_all()
        with p1, p2, p3:
            result = evolve_sync(
                "Design a better onboarding process",
                mode="fast",
                population_size=6,
                generations=1,
                save_results=False,
                enable_swarm=False,
                enable_web_search=False,
                enable_debate=False,
                enable_pressure_test=False,
                convergence_threshold=0.99,
            )

        assert isinstance(result, EvolutionResult)
        assert result.fitness > 0
        assert result.best_solution != ""
        assert result.duration_seconds > 0
        assert len(result.raw.final_top_solutions) > 0
        assert result.raw.finished_at is not None
