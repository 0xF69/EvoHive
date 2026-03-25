"""Tests for v3.0 new modules: embedding, swiss_tournament, events, swarm, web_search,
cost_tracker, persistence, smart error handling"""

import asyncio
import pytest
import math
import os
import json
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

# ═══ Embedding Tests ═══

from evohive.engine.embedding import (
    cosine_similarity,
    jaccard_similarity,
    compute_diversity_scores,
    clear_embedding_cache,
)


def test_cosine_similarity_identical():
    vec = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(vec, vec) - 1.0) < 0.001


def test_cosine_similarity_orthogonal():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(vec_a, vec_b)) < 0.001


def test_cosine_similarity_opposite():
    vec_a = [1.0, 0.0]
    vec_b = [-1.0, 0.0]
    assert abs(cosine_similarity(vec_a, vec_b) - (-1.0)) < 0.001


def test_cosine_similarity_empty():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], []) == 0.0


def test_jaccard_similarity_identical():
    text = "hello world foo bar"
    assert jaccard_similarity(text, text) == 1.0


def test_jaccard_similarity_disjoint():
    assert jaccard_similarity("hello world", "foo bar") == 0.0


def test_jaccard_similarity_partial():
    sim = jaccard_similarity("hello world foo", "hello world bar")
    # intersection: {hello, world} = 2, union: {hello, world, foo, bar} = 4
    assert abs(sim - 0.5) < 0.001


def test_jaccard_similarity_empty():
    assert jaccard_similarity("", "hello") == 0.0


# ═══ Swiss Tournament Tests ═══

from evohive.engine.swiss_tournament import run_swiss_tournament
from evohive.models import Solution


@pytest.mark.asyncio
async def test_swiss_tournament_single_solution():
    sol = Solution(content="test solution")
    ratings = await run_swiss_tournament([sol], "test", ["model"], "dims")
    assert sol.id in ratings
    assert ratings[sol.id] == 1500.0


@pytest.mark.asyncio
async def test_swiss_tournament_two_solutions():
    sol_a = Solution(content="solution A is great")
    sol_b = Solution(content="solution B is mediocre")

    # Mock pairwise_compare to always pick A as winner
    with patch("evohive.engine.swiss_tournament.pairwise_compare") as mock_compare:
        mock_compare.return_value = {
            "winner_id": sol_a.id,
            "loser_id": sol_b.id,
            "reason": "A is better",
            "confidence": 0.8,
        }

        ratings = await run_swiss_tournament(
            [sol_a, sol_b], "test problem", ["mock_model"], "dims"
        )

        assert ratings[sol_a.id] > ratings[sol_b.id]


# ═══ Events Tests ═══

from evohive.engine.events import EventEmitter, EvolutionEvent


def test_event_emitter_basic():
    emitter = EventEmitter()
    received = []

    def callback(event):
        received.append(event)

    emitter.on_event(callback)
    emitter.emit("test_event", "test_phase", key="value")

    assert len(received) == 1
    assert received[0].type == "test_event"
    assert received[0].phase == "test_phase"
    assert received[0].data["key"] == "value"


def test_event_emitter_multiple_callbacks():
    emitter = EventEmitter()
    count = [0, 0]

    emitter.on_event(lambda e: count.__setitem__(0, count[0] + 1))
    emitter.on_event(lambda e: count.__setitem__(1, count[1] + 1))

    emitter.emit("test", "phase")
    assert count == [1, 1]


def test_event_emitter_log():
    emitter = EventEmitter()
    emitter.emit("event1", "phase1")
    emitter.emit("event2", "phase2")

    log = emitter.get_event_log()
    assert len(log) == 2
    assert log[0].type == "event1"
    assert log[1].type == "event2"


def test_event_emitter_callback_error_doesnt_crash():
    emitter = EventEmitter()

    def bad_callback(event):
        raise ValueError("intentional error")

    emitter.on_event(bad_callback)
    # Should not raise
    emitter.emit("test", "phase")
    assert len(emitter.get_event_log()) == 1


def test_evolution_event_to_dict():
    event = EvolutionEvent(type="test", phase="init", data={"key": "val"})
    d = event.to_dict()
    assert d["type"] == "test"
    assert d["phase"] == "init"
    assert d["data"]["key"] == "val"
    assert "timestamp" in d


# ═══ Swarm Tests ═══

from evohive.engine.swarm import (
    select_representatives,
    _random_cluster,
)


def test_select_representatives_basic():
    clusters = [
        [{"persona": "a", "seed": "long seed content here about strategy"}],
        [{"persona": "b", "seed": "short"}, {"persona": "c", "seed": "another longer seed about market"}],
    ]
    reps = select_representatives(clusters, max_representatives=2)
    assert len(reps) <= 2
    assert all("seed" in r for r in reps)


def test_select_representatives_empty():
    reps = select_representatives([], max_representatives=10)
    assert reps == []


def test_random_cluster():
    seeds = [{"persona": f"p{i}", "seed": f"seed {i}"} for i in range(20)]
    clusters = _random_cluster(seeds, 5)
    assert len(clusters) == 5
    total = sum(len(c) for c in clusters)
    assert total == 20


# ═══ Config Tests ═══

from evohive.config import EvoHiveConfig


def test_config_defaults():
    cfg = EvoHiveConfig()
    assert cfg.enable_swarm is True
    assert cfg.swarm_count == 500
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.enable_web_search is True
    assert cfg.enable_swiss_tournament is True
    assert cfg.mode == "deep"


def test_config_swarm_models_fallback():
    cfg = EvoHiveConfig()
    # When swarm_models is empty, should fallback to first thinker
    models = cfg.get_swarm_models()
    assert len(models) > 0


# ═══ EvolutionConfig Tests ═══

from evohive.models import EvolutionConfig


def test_evolution_config_fast_mode():
    cfg = EvolutionConfig(problem="test", mode="fast")
    assert cfg.mode == "fast"
    assert cfg.enable_swarm is True


def test_evolution_config_deep_mode():
    cfg = EvolutionConfig(problem="test", mode="deep")
    assert cfg.mode == "deep"


# ═══ Web Search Tests ═══

from evohive.engine.web_search import web_search


@pytest.mark.asyncio
async def test_web_search_no_api_key():
    """Without API keys, should return empty list gracefully"""
    # Clear env vars temporarily
    import os
    tavily = os.environ.pop("TAVILY_API_KEY", None)
    serper = os.environ.pop("SERPER_API_KEY", None)
    try:
        results = await web_search("test query")
        assert results == []
    finally:
        if tavily:
            os.environ["TAVILY_API_KEY"] = tavily
        if serper:
            os.environ["SERPER_API_KEY"] = serper


# ═══ Cost Tracker Tests ═══

from evohive.engine.cost_tracker import (
    CostTracker, BudgetExceededError, estimate_run_cost,
)


def test_cost_tracker_basic():
    tracker = CostTracker()
    # Record a call for a registered model
    cost = tracker.record_call("openai/gpt-4o", 1000, 500, phase="generation")
    assert cost > 0
    assert tracker.total_cost > 0
    assert tracker.call_count == 1
    assert tracker.total_input_tokens == 1000
    assert tracker.total_output_tokens == 500


def test_cost_tracker_budget_exceeded():
    tracker = CostTracker(budget_limit=0.001)
    # This should eventually exceed the tiny budget
    with pytest.raises(BudgetExceededError):
        for _ in range(100):
            tracker.record_call("openai/gpt-4o", 10000, 10000, phase="test")


def test_cost_tracker_no_budget():
    tracker = CostTracker(budget_limit=None)
    # Should never raise
    for _ in range(10):
        tracker.record_call("openai/gpt-4o", 1000, 500, phase="test")
    assert tracker.call_count == 10


def test_cost_tracker_unknown_model():
    tracker = CostTracker()
    cost = tracker.record_call("unknown/fake-model", 1000, 500, phase="test")
    assert cost == 0.0  # Unknown model, no cost


def test_cost_tracker_format_report():
    tracker = CostTracker(budget_limit=10.0)
    tracker.record_call("openai/gpt-4o", 1000, 500, phase="generation")
    tracker.record_call("deepseek/deepseek-chat", 800, 400, phase="swarm")
    report = tracker.format_report()
    assert "Cost Report" in report
    assert "openai" in report
    assert "deepseek" in report
    assert "generation" in report
    assert "swarm" in report


def test_cost_tracker_reset():
    tracker = CostTracker()
    tracker.record_call("openai/gpt-4o", 1000, 500, phase="test")
    assert tracker.call_count == 1
    tracker.reset()
    assert tracker.call_count == 0
    assert tracker.total_cost == 0.0


def test_estimate_run_cost():
    cfg = EvolutionConfig(problem="test", population_size=10, generations=1, mode="fast")
    result = estimate_run_cost(cfg, n_available_models=3)
    assert "estimated_min" in result
    assert "estimated_max" in result
    assert "breakdown" in result
    assert result["estimated_min"] <= result["estimated_max"]
    assert result["estimated_min"] >= 0


# ═══ Persistence Tests ═══

from evohive.engine.persistence import (
    save_run_result, load_run_result, format_markdown_report, list_previous_runs,
)
from evohive.models import EvolutionRun, GenerationStats


def _make_test_run() -> EvolutionRun:
    """Create a minimal test EvolutionRun"""
    cfg = EvolutionConfig(problem="test problem for persistence", mode="fast")
    return EvolutionRun(
        id="test_20260322_000000",
        config=cfg,
        started_at=datetime(2026, 3, 22, 10, 0, 0),
        finished_at=datetime(2026, 3, 22, 10, 5, 30),
        baseline_solution="This is the baseline.",
        total_api_calls=42,
        estimated_cost=1.23,
        refined_top_solution="This is the refined solution.",
        mode="fast",
        generations_data=[
            GenerationStats(
                generation=1, best_fitness=0.85, avg_fitness=0.60,
                worst_fitness=0.30, alive_count=8, eliminated_count=2,
            )
        ],
        final_top_solutions=[
            {"id": "sol1", "content": "Top solution content", "fitness": 0.85},
        ],
    )


def test_format_markdown_report():
    run = _make_test_run()
    md = format_markdown_report(run)
    assert "# EvoHive Evolution Report" in md
    assert "test problem for persistence" in md
    assert "Baseline" in md
    assert "Top 1 Solution" in md
    assert "0.85" in md


def test_save_and_load_run():
    run = _make_test_run()
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = save_run_result(run, output_dir=tmpdir)
        assert os.path.exists(json_path)
        assert os.path.exists(json_path.replace(".json", ".md"))

        loaded = load_run_result(json_path)
        assert loaded.id == run.id
        assert loaded.config.problem == "test problem for persistence"
        assert loaded.total_api_calls == 42


def test_list_previous_runs():
    run = _make_test_run()
    with tempfile.TemporaryDirectory() as tmpdir:
        save_run_result(run, output_dir=tmpdir)
        runs = list_previous_runs(tmpdir)
        assert len(runs) == 1
        assert runs[0]["id"] == "test_20260322_000000"
        assert "test problem" in runs[0]["problem"]


def test_list_previous_runs_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        runs = list_previous_runs(tmpdir)
        assert runs == []


# ═══ Smart Error Handling Tests ═══

from evohive.llm.provider import (
    _classify_error, _ErrorType, CircuitBreaker, get_provider_status,
)


def test_classify_error_rate_limit():
    class FakeError(Exception):
        status_code = 429
    assert _classify_error(FakeError()) == _ErrorType.RATE_LIMIT


def test_classify_error_auth():
    class FakeError(Exception):
        status_code = 401
    assert _classify_error(FakeError()) == _ErrorType.AUTH


def test_classify_error_server():
    class FakeError(Exception):
        status_code = 500
    assert _classify_error(FakeError()) == _ErrorType.SERVER


def test_classify_error_timeout():
    assert _classify_error(TimeoutError("connection timed out")) == _ErrorType.TIMEOUT


def test_classify_error_other():
    assert _classify_error(ValueError("something else")) == _ErrorType.OTHER


@pytest.mark.asyncio
async def test_circuit_breaker_basic():
    cb = CircuitBreaker("test_provider")
    assert cb.state == "closed"
    assert await cb.allow_request() is True

    # Record 3 failures to open the circuit
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == "open"
    assert await cb.allow_request() is False

    # Record success to close
    await cb.record_success()
    assert cb.state == "closed"


def test_get_provider_status():
    status = get_provider_status()
    assert isinstance(status, dict)


# ═══ Dialogue Truncation Tests ═══

from evohive.engine.dialogue import _truncate_history, MAX_DIALOGUE_TURNS


def test_truncate_history_empty():
    assert _truncate_history([]) == []


def test_truncate_history_under_limit():
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    result = _truncate_history(history, max_turns=10)
    assert len(result) == 4
    assert result[0]["content"] == "hello"


def test_truncate_history_over_limit():
    """When history exceeds max_turns, keep first pair + last (max_turns-1) pairs."""
    history = []
    for i in range(20):  # 10 pairs
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"msg-{i}"})
    result = _truncate_history(history, max_turns=3)
    # Should keep first pair + last 2 pairs = 3 pairs = 6 messages
    assert len(result) == 6
    assert result[0]["content"] == "msg-0"  # first pair preserved
    assert result[1]["content"] == "msg-1"


def test_truncate_history_content_truncation():
    """Messages longer than 2000 chars should be truncated."""
    history = [
        {"role": "user", "content": "x" * 5000},
        {"role": "assistant", "content": "y" * 3000},
    ]
    result = _truncate_history(history)
    assert len(result[0]["content"]) == 2000
    assert len(result[1]["content"]) == 2000


# ═══ Pairwise Judge Debias Tests ═══

from evohive.engine.pairwise_judge import pairwise_compare
from evohive.models import Solution


@pytest.mark.asyncio
async def test_pairwise_compare_debias():
    """Debias mode should call LLM twice (A-B and B-A)."""
    mock_response = json.dumps({"winner": "A", "reason": "better", "confidence": 0.8})
    sol_a = Solution(id="a", content="Solution Alpha", fitness=0.5)
    sol_b = Solution(id="b", content="Solution Beta", fitness=0.5)
    with patch("evohive.engine.pairwise_judge.call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await pairwise_compare(
            sol_a=sol_a,
            sol_b=sol_b,
            problem="test problem",
            model="test/model",
            dimensions_text="quality",
            debias=True,
        )
    assert "winner_id" in result
    assert "confidence" in result


@pytest.mark.asyncio
async def test_pairwise_compare_no_debias():
    """Without debias, should work with single comparison."""
    mock_response = json.dumps({"winner": "A", "reason": "better", "confidence": 0.8})
    sol_a = Solution(id="a", content="Solution Alpha", fitness=0.5)
    sol_b = Solution(id="b", content="Solution Beta", fitness=0.5)
    with patch("evohive.engine.pairwise_judge.call_llm", new_callable=AsyncMock, return_value=mock_response):
        result = await pairwise_compare(
            sol_a=sol_a,
            sol_b=sol_b,
            problem="test problem",
            model="test/model",
            dimensions_text="quality",
            debias=False,
        )
    assert "winner_id" in result


# ═══ Configurable Memory Window Tests ═══

from evohive.engine.elimination_memory import EvolutionMemory


def test_memory_window_default():
    mem = EvolutionMemory()
    assert mem.max_memory_generations == 2


def test_memory_window_custom():
    mem = EvolutionMemory(max_memory_generations=5)
    mem.add_failures(1, ["reason1"])
    mem.add_failures(2, ["reason2"])
    mem.add_failures(3, ["reason3"])
    # At generation 6, only gen 2 and 3 should be active (age <= 5: 6-1=5, 6-2=4, 6-3=3)
    active = mem.get_active_memories(6)
    assert "reason1" in active
    assert "reason2" in active
    assert "reason3" in active


def test_memory_window_expiry():
    mem = EvolutionMemory(max_memory_generations=1)
    mem.add_failures(1, ["old_reason"])
    mem.add_failures(3, ["new_reason"])
    # At generation 3, only gen 3 should be active (age 0 <= 1), gen 1 expired (age 2 > 1)
    active = mem.get_active_memories(3)
    assert "new_reason" in active
    assert "old_reason" not in active


def test_evolution_config_memory_window():
    from evohive.models import EvolutionConfig
    cfg = EvolutionConfig(problem="test", memory_window=5)
    assert cfg.memory_window == 5


def test_evolution_config_memory_window_default():
    from evohive.models import EvolutionConfig
    cfg = EvolutionConfig(problem="test")
    assert cfg.memory_window == 2


# ═══ Quality Comparison Field Tests ═══

def test_evolution_run_quality_comparison_field():
    from evohive.models import EvolutionRun
    run = EvolutionRun(
        id="test-run",
        config={"problem": "test"},
        started_at=datetime.now(),
        quality_comparison={"winner": "evolution", "confidence": 0.9},
    )
    assert run.quality_comparison["winner"] == "evolution"


# ═══ Pre-flight Model Check Tests ═══

from evohive.llm.provider import preflight_check


@pytest.mark.asyncio
async def test_preflight_check_all_pass():
    """All models reachable → all in 'ok', none in 'failed'."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "ok"

    with patch("evohive.llm.provider.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
        result = await preflight_check(["model-a", "model-b"])

    assert set(result["ok"]) == {"model-a", "model-b"}
    assert result["failed"] == []


@pytest.mark.asyncio
async def test_preflight_check_some_fail():
    """One model fails, one passes → mixed result."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "ok"

    async def _side_effect(**kwargs):
        if "bad-model" in kwargs.get("model", ""):
            raise RuntimeError("connection refused")
        return mock_resp

    with patch("evohive.llm.provider.litellm.acompletion", side_effect=_side_effect):
        result = await preflight_check(["good-model", "bad-model"])

    assert "good-model" in result["ok"]
    assert len(result["failed"]) == 1
    assert result["failed"][0]["model"] == "bad-model"
    assert "connection refused" in result["failed"][0]["error"]


@pytest.mark.asyncio
async def test_preflight_check_all_fail():
    """All models unreachable → none in 'ok', all in 'failed'."""
    async def _fail(**kwargs):
        raise RuntimeError("timeout")

    with patch("evohive.llm.provider.litellm.acompletion", side_effect=_fail):
        result = await preflight_check(["model-x", "model-y"])

    assert result["ok"] == []
    assert len(result["failed"]) == 2
    failed_models = {f["model"] for f in result["failed"]}
    assert failed_models == {"model-x", "model-y"}


@pytest.mark.asyncio
async def test_preflight_check_deduplicates():
    """Passing the same model twice should only probe once."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "ok"

    with patch("evohive.llm.provider.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_call:
        result = await preflight_check(["model-dup", "model-dup"])

    # Only one actual call despite two entries
    assert mock_call.call_count == 1
    assert result["ok"] == ["model-dup"]
    assert result["failed"] == []


# ═══ Adaptive Controller Tests ═══

from evohive.engine.adaptive import AdaptiveController
from evohive.models.evolution_run import GenerationStats


def test_adaptive_controller_initial_state():
    """Controller starts at base rates."""
    ctrl = AdaptiveController(base_mutation_rate=0.3, base_survival_rate=0.2)
    assert ctrl.current_mutation_rate == 0.3
    assert ctrl.current_survival_rate == 0.2
    summary = ctrl.summary()
    assert summary["total_adjustments"] == 0
    assert summary["history"] == []


def test_adaptive_controller_stagnation():
    """Flat fitness with high similarity should increase mutation_rate."""
    ctrl = AdaptiveController(base_mutation_rate=0.3, base_survival_rate=0.2)

    # First generation establishes a baseline
    stats1 = GenerationStats(
        generation=1, best_fitness=0.5, avg_fitness=0.4,
        worst_fitness=0.3, alive_count=10, eliminated_count=5,
    )
    ctrl.update(stats1, population_similarity=0.65)

    # Second generation: same fitness (stagnation) + high similarity
    stats2 = GenerationStats(
        generation=2, best_fitness=0.5, avg_fitness=0.4,
        worst_fitness=0.3, alive_count=10, eliminated_count=5,
    )
    result = ctrl.update(stats2, population_similarity=0.65)

    assert result["mutation_rate"] > 0.3
    assert "stagnation" in result["reason"]


def test_adaptive_controller_rapid_improvement():
    """Rising fitness should decrease mutation_rate."""
    ctrl = AdaptiveController(base_mutation_rate=0.4, base_survival_rate=0.3)

    stats1 = GenerationStats(
        generation=1, best_fitness=0.5, avg_fitness=0.4,
        worst_fitness=0.3, alive_count=10, eliminated_count=5,
    )
    ctrl.update(stats1, population_similarity=0.4)

    # Big jump in fitness (>5%)
    stats2 = GenerationStats(
        generation=2, best_fitness=0.6, avg_fitness=0.5,
        worst_fitness=0.4, alive_count=10, eliminated_count=5,
    )
    result = ctrl.update(stats2, population_similarity=0.4)

    assert result["mutation_rate"] < 0.4
    assert "rapid improvement" in result["reason"]


def test_adaptive_controller_bounds():
    """Rates should stay within min/max bounds regardless of input."""
    ctrl = AdaptiveController(
        base_mutation_rate=0.3,
        base_survival_rate=0.2,
        min_mutation_rate=0.1,
        max_mutation_rate=0.7,
        min_survival_rate=0.1,
        max_survival_rate=0.5,
        adjustment_speed=0.5,  # aggressive speed to test bounds
    )

    # Feed many stagnation signals to push rates up
    for i in range(1, 30):
        stats = GenerationStats(
            generation=i, best_fitness=0.5, avg_fitness=0.4,
            worst_fitness=0.3, alive_count=10, eliminated_count=5,
        )
        ctrl.update(stats, population_similarity=0.8)

    assert ctrl.current_mutation_rate <= 0.7
    assert ctrl.current_mutation_rate >= 0.1
    assert ctrl.current_survival_rate <= 0.5
    assert ctrl.current_survival_rate >= 0.1


def test_adaptive_config_default():
    """enable_adaptive should be True by default in EvolutionConfig."""
    from evohive.models import EvolutionConfig
    cfg = EvolutionConfig(problem="test")
    assert cfg.enable_adaptive is True


def test_adaptive_controller_summary():
    """Summary should track history after updates."""
    ctrl = AdaptiveController()
    stats = GenerationStats(
        generation=1, best_fitness=0.5, avg_fitness=0.4,
        worst_fitness=0.3, alive_count=10, eliminated_count=5,
    )
    ctrl.update(stats, population_similarity=0.4)
    summary = ctrl.summary()
    assert summary["total_adjustments"] == 1
    assert len(summary["history"]) == 1
    assert "generation" in summary["history"][0]
    assert "reason" in summary["history"][0]


# ═══ Checkpoint Tests ═══

from evohive.engine.checkpoint import (
    save_checkpoint, load_checkpoint, list_checkpoints, cleanup_checkpoints,
)
from evohive.engine.elimination_memory import EvolutionMemory


def _make_test_population(n: int = 3) -> list:
    """Create a small list of Solution objects for checkpoint tests."""
    return [
        Solution(id=f"sol-{i}", content=f"Solution {i} content", fitness=0.5 + i * 0.1, generation=1)
        for i in range(n)
    ]


def _make_test_memory() -> EvolutionMemory:
    mem = EvolutionMemory(max_memory_generations=2)
    mem.add_failures(1, ["too vague", "lacked detail"])
    return mem


def _make_test_config() -> EvolutionConfig:
    return EvolutionConfig(problem="test checkpoint problem", mode="fast", population_size=5)


def test_save_and_load_checkpoint():
    """Save a checkpoint, load it back, verify all fields match."""
    population = _make_test_population()
    memory = _make_test_memory()
    cfg = _make_test_config()
    extra = {
        "baseline_solution": "baseline text",
        "generations_data": [
            {"generation": 1, "best_fitness": 0.8, "avg_fitness": 0.5,
             "worst_fitness": 0.2, "alive_count": 3, "eliminated_count": 2}
        ],
        "total_api_calls": 42,
        "elimination_memories": ["too vague"],
        "search_context": "some context",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(
            run_id="test_run_001",
            generation=3,
            population=population,
            evolution_memory=memory,
            config=cfg,
            extra_state=extra,
            checkpoint_dir=tmpdir,
        )

        assert os.path.exists(path)
        assert path.endswith(".ckpt.json")

        # Verify it's valid JSON
        with open(path, "r") as f:
            raw = json.load(f)
        assert raw["run_id"] == "test_run_001"
        assert raw["generation"] == 3

        # Load via API
        loaded = load_checkpoint("test_run_001", checkpoint_dir=tmpdir)
        assert loaded is not None
        assert loaded["run_id"] == "test_run_001"
        assert loaded["generation"] == 3
        assert len(loaded["population"]) == 3
        assert loaded["population"][0]["id"] == "sol-0"
        assert loaded["evolution_memory"]["memories"][0]["reasons"] == ["too vague", "lacked detail"]
        assert loaded["config"]["problem"] == "test checkpoint problem"
        assert loaded["extra_state"]["baseline_solution"] == "baseline text"
        assert loaded["extra_state"]["total_api_calls"] == 42


def test_checkpoint_cleanup_old():
    """Save 5 checkpoints for the same run, verify only the latest 2 remain."""
    population = _make_test_population(1)
    memory = _make_test_memory()
    cfg = _make_test_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        for gen in range(1, 6):
            save_checkpoint(
                run_id="cleanup_run",
                generation=gen,
                population=population,
                evolution_memory=memory,
                config=cfg,
                checkpoint_dir=tmpdir,
            )

        # Only 2 should remain (the latest)
        remaining = [f for f in os.listdir(tmpdir) if f.endswith(".ckpt.json")]
        assert len(remaining) == 2

        # The two highest generations should survive
        gens = sorted(int(f.split("_gen")[1].split(".")[0]) for f in remaining)
        assert gens == [4, 5]


def test_checkpoint_atomic_write():
    """Verify that no .tmp file persists after save_checkpoint."""
    population = _make_test_population(1)
    memory = _make_test_memory()
    cfg = _make_test_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint(
            run_id="atomic_run",
            generation=1,
            population=population,
            evolution_memory=memory,
            config=cfg,
            checkpoint_dir=tmpdir,
        )

        all_files = os.listdir(tmpdir)
        tmp_files = [f for f in all_files if f.endswith(".tmp")]
        assert tmp_files == [], f"Temporary files should not persist: {tmp_files}"

        ckpt_files = [f for f in all_files if f.endswith(".ckpt.json")]
        assert len(ckpt_files) == 1


def test_load_checkpoint_no_file():
    """Loading a checkpoint for a nonexistent run returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_checkpoint("nonexistent_run", checkpoint_dir=tmpdir)
        assert result is None

    # Also test with a nonexistent directory
    result = load_checkpoint("nonexistent_run", checkpoint_dir="/tmp/no_such_dir_xyz_abc")
    assert result is None


def test_list_checkpoints():
    """Save checkpoints for multiple runs, list all of them."""
    population = _make_test_population(1)
    memory = _make_test_memory()
    cfg = _make_test_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_checkpoint("run_a", 1, population, memory, cfg, checkpoint_dir=tmpdir)
        save_checkpoint("run_a", 2, population, memory, cfg, checkpoint_dir=tmpdir)
        save_checkpoint("run_b", 1, population, memory, cfg, checkpoint_dir=tmpdir)

        checkpoints = list_checkpoints(checkpoint_dir=tmpdir)
        assert len(checkpoints) == 3

        run_ids = [c["run_id"] for c in checkpoints]
        assert "run_a" in run_ids
        assert "run_b" in run_ids

        generations = {(c["run_id"], c["generation"]) for c in checkpoints}
        assert ("run_a", 1) in generations
        assert ("run_a", 2) in generations
        assert ("run_b", 1) in generations

        # Each entry should have a timestamp
        for c in checkpoints:
            assert c["timestamp"] is not None
            assert c["filename"].endswith(".ckpt.json")


