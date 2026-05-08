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


from evohive.engine.effort import (
    apply_reasoning_effort,
    normalize_reasoning_effort,
    normalize_token_budget_control,
)


def test_reasoning_effort_presets_adjust_config():
    quick = apply_reasoning_effort(EvolutionConfig(
        problem="test",
        reasoning_effort="quick",
        population_size=50,
        generations=10,
    ))
    assert quick.mode == "fast"
    assert quick.population_size == 8
    assert quick.generations == 1
    assert quick.enable_red_team is False

    max_effort = apply_reasoning_effort(EvolutionConfig(
        problem="test",
        reasoning_effort="max",
        population_size=5,
        generations=1,
    ))
    assert max_effort.mode == "deep"
    assert max_effort.population_size == 20
    assert max_effort.generations == 3
    assert max_effort.enable_debate is True


def test_reasoning_effort_mode_backwards_compatibility():
    assert normalize_reasoning_effort(None, "fast") == "quick"
    assert normalize_reasoning_effort(None, "deep") == "max"
    assert normalize_reasoning_effort("balanced", "fast") == "balanced"


def test_token_budget_control_presets_are_user_switchable():
    off = apply_reasoning_effort(EvolutionConfig(
        problem="test",
        reasoning_effort="quick",
        token_budget_control="off",
    ))
    assert off.enable_token_budget_control is False

    strict = apply_reasoning_effort(EvolutionConfig(
        problem="test",
        reasoning_effort="quick",
        token_budget_control="strict",
    ))
    assert strict.enable_token_budget_control is True
    assert strict.token_budget_control == "strict"
    assert strict.token_budget_multiplier < 0.45

    custom = apply_reasoning_effort(EvolutionConfig(
        problem="test",
        reasoning_effort="quick",
        token_budget_control="auto",
        token_budget_multiplier=2.0,
    ))
    assert custom.enable_token_budget_control is True
    assert custom.token_budget_multiplier == 2.0
    assert normalize_token_budget_control("cheap") == "strict"


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
from evohive.engine.answer_graph import build_answer_graph
from evohive.engine.claim_verifier import (
    build_claim_search_verification_report,
    build_claim_verification_report,
)
from evohive.engine.verification import build_verification_report
from evohive.engine.token_budget import (
    assess_live_token_budget,
    build_token_budget_plan,
    build_token_budget_report,
)
from evohive.engine.trajectory_replay import build_trajectory_replay
from evohive.llm.provider import (
    call_llm,
    clear_session_cost_tracker,
    clear_session_token_budget,
    reset_session_cost_tracker,
    set_session_cost_tracker,
    set_session_token_budget,
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


def test_cost_tracker_snapshot_breakdown():
    tracker = CostTracker()
    tracker.record_call("openai/gpt-4o", 1000, 500, phase="generation")
    tracker.record_call("openai/gpt-4o", 120, 80, phase="judge")

    snapshot = tracker.snapshot()

    assert snapshot["total_calls"] == 2
    assert snapshot["providers"]["openai"]["calls"] == 2
    assert snapshot["providers"]["openai"]["input_tokens"] == 1120
    assert snapshot["phases"]["generation"]["output_tokens"] == 500
    assert snapshot["phases"]["judge"]["calls"] == 1


@pytest.mark.asyncio
async def test_llm_call_records_session_cost_tracker():
    tracker = CostTracker()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "tracked response"
    mock_resp.usage.prompt_tokens = 123
    mock_resp.usage.completion_tokens = 45

    tokens = set_session_cost_tracker(tracker, phase="unit")
    try:
        with patch("evohive.llm.provider.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = await call_llm("openai/gpt-4o", "system", "user", max_retries=0)
    finally:
        reset_session_cost_tracker(tokens)

    assert result == "tracked response"
    assert tracker.call_count == 1
    assert tracker.total_input_tokens == 123
    assert tracker.total_output_tokens == 45
    assert tracker.total_cost > 0


@pytest.mark.asyncio
async def test_llm_call_clips_max_tokens_against_session_token_budget():
    tracker = CostTracker()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "budgeted response"
    mock_resp.usage.prompt_tokens = 20
    mock_resp.usage.completion_tokens = 12

    cost_tokens = set_session_cost_tracker(tracker, phase="budgeted")
    budget_token = set_session_token_budget(
        {
            "total_token_budget": 200,
            "phase_budgets": {"budgeted": 80},
            "policy": {"hard_stop_at": 1.0},
        },
        enabled=True,
    )
    try:
        with patch("evohive.llm.provider.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mocked:
            result = await call_llm(
                "openai/gpt-4o",
                "system",
                "user prompt",
                max_tokens=500,
                max_retries=0,
            )
    finally:
        reset_session_cost_tracker(cost_tokens)
        clear_session_token_budget()

    assert result == "budgeted response"
    assert mocked.call_args.kwargs["max_tokens"] < 500
    assert mocked.call_args.kwargs["max_tokens"] >= 1


@pytest.mark.asyncio
async def test_clear_session_cost_tracker_stops_recording():
    tracker = CostTracker()
    set_session_cost_tracker(tracker, phase="unit")
    clear_session_cost_tracker()

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "untracked response"
    mock_resp.usage.prompt_tokens = 100
    mock_resp.usage.completion_tokens = 50

    with patch("evohive.llm.provider.litellm.acompletion", new_callable=AsyncMock, return_value=mock_resp):
        result = await call_llm("openai/gpt-4o", "system", "user", max_retries=0)

    assert result == "untracked response"
    assert tracker.call_count == 0


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


def test_token_budget_report_flags_over_budget_phase():
    plan = {
        "total_token_budget": 1000,
        "phase_budgets": {"elo_tournament": 100, "crossover": 500},
    }
    report = build_token_budget_report(
        plan=plan,
        resource_report={
            "total_tokens": 1200,
            "phases": {
                "elo_tournament": {"total_tokens": 140},
                "crossover": {"total_tokens": 200},
            },
        },
    )

    assert report["status"] == "over_budget"
    assert "elo_tournament" in report["over_budget_phases"]
    assert report["phase_reports"]["crossover"]["recommended_action"] == "ok"


def test_live_token_budget_assessment_requests_stop():
    assessment = assess_live_token_budget(
        plan={
            "total_token_budget": 1000,
            "phase_budgets": {"evaluation": 300},
            "policy": {"recommend_stop_at": 0.9, "hard_stop_at": 1.15},
        },
        cost_snapshot={
            "total_input_tokens": 900,
            "total_output_tokens": 300,
            "phases": {
                "evaluation": {"input_tokens": 500, "output_tokens": 100},
            },
        },
        checkpoint="generation_1_complete",
    )

    assert assessment["status"] == "hard_stop"
    assert assessment["should_stop"] is True
    assert "evaluation" in assessment["over_hard_phases"]


def test_token_budget_plan_respects_disabled_phases():
    cfg = EvolutionConfig(
        problem="test",
        reasoning_effort="quick",
        mode="fast",
        enable_swarm=False,
        enable_debate=False,
        enable_pressure_test=False,
    )
    plan = build_token_budget_plan(cfg)
    assert plan["total_token_budget"] > 0
    assert "swarm" not in plan["phase_budgets"]
    assert "debate" not in plan["phase_budgets"]


def test_verification_report_extracts_claims_and_risks():
    report = build_verification_report(
        problem="Launch plan",
        final_answer=(
            "1. We should test a narrow ICP before scaling.\n"
            "2. This will guarantee 100% conversion lift in 30 days."
        ),
        lineage_graph={"summary": {"node_count": 6, "finalist_count": 2}},
    )

    assert report["summary"]["claim_count"] == 2
    assert report["summary"]["high_risk_claim_count"] == 1
    assert report["claims"][0]["kind"] == "recommendation"
    assert "absolute_language" in report["claims"][1]["risk_flags"]
    assert "numeric_claim" in report["claims"][1]["risk_flags"]


def test_claim_verification_loop_uses_search_context_evidence():
    verification = build_verification_report(
        problem="Launch plan",
        final_answer="We should test one target user segment before scaling.",
        lineage_graph={"summary": {"node_count": 3, "finalist_count": 1}},
    )
    report = build_claim_verification_report(
        verification_report=verification,
        search_context="Target user segment testing before scaling reduces launch risk.",
    )

    assert report["version"] == "claim-verification-loop.v1"
    assert report["claim_count"] == 1
    assert report["claims"][0]["evidence"]
    assert report["summary"]["average_support_score"] > 0


@pytest.mark.asyncio
async def test_claim_search_verification_adds_active_search_evidence(monkeypatch):
    verification = build_verification_report(
        problem="Launch plan",
        final_answer="This will guarantee 100% conversion lift in 30 days.",
        lineage_graph={"summary": {"node_count": 3, "finalist_count": 1}},
    )
    base = build_claim_verification_report(verification_report=verification)

    async def fake_search(query, max_results=3):
        return [
            {
                "title": "Conversion lift test",
                "snippet": "Conversion lift in 30 days should be validated with experiments.",
                "url": "https://example.com/conversion",
            }
        ]

    monkeypatch.setattr("evohive.engine.web_search.web_search", fake_search)

    report = await build_claim_search_verification_report(
        verification_report=verification,
        base_report=base,
    )

    assert report["version"] == "claim-search-verification.v1"
    assert report["searched_claim_count"] == 1
    assert report["claims"][0]["evidence"][0]["type"] == "active_search"


def test_trajectory_replay_builds_timeline():
    replay = build_trajectory_replay([
        {"seq": 2, "phase": "evaluation", "actor": "judge", "action": "score", "elapsed_sec": 2.0},
        {"seq": 1, "phase": "baseline", "actor": "model", "action": "draft", "elapsed_sec": 1.0},
    ])

    assert replay["version"] == "trajectory-replay.v1"
    assert replay["step_count"] == 2
    assert replay["timeline"][0]["phase"] == "baseline"


def test_answer_graph_combines_lineage_claims_and_verifiers():
    verification = build_verification_report(
        problem="Launch plan",
        final_answer="We should test a narrow ICP. This will guarantee 100% lift in 30 days.",
        lineage_graph={"summary": {"node_count": 2, "finalist_count": 1}},
    )
    graph = build_answer_graph(
        problem="Launch plan",
        final_answer="We should test a narrow ICP. This will guarantee 100% lift in 30 days.",
        lineage_graph={
            "nodes": [
                {"id": "s1", "label": "seed", "generation": 0, "fitness": 0.4, "state": "active"},
                {"id": "s2", "label": "winner", "generation": 1, "fitness": 0.9, "state": "finalist"},
            ],
            "edges": [{"source": "s1", "target": "s2", "type": "parent_a", "generation": 1}],
            "summary": {"node_count": 2, "edge_count": 1, "finalist_count": 1},
        },
        verification_report=verification,
        top_solutions=[{"id": "s2", "content": "Winner content", "fitness": 0.9}],
    )

    assert graph["version"] == "answer-graph.v1"
    assert graph["summary"]["solution_node_count"] == 2
    assert graph["summary"]["claim_node_count"] == 2
    assert any(node["id"] == "problem" for node in graph["nodes"])
    assert any(node["type"] == "answer_quantum" for node in graph["nodes"])
    assert any(edge["type"] == "collapses_to" for edge in graph["edges"])
    assert any(edge["type"] == "contains_claim" for edge in graph["edges"])


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
        cost_breakdown={
            "total_calls": 2,
            "total_input_tokens": 1120,
            "total_output_tokens": 580,
            "providers": {
                "openai": {
                    "calls": 2,
                    "input_tokens": 1120,
                    "output_tokens": 580,
                    "cost": 0.01,
                }
            },
            "phases": {
                "generation": {
                    "calls": 1,
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cost": 0.008,
                }
            },
        },
        resource_report={
            "version": "resource-report.v1",
            "duration_sec": 1.5,
            "total_tokens": 1700,
            "tokens_per_sec": 1133.33,
            "generations": [{"generation": 1, "duration_sec": 0.7}],
        },
        token_budget_report={
            "version": "token-budget-report.v1",
            "status": "within_budget",
            "usage_ratio": 0.4,
        },
        trajectory_log=[
            {"seq": 1, "phase": "baseline", "actor": "tester", "action": "generate"},
        ],
        trajectory_summary={
            "event_count": 1,
            "phases": {"baseline": 1},
            "actors": {"tester": 1},
        },
        refined_top_solution="This is the refined solution.",
        lineage_graph={
            "summary": {
                "node_count": 3,
                "edge_count": 2,
                "finalist_count": 1,
                "mutated_count": 1,
                "eliminated_count": 1,
            },
            "nodes": [],
            "edges": [],
        },
        answer_graph={
            "summary": {
                "node_count": 5,
                "edge_count": 4,
                "solution_node_count": 2,
                "claim_node_count": 1,
                "verifier_node_count": 2,
            },
            "nodes": [],
            "edges": [],
        },
        verification_report={
            "summary": {
                "claim_count": 1,
                "high_risk_claim_count": 0,
                "average_confidence": 0.7,
            },
            "claims": [
                {
                    "id": "claim-01",
                    "kind": "factual",
                    "confidence": 0.7,
                    "text": "This is a persisted verification claim.",
                    "risk_flags": [],
                }
            ],
        },
        claim_verification_report={
            "version": "claim-verification-loop.v1",
            "claim_count": 1,
            "evidence_source": "search_context",
            "summary": {"needs_external_verifier": 0, "average_support_score": 0.7},
            "claims": [],
        },
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
    assert "Verification Report" in md
    assert "Lineage Graph" in md
    assert "Answer Graph" in md
    assert "Cost By Provider" in md
    assert "Runtime Seconds" in md
    assert "Trajectory Summary" in md
    assert "Claim Verification Loop" in md
    assert "This is a persisted verification claim." in md


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
