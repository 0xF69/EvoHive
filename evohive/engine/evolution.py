"""进化主循环 — 核心编排模块 (v3.0)

v3.0 新增:
- Swarm 层: 千级 Agent 轻量策略探索
- 语义 Embedding 相似度 (替代 Jaccard)
- 瑞士轮锦标赛 (替代全配对 Elo)
- 事件流架构 (供 CLI、SDK、日志和服务端集成消费)
- Web Search 真实数据注入
- 快速/深度双模式
- 实时成本追踪 + 预算控制
- 结果持久化 (JSON + Markdown)
"""

import asyncio
import random
import time
from datetime import datetime
from functools import wraps
from typing import Callable, Optional

from evohive.models import (
    EvolutionConfig, EvolutionRun, GenerationStats, Solution,
)
from evohive.engine.genesis import (
    generate_thinkers, generate_thinkers_multi_model, generate_initial_solutions,
)
from evohive.engine.judge import (
    evaluate_population, compute_population_similarity,
    compute_population_similarity_async,
)
from evohive.engine.selection import tournament_select
from evohive.engine.crossover import generate_next_generation
from evohive.engine.mutation import mutate_batch
from evohive.engine.baseline import generate_baseline
from evohive.engine.refine import deep_refine

# v2.0 imports
from evohive.engine.pairwise_judge import run_elo_tournament, apply_elo_to_solutions, pairwise_compare
from evohive.engine.elimination_memory import EvolutionMemory, extract_failure_reasons
from evohive.engine.diversity_guard import (
    kill_homogeneous, inject_fresh_blood, should_inject_fresh_blood,
)
from evohive.engine.red_team import red_team_batch, apply_red_team_scores
from evohive.engine.debate import debate_tournament, apply_debate_scores
from evohive.engine.pressure_test import pressure_test_batch, apply_pressure_scores
from evohive.engine.tool_refine import tool_augmented_refine

# checkpoint imports
from evohive.engine.checkpoint import save_checkpoint, load_checkpoint, cleanup_checkpoints

# v3.0 imports
from evohive.engine.events import EventEmitter, EvolutionEvent
from evohive.engine import events as evt
from evohive.engine.cost_tracker import CostTracker, BudgetExceededError, estimate_run_cost
from evohive.engine.adaptive import AdaptiveController
from evohive.engine.logger import get_logger, log_event
from evohive.llm.provider import (
    clear_session_cost_tracker,
    clear_session_token_budget,
    get_session_token_budget_events,
    preflight_check,
    reset_session_cost_tracker,
    set_session_cost_phase,
    set_session_cost_tracker,
    set_session_token_budget,
)
from evohive.engine.answer_graph import build_answer_graph
from evohive.engine.claim_verifier import (
    build_claim_search_verification_report,
    build_claim_verification_report,
)
from evohive.engine.effort import apply_reasoning_effort, effort_metadata
from evohive.engine.lineage import build_lineage_graph
from evohive.engine.token_budget import (
    assess_live_token_budget,
    build_token_budget_plan,
    build_token_budget_report,
)
from evohive.engine.trajectory import TrajectoryLog
from evohive.engine.trajectory_replay import build_trajectory_replay
from evohive.engine.verification import build_verification_report

_logger = get_logger("evohive.engine.evolution")


async def validate_models(
    config: EvolutionConfig,
    emitter: Optional[EventEmitter] = None,
    on_status: Optional[Callable] = None,
    on_confirm: Optional[Callable] = None,
) -> None:
    """Pre-flight check: validate all configured models before starting evolution.

    Collects every unique model from config, sends a minimal probe to each,
    and raises RuntimeError if ALL fail.  If only some fail, removes them
    from config and optionally asks the user to confirm via on_confirm callback.

    Args:
        on_confirm: Optional callback(ok_models, failed_info) -> bool.
                    If provided and returns False, raises RuntimeError to abort.
                    If not provided, continues automatically with working models.
    """
    all_models: list[str] = []
    all_models.append(config.thinker_model)
    all_models.append(config.judge_model)
    all_models.extend(config.thinker_models)
    all_models.extend(config.judge_models)
    all_models.extend(config.red_team_models)
    all_models.extend(config.swarm_models)

    # Deduplicate while preserving order
    unique_models = list(dict.fromkeys(m for m in all_models if m))
    if not unique_models:
        return

    result = await preflight_check(unique_models)

    def _log(msg: str):
        if on_status:
            on_status(msg)

    def _emit(event_type: str, phase: str, **data):
        if emitter:
            emitter.emit(event_type, phase, **data)

    if not result["ok"] and result["failed"]:
        # ALL models failed
        failure_lines = [
            f"  - {f['model']}: {f['error']}" for f in result["failed"]
        ]
        raise RuntimeError(
            "Pre-flight check failed: ALL configured models are unreachable.\n"
            + "\n".join(failure_lines)
        )

    if result["failed"]:
        # SOME models failed — remove them from config so they aren't used
        failed_models = {f["model"] for f in result["failed"]}

        # Ask user to confirm if callback provided
        if on_confirm is not None:
            should_continue = on_confirm(result["ok"], result["failed"])
            if not should_continue:
                raise RuntimeError("User aborted: declined to continue with partial models.")

        # Remove failed models from all config lists
        config.thinker_models = [m for m in config.thinker_models if m not in failed_models]
        config.judge_models = [m for m in config.judge_models if m not in failed_models]
        config.red_team_models = [m for m in config.red_team_models if m not in failed_models]
        config.swarm_models = [m for m in config.swarm_models if m not in failed_models]

        # Reassign thinker_model / judge_model if they failed
        if config.thinker_model in failed_models:
            if config.thinker_models:
                config.thinker_model = config.thinker_models[0]
            elif result["ok"]:
                config.thinker_model = result["ok"][0]
        if config.judge_model in failed_models:
            if config.judge_models:
                config.judge_model = config.judge_models[0]
            elif result["ok"]:
                config.judge_model = result["ok"][0]

        _log(
            f"已移除 {len(failed_models)} 个不可用模型，"
            f"使用剩余 {len(result['ok'])} 个模型继续"
        )
        log_event(_logger, "preflight_partial",
                  ok=result["ok"],
                  failed=[f["model"] for f in result["failed"]])
        _emit(evt.PREFLIGHT_PARTIAL, "init",
              ok=result["ok"],
              failed=result["failed"])
    else:
        # ALL models passed
        _log(f"Pre-flight check passed: {len(result['ok'])} model(s) reachable.")
        log_event(_logger, "preflight_ok", models=result["ok"])
        _emit(evt.PREFLIGHT_OK, "init", models=result["ok"])


def _build_dimensions_text(dimensions: list[dict]) -> str:
    """构建评审维度文本"""
    return ", ".join(d["name"] for d in dimensions)


async def run_evolution(
    config: EvolutionConfig,
    on_generation_complete: Optional[Callable] = None,
    on_status: Optional[Callable] = None,
    emitter: Optional[EventEmitter] = None,
    budget_limit: Optional[float] = None,
    save_results: bool = False,
    output_dir: str = "evohive_results",
    resume_from: Optional[str] = None,
    checkpoint_dir: str = "evohive_checkpoints",
    on_preflight_confirm: Optional[Callable] = None,
) -> EvolutionRun:
    """运行完整的进化流程 (v3.0)

    Args:
        config: 进化配置
        on_generation_complete: 每代完成后的回调 (generation, stats, best_solution)
        on_status: 状态更新回调 (message)
        emitter: 事件发射器 (用于 CLI、SDK、日志或服务端集成)
        budget_limit: 预算上限 (USD), None 表示不限制
        save_results: 是否自动保存结果到文件
        output_dir: 结果保存目录
        resume_from: run_id to resume from a previous checkpoint (None = fresh start)
        checkpoint_dir: directory for checkpoint files
    """
    config = apply_reasoning_effort(config)
    effort_profile = effort_metadata(config)

    def _log(msg: str):
        if on_status:
            on_status(msg)

    def _emit(event_type: str, phase: str, **data):
        if emitter:
            emitter.emit(event_type, phase, **data)

    # ── Pre-flight model availability check ──
    await validate_models(config, emitter=emitter, on_status=on_status, on_confirm=on_preflight_confirm)

    # Register all working models as fallbacks for automatic degradation
    from evohive.llm.provider import set_fallback_models
    all_working = []
    all_working.append(config.thinker_model)
    all_working.append(config.judge_model)
    all_working.extend(config.thinker_models)
    all_working.extend(config.judge_models)
    all_working.extend(config.red_team_models)
    all_working = list(dict.fromkeys(m for m in all_working if m))
    set_fallback_models(all_working)

    # 初始化成本追踪器
    cost_tracker = CostTracker(budget_limit=budget_limit)
    cost_tracker_tokens = set_session_cost_tracker(cost_tracker, phase="evolution")
    token_budget_plan = build_token_budget_plan(config)
    if config.enable_token_budget_control:
        min_output_tokens = {
            "quick": 384,
            "balanced": 512,
            "max": 768,
        }.get(config.reasoning_effort or "balanced", 192)
        set_session_token_budget(
            token_budget_plan,
            enabled=True,
            min_output_tokens=min_output_tokens,
        )
    resource_started_at = time.perf_counter()
    phase_started_at = resource_started_at
    current_resource_phase = "init"
    phase_timings: dict[str, dict] = {}
    generation_resources: list[dict] = []
    token_budget_events: list[dict] = []
    skip_optional_due_to_budget = False
    early_stop_reason = None
    trajectory = TrajectoryLog()
    trajectory.record(
        phase="init",
        actor="effort_controller",
        action="apply_reasoning_effort",
        input_summary=f"requested={config.reasoning_effort}",
        output_summary=f"mode={config.mode}, population={config.population_size}, generations={config.generations}",
        metrics=effort_profile,
    )

    def _record_phase_elapsed(now: float) -> None:
        nonlocal phase_started_at
        elapsed = max(0.0, now - phase_started_at)
        phase_stat = phase_timings.setdefault(
            current_resource_phase,
            {"duration_sec": 0.0, "entries": 0},
        )
        phase_stat["duration_sec"] += elapsed
        phase_stat["entries"] += 1
        phase_started_at = now

    def _cost_phase(phase: str) -> None:
        nonlocal current_resource_phase
        now = time.perf_counter()
        _record_phase_elapsed(now)
        current_resource_phase = phase
        set_session_cost_phase(phase)

    def _record_generation_resource(stats: GenerationStats, started_at: float) -> None:
        generation_resources.append({
            "generation": stats.generation,
            "duration_sec": round(max(0.0, time.perf_counter() - started_at), 4),
            "best_fitness": round(stats.best_fitness, 6),
            "avg_fitness": round(stats.avg_fitness, 6),
            "alive_count": stats.alive_count,
            "eliminated_count": stats.eliminated_count,
        })

    def _build_resource_report(cost_snapshot: dict) -> dict:
        nonlocal current_resource_phase
        now = time.perf_counter()
        _record_phase_elapsed(now)
        current_resource_phase = "complete"
        total_duration = max(0.0, now - resource_started_at)
        phases = {}
        for phase, timing in phase_timings.items():
            cost_data = (cost_snapshot.get("phases") or {}).get(phase, {})
            duration = timing.get("duration_sec", 0.0)
            total_tokens = cost_data.get("input_tokens", 0) + cost_data.get("output_tokens", 0)
            phases[phase] = {
                "duration_sec": round(duration, 4),
                "calls": cost_data.get("calls", 0),
                "input_tokens": cost_data.get("input_tokens", 0),
                "output_tokens": cost_data.get("output_tokens", 0),
                "total_tokens": total_tokens,
                "cost": cost_data.get("cost", 0.0),
                "tokens_per_sec": round(total_tokens / duration, 2) if duration > 0 else 0.0,
            }
        total_tokens = cost_snapshot.get("total_input_tokens", 0) + cost_snapshot.get("total_output_tokens", 0)
        return {
            "version": "resource-report.v1",
            "duration_sec": round(total_duration, 4),
            "llm_calls": cost_snapshot.get("total_calls", 0),
            "input_tokens": cost_snapshot.get("total_input_tokens", 0),
            "output_tokens": cost_snapshot.get("total_output_tokens", 0),
            "total_tokens": total_tokens,
            "estimated_cost": cost_snapshot.get("total_cost", 0.0),
            "tokens_per_sec": round(total_tokens / total_duration, 2) if total_duration > 0 else 0.0,
            "cost_per_1k_tokens": round(cost_snapshot.get("total_cost", 0.0) / (total_tokens / 1000), 6)
            if total_tokens > 0 else 0.0,
            "phases": phases,
            "generations": generation_resources,
        }

    def _check_token_budget(checkpoint: str) -> dict:
        nonlocal skip_optional_due_to_budget
        if not config.enable_token_budget_control:
            return {}
        assessment = assess_live_token_budget(
            plan=token_budget_plan,
            cost_snapshot=cost_tracker.snapshot(),
            checkpoint=checkpoint,
        )
        if assessment["status"] != "within_budget":
            token_budget_events.append(assessment)
            skip_optional_due_to_budget = skip_optional_due_to_budget or assessment["should_skip_optional"]
            trajectory.record(
                phase="token_budget",
                actor="token_budget_controller",
                action="assess_live_budget",
                input_summary=checkpoint,
                output_summary=f"status={assessment['status']}, ratio={assessment['usage_ratio']}",
                metrics={
                    "should_stop": assessment["should_stop"],
                    "should_skip_optional": assessment["should_skip_optional"],
                    "over_hard_phases": assessment["over_hard_phases"],
                },
            )
        return assessment

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = EvolutionRun(
        id=run_id,
        config=config,
        started_at=datetime.now(),
        mode=config.mode,
        reasoning_effort=config.reasoning_effort,
        effort_profile=effort_profile,
        budget_limit=budget_limit,
        token_budget_plan=token_budget_plan,
    )

    # 运行前成本预估
    cost_estimate = estimate_run_cost(config, len(config.thinker_models or [config.thinker_model]))
    _log(f"成本预估: ${cost_estimate['estimated_min']:.2f} ~ ${cost_estimate['estimated_max']:.2f}")
    if budget_limit:
        _log(f"预算上限: ${budget_limit:.2f}")
        if cost_estimate['estimated_min'] > budget_limit:
            _log(f"  ⚠ 预估最低成本已超出预算!")

    _emit(evt.RUN_STARTED, "init", mode=config.mode, problem=config.problem)

    log_event(_logger, "run_start",
              run_id=run_id,
              mode=config.mode,
              generations=config.generations,
              population_size=config.population_size,
              budget_limit=budget_limit,
              thinker_model=config.thinker_model,
              judge_model=config.judge_model,
              enable_swarm=config.enable_swarm)

    # ── Resume from checkpoint (if requested) ──
    _resume_gen = 0  # generation to resume from (0 = fresh start)
    _resumed_population = None
    _resumed_memory_data = None
    _resumed_extra = None

    if resume_from:
        ckpt = load_checkpoint(resume_from, checkpoint_dir=checkpoint_dir)
        if ckpt is not None:
            _resume_gen = ckpt["generation"]
            _resumed_population = ckpt["population"]
            _resumed_memory_data = ckpt.get("evolution_memory", {})
            _resumed_extra = ckpt.get("extra_state", {})
            _log(f"Resuming run '{resume_from}' from generation {_resume_gen}")
            log_event(_logger, "resume_from_checkpoint",
                      run_id=resume_from,
                      generation=_resume_gen)
        else:
            _log(f"No checkpoint found for run '{resume_from}', starting fresh.")


    # 确定模型列表
    judge_models = config.judge_models if config.judge_models else [config.judge_model]
    red_team_models = config.red_team_models if config.red_team_models else [config.judge_model]
    thinker_models = config.thinker_models if config.thinker_models else [config.thinker_model]
    swarm_models = config.swarm_models if config.swarm_models else thinker_models
    dimensions_text = _build_dimensions_text(config.judge_dimensions)

    # 初始化遗传记忆
    evolution_memory = EvolutionMemory(max_memory_generations=config.memory_window)

    # 初始化自适应参数控制器
    adaptive_controller = AdaptiveController(
        base_mutation_rate=config.mutation_rate,
        base_survival_rate=config.survival_rate,
    ) if config.enable_adaptive else None

    # ── Restore state from checkpoint if resuming ──
    if _resume_gen > 0 and _resumed_population is not None:
        # Restore population from checkpoint
        population = [Solution(**sol_data) for sol_data in _resumed_population]

        # Restore evolution memory
        if _resumed_memory_data:
            evolution_memory.memories = _resumed_memory_data.get("memories", [])
            evolution_memory.max_memory_generations = _resumed_memory_data.get(
                "max_memory_generations", config.memory_window
            )

        # Restore extra state
        if _resumed_extra:
            run.baseline_solution = _resumed_extra.get("baseline_solution", "")
            # Restore generations_data
            for gd in _resumed_extra.get("generations_data", []):
                run.generations_data.append(GenerationStats(**gd))
            # Restore total_api_calls
            run.total_api_calls = _resumed_extra.get("total_api_calls", 0)
            # Restore elimination_memories
            run.elimination_memories = _resumed_extra.get("elimination_memories", [])

        all_solutions = list(population)
        search_context = _resumed_extra.get("search_context", "") if _resumed_extra else ""
        if search_context:
            run.search_context = search_context

        _log(f"Restored {len(population)} solutions from checkpoint (generation {_resume_gen})")

    else:
        # ── Normal (non-resume) initialization path ──
        search_context = ""
        if config.enable_web_search:
            _cost_phase("web_search")
            _log("搜索问题相关真实数据...")
            try:
                from evohive.engine.web_search import search_context_for_problem
                search_context = await search_context_for_problem(config.problem)
                if search_context:
                    _log(f"  获取到搜索上下文 ({len(search_context)} 字符)")
                    run.search_context = search_context
                else:
                    _log("  未配置搜索API或无结果，跳过")
            except Exception as e:
                _log(f"  搜索失败: {e}")

        # ── 1. 生成 Baseline ──
        _cost_phase("baseline")
        _log("生成Baseline对照...")
        run.baseline_solution = await generate_baseline(
            config.problem, config.thinker_model
        )
        run.total_api_calls += 1
        trajectory.record(
            phase="baseline",
            actor=config.thinker_model,
            action="generate_baseline",
            input_summary=config.problem[:160],
            output_summary=run.baseline_solution[:220],
        )

        # ── 2. 种群初始化 (Swarm 或传统模式) ──
        if config.enable_swarm:
            # ══ Swarm 模式 ══
            _cost_phase("swarm")
            _emit(evt.SWARM_STARTED, "swarm", total_seeds=config.swarm_count)

            from evohive.engine.swarm import run_swarm_phase

            # 快速模式用更少的种子
            swarm_count = config.swarm_count
            max_reps = config.swarm_max_representatives

            if config.mode == "fast":
                swarm_count = min(200, swarm_count)
                max_reps = min(config.population_size, max_reps)

            _log(f"Swarm 阶段: {swarm_count} 个种子 → {max_reps} 个代表...")
            population, swarm_stats = await run_swarm_phase(
                problem=config.problem,
                models=swarm_models,
                total_seeds=swarm_count,
                max_representatives=max_reps,
                embedding_model=config.embedding_model,
                search_context=search_context,
                on_progress=_log,
            )
            run.swarm_stats = swarm_stats
            run.total_api_calls += swarm_stats.get("total_seeds", 0) + swarm_stats.get("n_representatives", 0)
            trajectory.record(
                phase="swarm",
                actor="swarm",
                action="generate_representatives",
                input_summary=f"seeds={swarm_count}, max_representatives={max_reps}",
                output_summary=f"solutions={len(population)}",
                metrics=swarm_stats,
            )

            # 如果 swarm 产出不够, 用传统方式补充
            if len(population) < config.population_size:
                n_extra = config.population_size - len(population)
                _log(f"Swarm 产出不足，补充生成 {n_extra} 个方案...")
                _cost_phase("initialization")
                extra_thinkers = await generate_thinkers(config.problem, n_extra, thinker_models[0])
                extra_solutions = await generate_initial_solutions(config.problem, extra_thinkers)
                population.extend(extra_solutions)
                run.total_api_calls += n_extra + 1

            # 截断到目标种群大小
            if len(population) > config.population_size:
                population = population[:config.population_size]

            _emit(evt.SWARM_COMPLETE, "swarm",
                  n_solutions=len(population),
                  n_clusters=swarm_stats.get("n_clusters", 0))

        else:
            # ══ 传统 Thinker 模式 ══
            _cost_phase("initialization")
            if len(thinker_models) > 1:
                from collections import Counter
                model_counts = Counter(thinker_models)
                model_assignments = list(model_counts.items())
                total_assigned = sum(c for _, c in model_assignments)
                if total_assigned != config.population_size:
                    model_assignments = [
                        (m, max(1, round(c / total_assigned * config.population_size)))
                        for m, c in model_assignments
                    ]
                model_names = [m for m, _ in model_assignments]
                _log(f"生成 {config.population_size} 个Thinker角色 (模型: {', '.join(model_names)})...")
                thinkers = await generate_thinkers_multi_model(
                    config.problem, model_assignments,
                )
            else:
                _log(f"生成 {config.population_size} 个Thinker角色...")
                thinkers = await generate_thinkers(
                    config.problem, config.population_size, thinker_models[0]
                )
            run.total_api_calls += len(set(thinker_models))

            _log("各Thinker独立生成方案...")
            population = await generate_initial_solutions(config.problem, thinkers)
            run.total_api_calls += len(population)
            trajectory.record(
                phase="initialization",
                actor="thinker_pool",
                action="generate_initial_solutions",
                input_summary=f"thinkers={len(thinkers)}",
                output_summary=f"solutions={len(population)}",
                metrics={"models": thinker_models},
            )

        all_solutions = list(population)

    _emit(evt.EVOLUTION_STARTED, "evolution",
          population_size=len(population),
          generations=config.generations)

    # ── 3. 进化循环 ──
    generations = config.generations

    for gen in range(generations):
        # Skip already-completed generations when resuming
        if gen + 1 <= _resume_gen:
            continue

        generation_started_at = time.perf_counter()
        _log(f"\n═══ 第 {gen+1}/{generations} 代 ═══")
        _emit(evt.GENERATION_STARTED, "evolution", generation=gen+1)
        log_event(_logger, "generation_start",
                  generation=gen+1, total=generations,
                  population_size=len(population))

        # ── a. 绝对评审 ──
        _cost_phase("evaluation")
        _log("评审中...")
        _emit(evt.EVALUATION_STARTED, "evolution", generation=gen+1)

        def _eval_progress(done, total, sol_id):
            _emit("evaluation_progress", "evolution",
                  generation=gen+1, done=done, total=total, solution_id=sol_id)

        judgments = await evaluate_population(
            population,
            config.judge_dimensions,
            judge_models[0],
            rounds=config.judge_rounds,
            diversity_weight=config.diversity_weight,
            on_progress=_eval_progress,
        )
        run.total_api_calls += len(population) * config.judge_rounds
        trajectory.record(
            phase="evaluation",
            actor=judge_models[0],
            action="evaluate_population",
            input_summary=f"solutions={len(population)}, rounds={config.judge_rounds}",
            output_summary=f"judgments={len(judgments)}",
            metrics={"generation": gen + 1},
        )
        _emit(evt.EVALUATION_COMPLETE, "evolution", generation=gen+1)
        budget_assessment = _check_token_budget(f"generation_{gen+1}_evaluation_complete")
        if budget_assessment.get("should_stop"):
            early_stop_reason = (
                f"token budget hard stop after evaluation in generation {gen+1}: "
                f"usage ratio {budget_assessment['usage_ratio']}"
            )
            _log(f"⚠ {early_stop_reason}")
            break

        # ── a2. Executable fitness verification (if enabled) ──
        if config.enable_executable_fitness and config.test_cases and not skip_optional_due_to_budget:
            from evohive.engine.executable_fitness import evaluate_executable_fitness
            exec_reports = await evaluate_executable_fitness(
                population,
                config.test_cases,
                exec_weight=config.exec_weight,
                timeout_per_case=config.exec_timeout,
            )
            run.execution_results.append({
                "generation": gen + 1,
                "reports": exec_reports,
            })
            if on_status:
                n_executed = sum(1 for r in exec_reports if r.get("exec_score") is not None)
                n_passed = sum(1 for r in exec_reports if r.get("exec_score", 0) == 1.0)
                on_status(f"  ⚡ Executable fitness: {n_executed} solutions tested, {n_passed} passed all cases")

        # ── b. Elo 评审 (瑞士轮 or 全配对) ──
        if config.enable_pairwise_judge and len(judge_models) >= 1 and not skip_optional_due_to_budget:
            _cost_phase("elo_tournament")
            use_swiss = config.enable_swiss_tournament or len(population) > 10
            method = "瑞士轮" if use_swiss else "全配对"
            _log(f"Elo评审 ({method}, {len(judge_models)}个Judge模型)...")
            _emit(evt.ELO_TOURNAMENT_STARTED, "evolution",
                  generation=gen+1, method=method)

            if use_swiss:
                from evohive.engine.swiss_tournament import run_swiss_tournament

                def _elo_progress(round_num, total_rounds, n_pairs, top5_ratings):
                    _emit("elo_round_complete", "evolution",
                          generation=gen+1, round=round_num, total_rounds=total_rounds,
                          n_pairs=n_pairs, top5=top5_ratings)

                elo_ratings = await run_swiss_tournament(
                    population, config.problem, judge_models, dimensions_text,
                    on_progress=_elo_progress,
                )
                # Swiss tournament uses ~N*log2(N)/2 comparisons
                import math
                n_comparisons = len(population) * max(3, math.ceil(math.log2(len(population)))) // 2
            else:
                elo_ratings = await run_elo_tournament(
                    population, config.problem, judge_models, dimensions_text,
                )
                n_comparisons = len(population) * (len(population) - 1) // 2 * len(judge_models)

            apply_elo_to_solutions(population, elo_ratings)
            run.total_api_calls += n_comparisons
            trajectory.record(
                phase="elo_tournament",
                actor="judge_pool",
                action="rank_population",
                input_summary=f"solutions={len(population)}, method={method}",
                output_summary=f"comparisons={n_comparisons}",
                metrics={"generation": gen + 1, "judges": judge_models},
            )

            _emit(evt.ELO_TOURNAMENT_COMPLETE, "evolution",
                  generation=gen+1, n_comparisons=n_comparisons)
            budget_assessment = _check_token_budget(f"generation_{gen+1}_elo_complete")
            if budget_assessment.get("should_stop"):
                early_stop_reason = (
                    f"token budget hard stop after Elo in generation {gen+1}: "
                    f"usage ratio {budget_assessment['usage_ratio']}"
                )
                _log(f"⚠ {early_stop_reason}")
                break

        # ── c. 趋同检测 (embedding-based) ──
        try:
            _cost_phase("similarity")
            similarity = await compute_population_similarity_async(
                population, config.embedding_model
            )
        except Exception:
            similarity = compute_population_similarity(population)

        if similarity > config.convergence_threshold and gen > 0:
            early_stop_reason = (
                f"种群趋同度 {similarity:.2f} 超过阈值 {config.convergence_threshold}，"
                f"第{gen+1}代提前终止。"
            )
            _log(f"⚠ {early_stop_reason}")
            _emit(evt.CONVERGENCE_DETECTED, "evolution",
                  generation=gen+1, similarity=similarity)
            log_event(_logger, "early_stop",
                      generation=gen+1,
                      reason="convergence",
                      similarity=round(similarity, 4),
                      threshold=config.convergence_threshold)

            fitnesses = [s.fitness for s in population if s.fitness > 0]
            if not fitnesses:
                fitnesses = [0.0]
            stats = GenerationStats(
                generation=gen + 1,
                best_fitness=max(fitnesses),
                avg_fitness=sum(fitnesses) / len(fitnesses),
                worst_fitness=min(fitnesses),
                alive_count=len(population),
                eliminated_count=0,
            )
            run.generations_data.append(stats)
            _record_generation_resource(stats, generation_started_at)
            if on_generation_complete:
                best = max(population, key=lambda s: s.fitness)
                on_generation_complete(gen + 1, stats, best)
            break

        # ── d. 选择 ──
        _current_survival_rate = (
            adaptive_controller.current_survival_rate if adaptive_controller else config.survival_rate
        )
        _log("锦标赛选择...")
        survivors, eliminated = tournament_select(
            population,
            _current_survival_rate,
            config.elite_rate,
            config.tournament_size,
        )
        _emit(evt.SELECTION_COMPLETE, "evolution",
              generation=gen+1,
              survivors=len(survivors),
              eliminated=len(eliminated))
        trajectory.record(
            phase="selection",
            actor="selection_engine",
            action="tournament_select",
            input_summary=f"population={len(population)}",
            output_summary=f"survivors={len(survivors)}, eliminated={len(eliminated)}",
            metrics={"generation": gen + 1},
        )
        log_event(_logger, "elimination",
                  generation=gen+1,
                  survivors=len(survivors),
                  eliminated=len(eliminated))

        # ── e. 淘汰反馈遗传记忆 ──
        if config.enable_elimination_memory and eliminated and not skip_optional_due_to_budget:
            _cost_phase("elimination_memory")
            _log(f"提取 {len(eliminated)} 个淘汰方案的失败原因...")
            failure_reasons = await extract_failure_reasons(
                eliminated, config.problem, config.judge_model,
            )
            evolution_memory.add_failures(gen + 1, failure_reasons)
            run.total_api_calls += min(5, len(eliminated))
            run.elimination_memories.extend(failure_reasons)
            trajectory.record(
                phase="elimination_memory",
                actor=config.judge_model,
                action="extract_failure_reasons",
                input_summary=f"eliminated={len(eliminated)}",
                output_summary=f"reasons={len(failure_reasons)}",
                metrics={"generation": gen + 1},
            )
            _log(f"  记录 {len(failure_reasons)} 条失败教训")

        # ── f. 交叉重组 ──
        memory_text = evolution_memory.format_for_prompt(gen + 1)
        crossover_model = random.choice(thinker_models)
        _cost_phase("crossover")
        budget_assessment = _check_token_budget(f"generation_{gen+1}_crossover_start")
        if budget_assessment.get("should_stop"):
            early_stop_reason = (
                f"token budget hard stop before crossover in generation {gen+1}: "
                f"usage ratio {budget_assessment['usage_ratio']}"
            )
            _log(f"⚠ {early_stop_reason}")
            break
        _log("交叉重组...")
        _emit(evt.CROSSOVER_STARTED, "evolution", generation=gen+1)
        next_gen = await generate_next_generation(
            survivors,
            config.population_size,
            config.problem,
            crossover_model,
            generation=gen + 1,
            memory_injection=memory_text if config.enable_elimination_memory else "",
        )
        n_children = len(next_gen) - len(survivors)
        run.total_api_calls += n_children * 3
        trajectory.record(
            phase="crossover",
            actor=crossover_model,
            action="generate_next_generation",
            input_summary=f"survivors={len(survivors)}, target={config.population_size}",
            output_summary=f"children={n_children}",
            metrics={"generation": gen + 1},
        )
        _emit(evt.CROSSOVER_COMPLETE, "evolution",
              generation=gen+1, n_children=n_children)

        # ── g. 变异 ──
        _current_mutation_rate = (
            adaptive_controller.current_mutation_rate if adaptive_controller else config.mutation_rate
        )
        mutation_model = random.choice(thinker_models)
        _cost_phase("mutation")
        _log("变异...")
        children_only = next_gen[len(survivors):]
        mutated_children = await mutate_batch(
            children_only,
            _current_mutation_rate,
            config.problem,
            mutation_model,
        )
        n_mutated = sum(1 for c in mutated_children if c.mutation_applied)
        run.total_api_calls += n_mutated
        trajectory.record(
            phase="mutation",
            actor=mutation_model,
            action="mutate_children",
            input_summary=f"children={len(children_only)}, mutation_rate={_current_mutation_rate:.3f}",
            output_summary=f"mutated={n_mutated}",
            metrics={"generation": gen + 1},
        )
        _emit(evt.MUTATION_COMPLETE, "evolution",
              generation=gen+1, n_mutated=n_mutated)

        # ── h. 反同质化猎杀 ──
        if config.enable_diversity_guard:
            elite_id = survivors[0].id if survivors else None
            alive_children, killed = kill_homogeneous(
                mutated_children,
                survivors,
                threshold=config.homogeneity_threshold,
                elite_id=elite_id,
            )
            if killed:
                _log(f"  反同质化猎杀: {len(killed)} 个方案被杀")
                _emit(evt.HOMOGENEITY_CULLED, "evolution",
                      generation=gen+1, n_killed=len(killed))
        else:
            alive_children = list(mutated_children)
            killed = []

        # ── i. 新血注入 ──
        fresh = []
        if config.enable_fresh_blood and not skip_optional_due_to_budget:
            n_killed = len(killed)
            need_inject = should_inject_fresh_blood(
                survivors, gen + 1,
                inject_interval=config.fresh_blood_interval,
                similarity_threshold=0.5,
            )
            n_inject = n_killed if n_killed > 0 else (2 if need_inject else 0)
            if n_inject > 0:
                fresh_model = random.choice(thinker_models)
                _cost_phase("fresh_blood")
                _log(f"  新血注入: 生成 {n_inject} 个全新方案 (模型: {fresh_model})...")
                fresh = await inject_fresh_blood(
                    config.problem,
                    fresh_model,
                    n_inject,
                    gen + 1,
                    survivors + alive_children,
                )
                run.total_api_calls += n_inject + 1
                trajectory.record(
                    phase="fresh_blood",
                    actor=fresh_model,
                    action="inject_fresh_blood",
                    input_summary=f"n_inject={n_inject}",
                    output_summary=f"fresh={len(fresh)}",
                    metrics={"generation": gen + 1},
                )
                _emit(evt.FRESH_BLOOD_INJECTED, "evolution",
                      generation=gen+1, n_injected=len(fresh))

        # ── j. 组装新种群 ──
        population = list(survivors) + list(alive_children) + list(fresh)
        all_solutions.extend([s for s in population if s not in all_solutions])

        # ── k. 统计 ──
        fitnesses = [s.fitness for s in population if s.fitness > 0]
        if not fitnesses:
            fitnesses = [0.0]

        stats = GenerationStats(
            generation=gen + 1,
            best_fitness=max(fitnesses),
            avg_fitness=sum(fitnesses) / len(fitnesses),
            worst_fitness=min(fitnesses),
            alive_count=len(survivors),
            eliminated_count=len(eliminated),
        )
        run.generations_data.append(stats)
        _record_generation_resource(stats, generation_started_at)
        budget_assessment = _check_token_budget(f"generation_{gen+1}_complete")
        if budget_assessment.get("should_stop"):
            early_stop_reason = (
                f"token budget hard stop at generation {gen+1}: "
                f"usage ratio {budget_assessment['usage_ratio']}"
            )
            _log(f"⚠ {early_stop_reason}")
            break

        # ── l. 自适应参数调整 ──
        if adaptive_controller:
            adapt_result = adaptive_controller.update(stats, similarity)
            run.adaptive_history.append({
                "generation": gen + 1,
                "mutation_rate": round(adapt_result["mutation_rate"], 4),
                "survival_rate": round(adapt_result["survival_rate"], 4),
                "reason": adapt_result["reason"],
            })
            _log(
                f"  自适应调整: mutation_rate={adapt_result['mutation_rate']:.3f}, "
                f"survival_rate={adapt_result['survival_rate']:.3f} ({adapt_result['reason']})"
            )
            log_event(_logger, "adaptive_adjustment",
                      generation=gen+1,
                      mutation_rate=round(adapt_result["mutation_rate"], 4),
                      survival_rate=round(adapt_result["survival_rate"], 4),
                      reason=adapt_result["reason"])

        _emit(evt.GENERATION_COMPLETE, "evolution",
              generation=gen+1,
              best_fitness=stats.best_fitness,
              avg_fitness=stats.avg_fitness)
        log_event(_logger, "generation_end",
                  generation=gen+1,
                  best_fitness=round(stats.best_fitness, 4),
                  avg_fitness=round(stats.avg_fitness, 4),
                  worst_fitness=round(stats.worst_fitness, 4),
                  alive=stats.alive_count,
                  eliminated=stats.eliminated_count)

        if on_generation_complete:
            best = max(population, key=lambda s: s.fitness)
            on_generation_complete(gen + 1, stats, best)

        # ── Checkpoint: save state after each generation ──
        try:
            save_checkpoint(
                run_id=run_id,
                generation=gen + 1,
                population=population,
                evolution_memory=evolution_memory,
                config=config,
                extra_state={
                    "baseline_solution": run.baseline_solution,
                    "generations_data": [g.model_dump() for g in run.generations_data],
                    "total_api_calls": run.total_api_calls,
                    "elimination_memories": run.elimination_memories,
                    "search_context": getattr(run, "search_context", ""),
                },
                checkpoint_dir=checkpoint_dir,
            )
        except Exception as _ckpt_err:
            _log(f"Checkpoint save failed: {_ckpt_err}")

    # ── Clean up checkpoints on successful completion ──
    try:
        cleanup_checkpoints(run_id, checkpoint_dir=checkpoint_dir)
    except Exception:
        pass

    # ── 4. 最终评估 ──
    _cost_phase("final_evaluation")
    _log("\n═══ 最终评估 ═══")
    unevaluated = [s for s in population if s.fitness == 0.0]
    final_budget_assessment = _check_token_budget("final_evaluation_start")
    if unevaluated and not final_budget_assessment.get("should_skip_optional"):
        await evaluate_population(
            population,
            config.judge_dimensions,
            judge_models[0],
            rounds=config.judge_rounds,
            diversity_weight=config.diversity_weight,
        )
        run.total_api_calls += len(unevaluated) * config.judge_rounds
        trajectory.record(
            phase="final_evaluation",
            actor=judge_models[0],
            action="evaluate_unevaluated_solutions",
            input_summary=f"unevaluated={len(unevaluated)}",
            output_summary="fitness updated",
        )
    elif unevaluated and final_budget_assessment.get("should_skip_optional"):
        _log(f"  跳过最终补评审: token budget {final_budget_assessment['status']}")
        trajectory.record(
            phase="final_evaluation",
            actor="token_budget_controller",
            action="skip_final_evaluation",
            input_summary=f"unevaluated={len(unevaluated)}",
            output_summary=f"status={final_budget_assessment['status']}",
        )

    # Executable fitness for solutions that haven't been tested yet
    if (
        config.enable_executable_fitness
        and config.test_cases
        and not final_budget_assessment.get("should_skip_optional")
    ):
        untested = [s for s in population if s.execution_score is None]
        if untested:
            from evohive.engine.executable_fitness import evaluate_executable_fitness
            final_exec_reports = await evaluate_executable_fitness(
                untested,
                config.test_cases,
                exec_weight=config.exec_weight,
                timeout_per_case=config.exec_timeout,
            )
            n_tested = sum(1 for r in final_exec_reports if r.get("exec_score") is not None)
            _log(f"  ⚡ 最终可执行验证: {n_tested} 个方案补充测试")
            run.execution_results.append({
                "generation": "final",
                "reports": final_exec_reports,
            })

    population.sort(key=lambda s: s.fitness, reverse=True)

    _emit(evt.EVOLUTION_COMPLETE, "evolution",
          total_solutions=len(all_solutions),
          best_fitness=population[0].fitness if population else 0)

    # ── 5. 后进化阶段 (快速模式跳过 debate 和 pressure) ──
    _check_token_budget("post_evolution_start")

    # 红队攻击 (Top 5)
    if config.enable_red_team and not skip_optional_due_to_budget:
        _cost_phase("red_team")
        _log("红队攻击中...")
        _emit(evt.RED_TEAM_STARTED, "post_evolution")
        rt_results = await red_team_batch(
            population, config.problem, red_team_models, top_n=5,
        )
        apply_red_team_scores(population, rt_results)
        run.red_team_results = rt_results
        run.total_api_calls += 5 * len(red_team_models)
        vuln_map = {r["solution_id"]: r["overall_vulnerability"] for r in rt_results}
        for sol in population:
            if sol.id in vuln_map:
                sol.red_team_vulnerability = vuln_map[sol.id]
        trajectory.record(
            phase="red_team",
            actor="red_team_pool",
            action="attack_top_solutions",
            input_summary=f"top_n={min(5, len(population))}",
            output_summary=f"attacks={len(rt_results)}",
            metrics={"models": red_team_models},
        )
        _emit(evt.RED_TEAM_COMPLETE, "post_evolution",
              n_attacked=len(rt_results))

    # 辩论淘汰赛 (Top 5) — 快速模式跳过
    if config.enable_debate and config.mode != "fast" and not skip_optional_due_to_budget:
        _cost_phase("debate")
        _log("辩论淘汰赛中...")
        _emit(evt.DEBATE_STARTED, "post_evolution")
        debate_elos = await debate_tournament(
            population, config.problem, thinker_models[0],
            judge_models, top_n=5,
        )
        apply_debate_scores(population, debate_elos)
        run.debate_results = debate_elos
        n_debate_pairs = min(5, len(population)) * (min(5, len(population)) - 1) // 2
        run.total_api_calls += n_debate_pairs * len(judge_models) * 5
        for sol in population:
            if sol.id in debate_elos:
                sol.debate_elo = debate_elos[sol.id]
        trajectory.record(
            phase="debate",
            actor="debate_pool",
            action="debate_tournament",
            input_summary=f"top_n={min(5, len(population))}",
            output_summary=f"elo_updates={len(debate_elos)}",
            metrics={"judges": judge_models},
        )
        _emit(evt.DEBATE_COMPLETE, "post_evolution")

    # 极端压力测试 (Top 5) — 快速模式跳过
    if config.enable_pressure_test and config.mode != "fast" and not skip_optional_due_to_budget:
        _cost_phase("pressure_test")
        _log("极端压力测试中...")
        _emit(evt.PRESSURE_TEST_STARTED, "post_evolution")
        pt_results = await pressure_test_batch(
            population, config.problem, judge_models[0],
            n_scenarios=3, top_n=5,
        )
        apply_pressure_scores(population, pt_results)
        run.pressure_test_results = pt_results
        run.total_api_calls += 5 * 3
        res_map = {r["solution_id"]: r["avg_resilience"] for r in pt_results}
        for sol in population:
            if sol.id in res_map:
                sol.pressure_resilience = res_map[sol.id]
        trajectory.record(
            phase="pressure_test",
            actor=judge_models[0],
            action="pressure_test_batch",
            input_summary=f"top_n={min(5, len(population))}, scenarios=3",
            output_summary=f"reports={len(pt_results)}",
        )
        _emit(evt.PRESSURE_TEST_COMPLETE, "post_evolution")

    # ── 6. 最终排序 ──
    population.sort(key=lambda s: s.fitness, reverse=True)
    top_n = min(5, len(population))
    run.final_top_solutions = [s.model_dump() for s in population[:top_n]]
    run.all_solutions = [s.model_dump() for s in all_solutions]
    run.lineage_graph = build_lineage_graph(
        all_solutions,
        final_ids={solution.id for solution in population[:top_n]},
    )
    trajectory.record(
        phase="lineage",
        actor="lineage_builder",
        action="build_lineage_graph",
        input_summary=f"solutions={len(all_solutions)}, finalists={top_n}",
        output_summary=f"nodes={run.lineage_graph.get('summary', {}).get('node_count', 0)}",
        metrics=run.lineage_graph.get("summary", {}),
    )
    run.finished_at = datetime.now()

    # ── 7. 深度扩写 Top 1 (快速模式跳过) ──
    _check_token_budget("refinement_start")
    if population and config.mode != "fast" and not skip_optional_due_to_budget:
        _cost_phase("refinement")
        _log("深度扩写Top 1方案...")
        _emit(evt.REFINEMENT_STARTED, "refinement")
        top_solution = population[0]
        refined_content = await tool_augmented_refine(
            top_solution.content, config.problem, thinker_models[0],
            search_context=search_context,
        )
        run.refined_top_solution = refined_content
        run.total_api_calls += 7
        trajectory.record(
            phase="refinement",
            actor=thinker_models[0],
            action="tool_augmented_refine",
            input_summary=top_solution.content[:180],
            output_summary=run.refined_top_solution[:220],
        )
        _emit(evt.REFINEMENT_COMPLETE, "refinement")
    elif population and config.mode == "fast":
        # 快速模式: 直接用 Top1 原始内容，不做深度扩写
        run.refined_top_solution = population[0].content
        trajectory.record(
            phase="refinement",
            actor="fast_mode",
            action="use_top_solution_without_refine",
            output_summary=run.refined_top_solution[:220],
        )

    run.verification_report = build_verification_report(
        problem=config.problem,
        final_answer=run.refined_top_solution,
        lineage_graph=run.lineage_graph,
        search_context=search_context,
    )
    trajectory.record(
        phase="verification",
        actor="verification_engine",
        action="build_verification_report",
        input_summary=f"answer_chars={len(run.refined_top_solution)}",
        output_summary=f"claims={run.verification_report.get('summary', {}).get('claim_count', 0)}",
        metrics=run.verification_report.get("summary", {}),
    )
    run.claim_verification_report = build_claim_verification_report(
        verification_report=run.verification_report,
        search_context=search_context,
    )
    if config.enable_claim_search_verification:
        search_verification = await build_claim_search_verification_report(
            verification_report=run.verification_report,
            base_report=run.claim_verification_report,
        )
        if search_verification["searched_claim_count"]:
            searched_by_id = {
                claim["id"]: claim
                for claim in search_verification["claims"]
            }
            run.claim_verification_report["claims"] = [
                searched_by_id.get(claim.get("id"), claim)
                for claim in run.claim_verification_report.get("claims", [])
            ]
            run.claim_verification_report["active_search"] = search_verification
    trajectory.record(
        phase="claim_verification",
        actor="claim_verifier",
        action="verify_claims",
        input_summary=f"claims={run.claim_verification_report.get('claim_count', 0)}",
        output_summary=(
            f"needs_external_verifier="
            f"{run.claim_verification_report.get('summary', {}).get('needs_external_verifier', 0)}"
        ),
        metrics=run.claim_verification_report.get("summary", {}),
    )
    run.answer_graph = build_answer_graph(
        problem=config.problem,
        final_answer=run.refined_top_solution,
        lineage_graph=run.lineage_graph,
        verification_report=run.verification_report,
        top_solutions=run.final_top_solutions,
    )
    trajectory.record(
        phase="answer_graph",
        actor="graph_builder",
        action="build_answer_graph",
        output_summary=(
            f"nodes={run.answer_graph.get('summary', {}).get('node_count', 0)}, "
            f"edges={run.answer_graph.get('summary', {}).get('edge_count', 0)}"
        ),
        metrics=run.answer_graph.get("summary", {}),
    )

    # ── 8. Baseline vs Top1 质量对比 ──
    if population and run.baseline_solution and not skip_optional_due_to_budget:
        _cost_phase("quality_comparison")
        _log("Baseline vs Top1 自动质量对比...")
        try:
            baseline_sol = Solution(content=run.baseline_solution, id="baseline")
            top1_sol = population[0]
            # 用多个 Judge 模型做 debias 对比
            comparison_model = judge_models[0]
            comparison = await pairwise_compare(
                baseline_sol, top1_sol, config.problem,
                comparison_model, dimensions_text,
                debias=True,
            )
            evolution_wins = comparison["winner_id"] == top1_sol.id
            run.quality_comparison = {
                "winner": "evolution" if evolution_wins else "baseline",
                "confidence": comparison["confidence"],
                "reason": comparison["reason"],
                "judge_model": comparison_model,
            }
            run.total_api_calls += 2  # debias 模式下 2 次调用
            trajectory.record(
                phase="quality_comparison",
                actor=comparison_model,
                action="compare_baseline_vs_top1",
                input_summary="baseline vs champion",
                output_summary=f"winner={run.quality_comparison['winner']}",
                metrics={"confidence": run.quality_comparison["confidence"]},
            )
            improvement = "进化胜出" if evolution_wins else "基线胜出"
            _log(f"  质量对比: {improvement} (置信度: {comparison['confidence']:.2f})")
        except Exception as e:
            _log(f"  质量对比失败: {e}")
            run.quality_comparison = {"error": str(e)}

    # 记录提前终止原因
    if early_stop_reason:
        run.early_stop_reason = early_stop_reason

    # 保存事件日志
    if emitter:
        run.event_log = [e.to_dict() for e in emitter.get_event_log()]

    # 记录成本数据
    cost_snapshot = cost_tracker.snapshot()
    if cost_snapshot.get("total_calls", 0) > 0:
        run.total_api_calls = cost_snapshot["total_calls"]
    run.estimated_cost = cost_snapshot["total_cost"]
    run.cost_breakdown = cost_snapshot
    run.resource_report = _build_resource_report(cost_snapshot)
    run.token_budget_events = token_budget_events
    run.token_budget_report = build_token_budget_report(
        plan=run.token_budget_plan,
        resource_report=run.resource_report,
    )
    provider_budget_events = get_session_token_budget_events()
    if provider_budget_events:
        token_budget_events.extend(provider_budget_events)
    if token_budget_events:
        run.token_budget_report["runtime_events"] = token_budget_events
    trajectory.record(
        phase="resource",
        actor="resource_monitor",
        action="build_resource_report",
        output_summary=f"calls={run.total_api_calls}, tokens={run.resource_report.get('total_tokens', 0)}",
        metrics={
            "duration_sec": run.resource_report.get("duration_sec", 0),
            "estimated_cost": run.estimated_cost,
        },
    )
    run.trajectory_log = trajectory.to_list()
    run.trajectory_summary = trajectory.summary()
    run.trajectory_replay = build_trajectory_replay(run.trajectory_log)
    log_event(_logger, "cost_checkpoint",
              total_cost=run.estimated_cost,
              total_calls=run.total_api_calls,
              budget_limit=budget_limit)

    _emit(evt.RUN_COMPLETE, "complete",
          total_api_calls=run.total_api_calls,
          estimated_cost=run.estimated_cost,
          duration=(run.finished_at - run.started_at).total_seconds() if run.finished_at else 0)

    log_event(_logger, "run_complete",
              run_id=run_id,
              total_api_calls=run.total_api_calls,
              estimated_cost=run.estimated_cost,
              duration_s=(run.finished_at - run.started_at).total_seconds() if run.finished_at else 0,
              best_fitness=population[0].fitness if population else 0,
              total_solutions=len(all_solutions),
              early_stop=early_stop_reason)

    # 自动保存结果
    if save_results:
        try:
            from evohive.engine.persistence import save_run_result
            saved_path = save_run_result(run, output_dir=output_dir)
            _log(f"结果已保存: {saved_path}")
        except Exception as e:
            _log(f"保存结果失败: {e}")

    reset_session_cost_tracker(cost_tracker_tokens)
    clear_session_token_budget()
    return run


_run_evolution_impl = run_evolution


@wraps(_run_evolution_impl)
async def run_evolution(*args, **kwargs):
    """Run evolution and always clear per-run cost context afterward."""
    try:
        return await _run_evolution_impl(*args, **kwargs)
    finally:
        clear_session_cost_tracker()
        clear_session_token_budget()
