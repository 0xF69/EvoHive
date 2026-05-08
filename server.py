"""EvoHive backend server.

Mock mode: simulates the full 13-phase evolution pipeline using the real
event protocol, with fake LLM responses (no API keys needed).

Real mode: import the actual engine and run with real LLM calls.
"""

import asyncio
import json
import logging
import math
import os
import random
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

from evohive.server.providers import (
    PROVIDER_ENV_VARS,
    discover_provider_models_with_source as _discover_provider_models_with_source,
    normalize_api_keys as _normalize_api_keys,
    normalize_search_api_keys as _normalize_search_api_keys,
    probe_manual_models as _probe_manual_models,
    probe_provider_access as _probe_provider_access,
    redact_config as _redact_config,
)
from evohive.server.artifacts import persist_run_artifact as _persist_run_artifact_impl
from evohive.server.catalog import PROVIDER_MODEL_MAP, PROVIDERS
from evohive.server.costing import (
    estimate_model_unit_rate as _estimate_model_unit_rate,
    estimate_run_cost,
    provider_models_for_config as _provider_models_for_config,
)
from evohive.server.schemas import (
    ProviderModelCheckRequest,
    ProviderModelDiscoverRequest,
    ProviderPreflightRequest,
    RunEstimateRequest,
)
from evohive.engine.answer_graph import build_answer_graph
from evohive.engine.claim_verifier import build_claim_verification_report
from evohive.engine.effort import normalize_token_budget_control
from evohive.engine.token_budget import build_token_budget_report
from evohive.engine.trajectory_replay import build_trajectory_replay
from evohive.engine.verification import build_verification_report

app = FastAPI(title="EvoHive Backend Server")
logger = logging.getLogger("evohive.server")

ARTIFACT_ROOT = Path(__file__).parent / "evohive_runs"
CHECKPOINT_DIR = Path(__file__).parent / "evohive_checkpoints"
DEFAULT_REAL_RUN_TIMEOUT_SEC = int(os.environ.get("EVOHIVE_WEB_TIMEOUT_SEC", "1800"))
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,80}$")


class BudgetGuardError(Exception):
    """Raised when a run is likely to exceed the configured budget."""


class ClientMessageError(ValueError):
    """Raised when a WebSocket client sends an invalid control message."""


def _now_ts() -> float:
    return round(time.time(), 3)


def _make_event_message(event_type: str, phase: str, data: dict) -> dict:
    return {
        "type": "evolution_event",
        "event_type": event_type,
        "phase": phase,
        "ts": _now_ts(),
        "data": data,
    }


def _enforce_budget_guard(config: dict | None) -> dict:
    estimate = estimate_run_cost(config)
    if estimate["risk"] != "safe" and not bool((config or {}).get("allow_budget_override")):
        raise BudgetGuardError(
            f"Estimated cost ${estimate['low']:.2f}-${estimate['high']:.2f} may exceed the configured budget "
            f"(${estimate['budget']:.2f})."
        )
    return estimate


def _resolve_token_budget_settings(config: dict | None) -> dict:
    cfg = config or {}
    raw_control = cfg.get("token_budget_control", cfg.get("budget_control", ""))
    enabled_flag = bool(cfg.get("enable_token_budget_control", False))
    control = normalize_token_budget_control(str(raw_control or ""), enabled=enabled_flag)
    if control == "off":
        enabled_flag = False
    else:
        enabled_flag = True

    try:
        multiplier = float(cfg.get("token_budget_multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        multiplier = 1.0
    multiplier = min(max(multiplier, 0.1), 10.0)

    return {
        "mode": control,
        "enabled": enabled_flag,
        "multiplier": multiplier,
        "options": ["off", "auto", "relaxed", "strict"],
    }


def _persist_run_artifact(run_id: str, mode: str, config: dict, results: dict, events: list[dict]) -> dict:
    return _persist_run_artifact_impl(
        ARTIFACT_ROOT,
        PROVIDERS,
        run_id,
        mode,
        config,
        results,
        events,
    )


async def _persist_run_artifact_async(run_id: str, mode: str, config: dict, results: dict, events: list[dict]) -> dict:
    return await asyncio.to_thread(_persist_run_artifact, run_id, mode, config, results, events)


def _validate_run_id(run_id: str) -> str:
    clean = str(run_id or "").strip()
    if not RUN_ID_RE.fullmatch(clean):
        raise HTTPException(status_code=400, detail="invalid run_id")
    return clean


def _read_json_file(path: Path) -> dict | list:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"corrupt JSON artifact: {path.name}") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not read artifact: {path.name}") from exc


async def _read_json_file_async(path: Path) -> dict | list:
    return await asyncio.to_thread(_read_json_file, path)


def _find_run_dir(run_id: str) -> Path | None:
    run_id = _validate_run_id(run_id)
    if not ARTIFACT_ROOT.exists():
        return None
    candidates = sorted(
        [path for path in ARTIFACT_ROOT.iterdir() if path.is_dir() and path.name.startswith(f"{run_id}-")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _normalize_runtime_error(exc: Exception) -> dict:
    if isinstance(exc, ClientMessageError):
        return {
            "error": str(exc),
            "error_code": "invalid_message",
            "retryable": False,
        }
    if isinstance(exc, asyncio.TimeoutError):
        return {
            "error": "Evolution run exceeded the configured timeout.",
            "error_code": "run_timeout",
            "retryable": True,
        }
    if exc.__class__.__name__ == "BudgetExceededError":
        return {
            "error": str(exc),
            "error_code": "budget_exceeded",
            "retryable": True,
        }
    if isinstance(exc, BudgetGuardError):
        return {
            "error": str(exc),
            "error_code": "budget_exceeded",
            "retryable": True,
        }
    lowered = str(exc).lower()
    if "pre-flight" in lowered or "unreachable" in lowered:
        return {
            "error": str(exc),
            "error_code": "model_unavailable",
            "retryable": True,
        }
    return {
        "error": str(exc),
        "error_code": "run_failed",
        "retryable": False,
    }


def _make_ws_error_payload(exc: Exception, config: dict | None = None) -> dict:
    err = _normalize_runtime_error(exc)
    error_id = f"err-{uuid.uuid4().hex[:10]}"
    payload = {
        "type": "war_error",
        "error": err["error"],
        "error_code": err["error_code"],
        "retryable": err["retryable"],
        "error_id": error_id,
    }
    if err["error_code"] == "budget_exceeded":
        payload["budget_estimate"] = estimate_run_cost(config or {})
    return payload


def _service_metadata() -> dict:
    return {
        "service": "EvoHive Backend",
        "status": "ok",
        "endpoints": {
            "status": "/api/status",
            "websocket": "/ws",
            "runs": "/api/runs",
            "estimate": "/api/runs/estimate",
            "provider_preflight": "/api/providers/preflight",
            "provider_model_check": "/api/providers/model-check",
        },
    }


@app.api_route("/", methods=["GET", "HEAD"])
async def service_index():
    return _service_metadata()


@app.api_route("/api/status", methods=["GET", "HEAD"])
async def backend_index():
    return _service_metadata()


@app.get("/api/runs")
async def list_runs(limit: int = Query(20, ge=1, le=100)):
    if not ARTIFACT_ROOT.exists():
        return {"runs": []}
    runs = []
    for run_dir in sorted(
        [path for path in ARTIFACT_ROOT.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]:
        run_json = run_dir / "run.json"
        if not run_json.exists():
            continue
        try:
            payload = await _read_json_file_async(run_json)
        except HTTPException as exc:
            logger.warning("Skipping unreadable run manifest %s: %s", run_json, exc.detail)
            continue
        if not isinstance(payload, dict):
            logger.warning("Skipping non-object run manifest %s", run_json)
            continue
        runs.append({
            "run_id": payload.get("run_id"),
            "mode": payload.get("mode"),
            "problem": payload.get("results", {}).get("problem", ""),
            "estimated_cost": payload.get("results", {}).get("estimated_cost", 0),
            "api_calls": payload.get("results", {}).get("total_api_calls", 0),
            "dir": str(run_dir),
            "updated_at": run_dir.stat().st_mtime,
        })
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise HTTPException(status_code=404, detail="run manifest not found")
    return JSONResponse(content=await _read_json_file_async(run_json))


@app.get("/api/runs/{run_id}/replay")
async def get_run_replay(run_id: str):
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    replay_path = run_dir / "replay.json"
    if not replay_path.exists():
        raise HTTPException(status_code=404, detail="replay not found")
    return JSONResponse(content=await _read_json_file_async(replay_path))


@app.get("/api/runs/{run_id}/answer-graph")
async def get_run_answer_graph(run_id: str):
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise HTTPException(status_code=404, detail="run manifest not found")
    payload = await _read_json_file_async(run_json)
    answer_graph = (payload.get("results") or {}).get("answer_graph")
    if not answer_graph:
        raise HTTPException(status_code=404, detail="answer graph not found")
    return JSONResponse(content=answer_graph)


@app.get("/api/runs/{run_id}/trajectory-replay")
async def get_run_trajectory_replay(run_id: str):
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise HTTPException(status_code=404, detail="run manifest not found")
    payload = await _read_json_file_async(run_json)
    results = payload.get("results") or {}
    replay = results.get("trajectory_replay")
    if not replay:
        replay = build_trajectory_replay(results.get("trajectory_log") or [])
    return JSONResponse(content=replay)


@app.get("/api/runs/{run_id}/claim-verification")
async def get_run_claim_verification(run_id: str):
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise HTTPException(status_code=404, detail="run manifest not found")
    payload = await _read_json_file_async(run_json)
    report = (payload.get("results") or {}).get("claim_verification_report")
    if not report:
        raise HTTPException(status_code=404, detail="claim verification not found")
    return JSONResponse(content=report)


@app.get("/api/checkpoints")
async def get_checkpoints():
    from evohive.engine.checkpoint import list_checkpoints

    return {
        "checkpoints": await asyncio.to_thread(
            list_checkpoints,
            str(CHECKPOINT_DIR),
        )
    }


@app.get("/api/checkpoints/{run_id}")
async def get_checkpoint(run_id: str):
    from evohive.engine.checkpoint import load_checkpoint

    run_id = _validate_run_id(run_id)
    checkpoint = await asyncio.to_thread(load_checkpoint, run_id, str(CHECKPOINT_DIR))
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="checkpoint not found")
    return JSONResponse(content=checkpoint)


@app.post("/api/provider-models/discover")
async def discover_provider_models(payload: ProviderModelDiscoverRequest):
    provider = payload.provider.strip().lower()
    api_key = payload.api_key.strip()
    if provider not in PROVIDER_ENV_VARS:
        raise HTTPException(status_code=400, detail="unsupported provider")
    models, source = await _discover_provider_models_with_source(provider, api_key)
    return {
        "provider": provider,
        "models": models,
        "source": source,
    }


@app.post("/api/runs/estimate")
async def estimate_run(payload: RunEstimateRequest):
    config = payload.model_dump()
    estimate = estimate_run_cost(config)
    token_budget = _resolve_token_budget_settings(config)
    return {
        "estimate": estimate,
        "guarded": estimate["risk"] == "safe" or bool(config.get("allow_budget_override")),
        "token_budget_control": token_budget,
    }


@app.post("/api/providers/model-check")
async def provider_model_check(payload: ProviderModelCheckRequest):
    provider = payload.provider.strip().lower()
    api_key = _normalize_api_keys({provider: payload.api_key}).get(provider, "")
    models = [str(model).strip() for model in payload.models if str(model).strip()]
    result = await _probe_manual_models(provider, api_key, models)
    return result


@app.post("/api/providers/preflight")
async def provider_preflight(payload: ProviderPreflightRequest):
    providers = _normalize_selected_providers(payload.providers) if payload.providers else []
    api_keys = _normalize_api_keys(payload.api_keys)
    results = []
    for provider in providers:
        results.append(await _probe_provider_access(provider, api_keys.get(provider, "")))
    ready = [item for item in results if item["ok"]]
    return {
        "providers": results,
        "ready": len(ready) > 0,
        "ready_count": len(ready),
    }


def _normalize_selected_providers(raw_providers) -> list[str]:
    """Return a non-empty, de-duplicated provider list.

    EvoHive optimizes answers, not provider rankings, so one provider is a
    valid run: the selected model can spawn the whole agent population.
    """
    providers: list[str] = []
    for provider in raw_providers or ["deepseek"]:
        normalized = str(provider or "").strip().lower()
        if not normalized:
            continue
        if normalized not in PROVIDER_MODEL_MAP and normalized not in PROVIDERS:
            continue
        if normalized not in providers:
            providers.append(normalized)
    return providers or ["deepseek"]


def _format_litellm_model(provider: str, model: str) -> str:
    model_id = str(model or "").strip()
    if not model_id:
        return ""
    first_segment = model_id.split("/", 1)[0]
    known_litellm_prefixes = set(PROVIDER_MODEL_MAP) | {"together_ai", "fireworks_ai"}
    if first_segment in known_litellm_prefixes:
        return model_id
    return f"{provider}/{model_id}"


def _build_litellm_model_pool(providers: list[str], provider_models_map: dict | None = None) -> list[str]:
    """Build the model pool used by the engine.

    With a single provider/model this intentionally returns one model. The
    engine then creates many thinkers from that same model, producing many
    competing solution agents for the user's task.
    """
    model_pool: list[str] = []
    explicit_map = provider_models_map or {}
    for provider in _normalize_selected_providers(providers):
        explicit_models = explicit_map.get(provider)
        if isinstance(explicit_models, list) and explicit_models:
            for model in explicit_models:
                formatted = _format_litellm_model(provider, model)
                if formatted:
                    model_pool.append(formatted)
        elif provider in PROVIDER_MODEL_MAP:
            model_pool.extend(PROVIDER_MODEL_MAP[provider])
    return list(dict.fromkeys(model_pool)) or ["deepseek/deepseek-chat"]


# ═══════════════════════════════════════════════════════════════════
# REAL MODE — Bridge core EvoHive engine to WebSocket
# ═══════════════════════════════════════════════════════════════════

async def run_real_evolution(ws: WebSocket, config: dict):
    """Run the REAL EvoHive evolution engine, bridging events to WebSocket."""
    from evohive.models import EvolutionConfig
    from evohive.engine.evolution import run_evolution
    from evohive.engine.events import EventEmitter
    from evohive.engine.web_search import reset_session_search_api_keys, set_session_search_api_keys
    from evohive.llm.provider import reset_session_api_keys, set_session_api_keys

    providers = _normalize_selected_providers(config.get("providers", ["deepseek"]))
    total = config.get("total", 15)
    gens = config.get("gens", 2)
    mode = config.get("mode", "fast")
    reasoning_effort = config.get("reasoning_effort") or mode
    budget = float(config.get("budget", 0.5) or 0.5)
    token_budget = _resolve_token_budget_settings(config)
    run_timeout_sec = int(config.get("run_timeout_sec") or DEFAULT_REAL_RUN_TIMEOUT_SEC)
    resume_from = str(config.get("resume_from") or "").strip() or None
    enable_search = bool(config.get("enable_search", False))
    session_api_keys = _normalize_api_keys(config.get("api_keys"))
    session_search_api_keys = _normalize_search_api_keys(config.get("search_api_keys"))
    safe_config = _redact_config(config)
    problem = config.get("problem",
        "Write a Python function `longest_palindrome(s: str) -> str` that "
        "returns the longest palindromic substring in s. "
        "Handle edge cases. Optimize for O(n^2) or better."
    )

    provider_models_map = config.get("provider_models") or {}
    budget_estimate = _enforce_budget_guard(config)

    # One model is enough: it will spawn many independent solution agents.
    thinker_models = _build_litellm_model_pool(providers, provider_models_map)

    # Use the first model as primary, rest as multi-model pool
    primary_model = thinker_models[0]
    single_model_mode = len(thinker_models) == 1

    # Default judge dimensions (same as sdk.py _DEFAULT_DIMENSIONS)
    default_dimensions = [
        {"name": "feasibility", "weight": 0.3, "description": "feasibility"},
        {"name": "innovation", "weight": 0.25, "description": "innovation"},
        {"name": "specificity", "weight": 0.25, "description": "specificity"},
        {"name": "cost_efficiency", "weight": 0.2, "description": "cost efficiency"},
    ]

    # Build EvolutionConfig — conservative settings for testing
    evo_config = EvolutionConfig(
        problem=problem,
        population_size=min(total, 20),   # cap for cost control
        generations=min(gens, 3),          # cap for cost control
        thinker_model=primary_model,
        judge_model=primary_model,
        thinker_models=thinker_models,
        judge_models=thinker_models[:2],   # use first 2 as judges
        red_team_models=thinker_models[:1],
        judge_dimensions=default_dimensions,
        # Feature toggles — enable core features, disable expensive ones
        enable_pairwise_judge=True,
        enable_elimination_memory=True,
        enable_diversity_guard=True,
        enable_fresh_blood=True,
        enable_red_team=True,
        enable_debate=True,
        enable_pressure_test=True,
        # Disable features that need OpenAI embedding or web search
        enable_swarm=False,
        enable_web_search=enable_search,
        enable_swiss_tournament=True,
        enable_adaptive=True,
        # Mode
        mode=mode,
        reasoning_effort=reasoning_effort,
        token_budget_control=token_budget["mode"],
        token_budget_multiplier=token_budget["multiplier"],
        enable_token_budget_control=token_budget["enabled"],
    )

    # Send war_started
    await ws.send_json({
        "type": "war_started",
        "config": {
            "total": evo_config.population_size,
            "gens": evo_config.generations,
            "mode": "REAL",
            "reasoning_effort": evo_config.reasoning_effort,
            "token_budget_control": token_budget,
            "providers": providers,
            "problem": problem,
            "budget": budget,
            "budget_estimate": budget_estimate,
            "run_timeout_sec": run_timeout_sec,
            "resume_from": resume_from,
            "enable_search": enable_search,
            "model_pool_size": len(thinker_models),
            "single_model_mode": single_model_mode,
        }
    })

    # Create EventEmitter and bridge to WebSocket
    emitter = EventEmitter()
    all_events = []
    start_time = time.time()

    def on_event(event):
        """Synchronous callback from EventEmitter — queue for async send."""
        msg = _make_event_message(event.type, event.phase, event.data)
        all_events.append(msg)
        # Schedule async send on the event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_safe_send(ws, msg))
        except Exception:
            pass

    emitter.on_event(on_event)

    # Status callback
    def on_status(message: str):
        msg = _make_event_message("status_update", "info", {"message": message})
        all_events.append(msg)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_safe_send(ws, msg))
        except Exception:
            pass

    # Generation complete callback
    def on_gen_complete(generation, stats, best_solution):
        gen_data = {
            "generation": generation,
            "best_fitness": round(getattr(stats, 'best_fitness', 0), 3),
            "avg_fitness": round(getattr(stats, 'avg_fitness', 0), 3),
            "alive_count": getattr(stats, 'population_size', 0),
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_safe_send(ws, {"type": "generation_summary", "data": gen_data}))
        except Exception:
            pass

    # Run the real engine
    if single_model_mode:
        on_status(f"Starting REAL single-model evolution: {thinker_models[0]} will spawn {evo_config.population_size} agents.")
    else:
        on_status(f"Starting REAL multi-model evolution with {len(thinker_models)} model(s): {', '.join(thinker_models)}")
    on_status(f"Problem: {problem[:100]}...")

    llm_key_token = set_session_api_keys(session_api_keys)
    search_key_token = set_session_search_api_keys(session_search_api_keys)
    try:
        result = await asyncio.wait_for(
            run_evolution(
                config=evo_config,
                on_generation_complete=on_gen_complete,
                on_status=on_status,
                emitter=emitter,
                budget_limit=budget,
                save_results=False,
                resume_from=resume_from,
                checkpoint_dir=str(CHECKPOINT_DIR),
            ),
            timeout=run_timeout_sec,
        )
    finally:
        reset_session_search_api_keys(search_key_token)
        reset_session_api_keys(llm_key_token)

    # Send war_complete with real results
    elapsed = time.time() - start_time
    champion_data = {}
    top5_data = []

    # Extract champion from final_top_solutions
    if result.final_top_solutions:
        top_sol = result.final_top_solutions[0]
        champion_data = {
            "id": top_sol.get("id", "champion"),
            "provider": top_sol.get("model", primary_model).split("/")[0],
            "model": top_sol.get("model", primary_model),
            "elo": round(top_sol.get("elo", 1200), 1),
            "fitness": round(top_sol.get("fitness", 0), 3),
            "hp": 100,
            "gen": evo_config.generations,
            "alive": True,
            "is_elite": True,
            "kills": 0,
        }
        for i, sol in enumerate(result.final_top_solutions[:5]):
            top5_data.append({
                "id": sol.get("id", f"top-{i+1}"),
                "provider": sol.get("model", primary_model).split("/")[0],
                "model": sol.get("model", primary_model),
                "elo": round(sol.get("elo", 1200), 1),
                "fitness": round(sol.get("fitness", 0), 3),
                "hp": 100 - i * 10,
                "gen": evo_config.generations,
                "alive": True,
                "is_elite": i == 0,
                "kills": 0,
            })

    generations_data = []
    for gs in result.generations_data:
        generations_data.append({
            "generation": gs.generation,
            "best_fitness": round(gs.best_fitness, 3),
            "avg_fitness": round(gs.avg_fitness, 3),
            "alive_count": gs.alive_count,
        })

    results_payload = {
        "problem": problem,
        "champion": champion_data,
        "total_api_calls": result.total_api_calls,
        "estimated_cost": round(result.estimated_cost, 3),
        "cost_breakdown": result.cost_breakdown,
        "resource_report": result.resource_report,
        "token_budget_control": {
            **token_budget,
            "effective_enabled": result.config.enable_token_budget_control,
            "effective_multiplier": result.config.token_budget_multiplier,
        },
        "token_budget_plan": result.token_budget_plan,
        "token_budget_report": result.token_budget_report,
        "token_budget_events": result.token_budget_events,
        "trajectory_log": result.trajectory_log,
        "trajectory_summary": result.trajectory_summary,
        "trajectory_replay": result.trajectory_replay,
        "lineage_graph": result.lineage_graph,
        "verification_report": result.verification_report,
        "claim_verification_report": result.claim_verification_report,
        "answer_graph": result.answer_graph,
        "event_count": len(all_events),
        "generations_data": generations_data,
        "top5": top5_data if top5_data else [champion_data],
        "evolved_answer": result.refined_top_solution or (
            result.final_top_solutions[0].get("content", "") if result.final_top_solutions else ""
        ),
    }
    artifact = await _persist_run_artifact_async(
        run_id=getattr(result, "id", uuid.uuid4().hex[:12]),
        mode="real",
        config={
            **safe_config,
            "providers": providers,
            "total": evo_config.population_size,
            "gens": evo_config.generations,
            "mode": mode,
            "reasoning_effort": result.reasoning_effort,
            "token_budget_control": {
                **token_budget,
                "effective_enabled": result.config.enable_token_budget_control,
                "effective_multiplier": result.config.token_budget_multiplier,
            },
            "budget": budget,
            "run_timeout_sec": run_timeout_sec,
            "resume_from": resume_from,
            "enable_search": enable_search,
            "problem": problem,
            "model_pool_size": len(thinker_models),
            "single_model_mode": single_model_mode,
        },
        results=results_payload,
        events=all_events,
    )
    results_payload["structured_result"] = artifact["structured_result"]
    results_payload["telemetry"] = artifact["telemetry"]
    results_payload["artifact"] = {k: v for k, v in artifact.items() if k != "structured_result"}

    await ws.send_json({
        "type": "war_complete",
        "results": results_payload,
    })


async def _safe_send(ws: WebSocket, msg: dict):
    """Send a message over WebSocket, ignoring errors."""
    try:
        await ws.send_json(msg)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# MOCK MODE — Simulated evolution (no API keys needed)
# ═══════════════════════════════════════════════════════════════════

class MockAgent:
    """Simulated agent for the mock evolution."""
    _counter = 0

    def __init__(self, provider: str, model: str, gen: int = 0):
        MockAgent._counter += 1
        self.id = f"{provider[:3].lower()}-{MockAgent._counter:03d}"
        self.provider = provider
        self.provider_name = PROVIDERS.get(provider, {}).get("name", provider)
        self.model = model
        self.elo = 1200.0 + random.gauss(0, 50)
        self.fitness = random.uniform(0.3, 0.8)
        self.hp = 100
        self.gen = gen
        self.alive = True
        self.is_elite = False
        self.kills = 0
        self.content = f"Solution by {model}: approach #{MockAgent._counter}"

    def to_dict(self):
        return {
            "id": self.id, "provider": self.provider, "model": self.model,
            "elo": round(self.elo, 1), "fitness": round(self.fitness, 3),
            "hp": self.hp, "gen": self.gen, "alive": self.alive,
            "is_elite": self.is_elite, "kills": self.kills,
        }


async def run_mock_evolution(ws: WebSocket, config: dict):
    """Run a full mock evolution pipeline, emitting events over WebSocket."""
    safe_config = _redact_config(config)
    providers = _normalize_selected_providers(config.get("providers", ["deepseek", "gemini"]))
    total = config.get("total", 30)
    gens = config.get("gens", 3)
    mode = config.get("mode", "standard")
    reasoning_effort = config.get("reasoning_effort") or mode

    SURVIVAL_RATE = 0.2
    ELITE_RATE = 0.05
    MUTATION_RATE = 0.3
    HOMOGENEITY_THRESHOLD = 0.6
    FRESH_BLOOD_INTERVAL = 2

    MockAgent._counter = 0
    api_calls = 0
    all_events = []
    mock_started_at = time.perf_counter()

    async def emit(event_type: str, phase: str, **data):
        nonlocal api_calls
        msg = _make_event_message(event_type, phase, data)
        all_events.append(msg)
        try:
            await ws.send_json(msg)
        except Exception:
            pass
        await asyncio.sleep(0.15)

    async def status(message: str):
        await emit("status_update", "info", message=message)

    await ws.send_json({
        "type": "war_started",
        "config": {
            "total": total,
            "gens": gens,
            "mode": mode,
            "reasoning_effort": reasoning_effort,
            "providers": providers,
            "single_model_mode": len(providers) == 1,
        }
    })
    await asyncio.sleep(0.5)

    await emit("preflight_ok", "init", models=[f"{PROVIDERS[p]['name']}/{PROVIDERS[p]['models'][0]}" for p in providers if p in PROVIDERS])
    if len(providers) == 1:
        await status(f"Single-model evolution ready: {PROVIDERS[providers[0]]['name']} will generate the full agent population.")
    else:
        await status(f"Pre-flight check passed: {len(providers)} provider(s) reachable.")
    api_calls += len(providers)

    agents = []
    per_provider = max(1, total // len(providers))
    for p in providers:
        if p not in PROVIDERS:
            continue
        model = random.choice(PROVIDERS[p]["models"])
        for _ in range(per_provider):
            if len(agents) >= total:
                break
            agents.append(MockAgent(p, model))
    while len(agents) < total:
        p = random.choice(providers)
        if p in PROVIDERS:
            agents.append(MockAgent(p, random.choice(PROVIDERS[p]["models"])))

    await emit("run_started", "init", mode=mode, problem="Mock evolution test")
    await emit("evolution_started", "evolution", population_size=len(agents), generations=gens)
    api_calls += total

    for gen in range(1, gens + 1):
        await emit("generation_started", "evolution", generation=gen)
        await status(f"═══ Generation {gen}/{gens} ═══")
        await asyncio.sleep(0.3)

        alive = [a for a in agents if a.alive]

        await emit("evaluation_started", "evolution", generation=gen)
        for idx, a in enumerate(alive):
            a.fitness = random.uniform(0.3, 0.95)
            a.fitness += gen * 0.03
            a.fitness = min(1.0, a.fitness)
            await emit("evaluation_progress", "evolution",
                        done=idx+1, total=len(alive), solution_id=a.id)
            await asyncio.sleep(0.12)
        alive.sort(key=lambda a: a.fitness, reverse=True)
        api_calls += len(alive)

        best_f = max(a.fitness for a in alive)
        avg_f = sum(a.fitness for a in alive) / len(alive)
        await emit("evaluation_complete", "evolution",
                    generation=gen, best_fitness=round(best_f, 3), avg_fitness=round(avg_f, 3))

        await emit("elo_tournament_started", "evolution", generation=gen, method="Swiss")
        n_rounds = max(3, math.ceil(math.log2(len(alive))))
        for r in range(n_rounds):
            random.shuffle(alive)
            n_pairs = 0
            for i in range(0, len(alive) - 1, 2):
                a, b = alive[i], alive[i + 1]
                winner = a if random.random() < 0.5 + (a.fitness - b.fitness) * 0.3 else b
                loser = b if winner == a else a
                k = 32
                ea = 1 / (1 + 10 ** ((loser.elo - winner.elo) / 400))
                winner.elo += k * (1 - ea)
                loser.elo -= k * ea
                winner.kills += 1
                loser.hp = max(0, loser.hp - random.randint(5, 10))
                api_calls += 1
                n_pairs += 1
            # Emit round progress
            top5 = {a.id: round(a.elo, 1) for a in sorted(alive, key=lambda x: x.elo, reverse=True)[:5]}
            await emit("elo_round_complete", "evolution",
                        round=r+1, total_rounds=n_rounds, n_pairs=n_pairs, top5=top5)
            await asyncio.sleep(0.2)

        n_comparisons = len(alive) * n_rounds // 2
        await emit("elo_tournament_complete", "evolution",
                    generation=gen, n_comparisons=n_comparisons)

        alive.sort(key=lambda a: a.elo, reverse=True)
        n_survive = max(2, int(len(alive) * SURVIVAL_RATE))
        n_elite = max(1, int(len(alive) * ELITE_RATE))
        for i, a in enumerate(alive):
            a.is_elite = i < n_elite
        survivors = alive[:n_survive]
        eliminated = alive[n_survive:]
        for a in eliminated:
            a.alive = False
            a.hp = 0
        await emit("selection_complete", "evolution",
                    generation=gen, survivors=len(survivors),
                    eliminated=len(eliminated),
                    survived_count=len(survivors), eliminated_count=len(eliminated))
        await asyncio.sleep(0.5)

        await emit("crossover_started", "evolution", generation=gen)
        await asyncio.sleep(0.3)
        children = []
        while len(survivors) + len(children) < total:
            p1, p2 = random.sample(survivors, min(2, len(survivors)))
            child = MockAgent(
                random.choice([p1.provider, p2.provider]),
                random.choice([p1.model, p2.model]),
                gen=gen
            )
            child.elo = (p1.elo + p2.elo) / 2 + random.gauss(0, 20)
            child.fitness = (p1.fitness + p2.fitness) / 2 + random.gauss(0, 0.05)
            children.append(child)
            api_calls += 3

        n_children = len(children)
        await emit("crossover_complete", "evolution",
                    generation=gen, n_children=n_children, children_count=n_children, new_count=n_children)

        n_mutated = 0
        for c in children:
            if random.random() < MUTATION_RATE:
                c.fitness += random.gauss(0, 0.1)
                c.fitness = max(0.1, min(1.0, c.fitness))
                c.elo += random.gauss(0, 30)
                n_mutated += 1
                api_calls += 1
        await emit("mutation_complete", "evolution",
                    generation=gen, n_mutated=n_mutated, mutated_count=n_mutated)

        culled = []
        alive_children = []
        for c in children:
            sim = random.random()
            if sim > HOMOGENEITY_THRESHOLD and random.random() < 0.3:
                c.alive = False
                culled.append(c)
            else:
                alive_children.append(c)
        if culled:
            await emit("homogeneity_culled", "evolution",
                        generation=gen, n_killed=len(culled),
                        culled_count=len(culled), killed_count=len(culled))

        fresh = []
        if gen % FRESH_BLOOD_INTERVAL == 0 or culled:
            n_inject = len(culled) if culled else max(1, total // 10)
            for _ in range(n_inject):
                p = random.choice(providers)
                if p in PROVIDERS:
                    fb = MockAgent(p, random.choice(PROVIDERS[p]["models"]), gen=gen)
                    fb.fitness = random.uniform(0.3, 0.7)
                    fresh.append(fb)
                    api_calls += 1
            if fresh:
                await emit("fresh_blood_injected", "evolution",
                            generation=gen, n_injected=len(fresh),
                            injected_count=len(fresh), count=len(fresh))

        agents = [a for a in agents if a.alive]
        agents = list(survivors) + alive_children + fresh
        for a in agents:
            a.alive = True
            a.hp = min(100, a.hp + 30)

        fitnesses = [a.fitness for a in agents]
        gen_summary = {
            "generation": gen,
            "best_fitness": round(max(fitnesses), 3),
            "avg_fitness": round(sum(fitnesses) / len(fitnesses), 3),
            "alive_count": len(agents),
        }
        await ws.send_json({"type": "generation_summary", "data": gen_summary})
        await emit("generation_complete", "evolution", generation=gen,
                    best_fitness=gen_summary["best_fitness"],
                    avg_fitness=gen_summary["avg_fitness"])
        await asyncio.sleep(0.5)

    await emit("red_team_started", "post_evolution")
    await status("Red team attacking top 5...")
    top5 = sorted(agents, key=lambda a: a.elo, reverse=True)[:5]
    for a in top5:
        vuln = random.uniform(0.1, 0.6)
        await emit("red_team_attack", "post_evolution",
                    target=a.id, vulnerability=round(vuln, 2))
        a.fitness *= (1 - vuln * 0.15)
        api_calls += len(providers)
        await asyncio.sleep(0.2)
    await emit("red_team_complete", "post_evolution", n_attacked=5)

    await emit("debate_started", "post_evolution")
    await status("Debate tournament...")
    pairs = [(top5[i], top5[j]) for i in range(len(top5)) for j in range(i+1, len(top5))]
    for i, (a, b) in enumerate(pairs[:4]):
        winner = a if a.elo > b.elo else b
        await emit("debate_round", "post_evolution",
                    round=i+1, a=a.id, b=b.id, winner=winner.id)
        winner.elo += 15
        api_calls += 5
        await asyncio.sleep(0.2)
    await emit("debate_complete", "post_evolution")

    await emit("pressure_test_started", "post_evolution")
    await status("Extreme pressure testing...")
    await asyncio.sleep(0.8)
    for a in top5:
        resilience = random.uniform(0.5, 0.95)
        a.fitness *= (0.7 + resilience * 0.3)
        api_calls += 3
    await emit("pressure_test_complete", "post_evolution")

    await emit("refinement_started", "refinement")
    await status("Deep refining top solution...")
    await asyncio.sleep(0.8)
    await emit("refinement_complete", "refinement")
    api_calls += 7

    agents.sort(key=lambda a: a.fitness, reverse=True)
    champion = agents[0]
    generations_data = []
    for g in range(1, gens + 1):
        generations_data.append({
            "generation": g,
            "best_fitness": round(0.5 + g * 0.1 + random.uniform(0, 0.1), 3),
            "avg_fitness": round(0.4 + g * 0.08 + random.uniform(0, 0.05), 3),
            "alive_count": total,
        })

    evolved_answer = (
        f"Final EvoHive mock answer for: {config.get('problem', 'Mock evolution test')}\n\n"
        "1. Define a narrow target segment and position the offer around a measurable outcome.\n"
        "2. Start with a low-friction entry tier to maximize adoption and feedback volume.\n"
        "3. Reserve premium pricing for automation depth, team workflows, and advanced reliability.\n"
        "4. Use battle-tested proof points, benchmarks, and before/after comparisons in the final pitch.\n"
        "5. Keep iterating based on real user objections, not just internal assumptions."
    )
    lineage_graph = {
        "nodes": [
            {"id": agent.id, "generation": agent.gen, "fitness": round(agent.fitness, 3)}
            for agent in agents[:5]
        ],
        "edges": [],
        "summary": {
            "node_count": min(5, len(agents)),
            "edge_count": 0,
            "finalist_count": min(5, len(agents)),
        },
    }
    verification_report = build_verification_report(
        problem=config.get("problem", "Mock evolution test"),
        final_answer=evolved_answer,
        lineage_graph=lineage_graph,
    )
    claim_verification_report = build_claim_verification_report(
        verification_report=verification_report,
        search_context="",
    )
    answer_graph = build_answer_graph(
        problem=config.get("problem", "Mock evolution test"),
        final_answer=evolved_answer,
        lineage_graph=lineage_graph,
        verification_report=verification_report,
        top_solutions=[a.to_dict() for a in agents[:5]],
    )
    mock_duration = max(0.0, time.perf_counter() - mock_started_at)
    mock_input_tokens = api_calls * 100
    mock_output_tokens = api_calls * 50
    mock_total_tokens = mock_input_tokens + mock_output_tokens
    resource_report = {
        "version": "resource-report.v1",
        "duration_sec": round(mock_duration, 4),
        "llm_calls": api_calls,
        "input_tokens": mock_input_tokens,
        "output_tokens": mock_output_tokens,
        "total_tokens": mock_total_tokens,
        "estimated_cost": round(api_calls * 0.001, 3),
        "tokens_per_sec": round(mock_total_tokens / mock_duration, 2) if mock_duration > 0 else 0.0,
        "cost_per_1k_tokens": round((api_calls * 0.001) / (mock_total_tokens / 1000), 6)
        if mock_total_tokens > 0 else 0.0,
        "phases": {},
        "generations": generations_data,
    }
    token_budget_plan = {
        "version": "token-budget-plan.v1",
        "reasoning_effort": reasoning_effort,
        "total_token_budget": max(4000, total * gens * 1500),
        "phase_budgets": {"mock": max(4000, total * gens * 1500)},
        "policy": {"soft_limit": True, "recommend_stop_at": 0.9, "hard_stop_at": 1.15},
    }
    token_budget_report = build_token_budget_report(
        plan=token_budget_plan,
        resource_report={
            **resource_report,
            "phases": {
                "mock": {
                    "total_tokens": mock_total_tokens,
                    "input_tokens": mock_input_tokens,
                    "output_tokens": mock_output_tokens,
                    "calls": api_calls,
                    "cost": round(api_calls * 0.001, 3),
                }
            },
        },
    )
    trajectory_log = [
        {
            "seq": 1,
            "phase": "mock",
            "actor": "mock_engine",
            "action": "simulate_evolution",
            "input_summary": f"total={total}, gens={gens}",
            "output_summary": f"api_calls={api_calls}, tokens={mock_total_tokens}",
            "metrics": {"reasoning_effort": reasoning_effort},
        },
        {
            "seq": 2,
            "phase": "answer_graph",
            "actor": "graph_builder",
            "action": "build_answer_graph",
            "input_summary": "mock result",
            "output_summary": f"nodes={answer_graph['summary']['node_count']}",
            "metrics": answer_graph["summary"],
        },
    ]
    trajectory_summary = {
        "event_count": len(trajectory_log),
        "phases": {"mock": 1, "answer_graph": 1},
        "actors": {"mock_engine": 1, "graph_builder": 1},
    }
    trajectory_replay = build_trajectory_replay(trajectory_log)

    await emit("run_complete", "complete",
                total_api_calls=api_calls, estimated_cost=round(api_calls * 0.001, 3),
                duration=round(time.time() % 1000, 1))

    results_payload = {
        "problem": config.get("problem", "Mock evolution test"),
        "champion": champion.to_dict(),
        "total_api_calls": api_calls,
        "estimated_cost": round(api_calls * 0.001, 3),
        "resource_report": resource_report,
        "token_budget_plan": token_budget_plan,
        "token_budget_report": token_budget_report,
        "token_budget_events": [],
        "trajectory_log": trajectory_log,
        "trajectory_summary": trajectory_summary,
        "trajectory_replay": trajectory_replay,
        "event_count": len(all_events),
        "generations_data": generations_data,
        "top5": [a.to_dict() for a in agents[:5]],
        "evolved_answer": evolved_answer,
        "lineage_graph": lineage_graph,
        "verification_report": verification_report,
        "claim_verification_report": claim_verification_report,
        "answer_graph": answer_graph,
    }
    artifact = await _persist_run_artifact_async(
        run_id=f"mock-{uuid.uuid4().hex[:8]}",
        mode="mock",
        config=safe_config,
        results=results_payload,
        events=all_events,
    )
    results_payload["structured_result"] = artifact["structured_result"]
    results_payload["telemetry"] = artifact["telemetry"]
    results_payload["artifact"] = {k: v for k, v in artifact.items() if k != "structured_result"}

    await ws.send_json({
        "type": "war_complete",
        "results": results_payload,
    })


# ── Detect real mode: check if any API keys are set ──
def _has_real_keys() -> bool:
    """Check if any LLM API keys are present in environment."""
    key_vars = [
        "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    ]
    return any(os.environ.get(k) for k in key_vars)


def _has_real_keys_for_config(config: dict | None) -> bool:
    return _has_real_keys() or bool(_normalize_api_keys((config or {}).get("api_keys")))


# ── WebSocket endpoint ──

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_json(_make_ws_error_payload(ClientMessageError("Malformed JSON message.")))
                continue
            if not isinstance(msg, dict):
                await ws.send_json(_make_ws_error_payload(ClientMessageError("WebSocket message must be a JSON object.")))
                continue
            logger.info("WebSocket message received type=%s bytes=%s", msg.get("type"), len(data))

            if msg.get("type") == "start_war":
                config = msg.get("config", {})
                if config is None:
                    config = {}
                if not isinstance(config, dict):
                    await ws.send_json(_make_ws_error_payload(ClientMessageError("start_war config must be an object.")))
                    continue
                try:
                    real_mode = _has_real_keys_for_config(config)
                    if real_mode:
                        logger.info("Starting real evolution with config=%s", _redact_config(config))
                        await run_real_evolution(ws, config)
                    else:
                        logger.info("Starting mock evolution with config=%s", _redact_config(config))
                        await run_mock_evolution(ws, config)
                except Exception as e:
                    payload = _make_ws_error_payload(e, config)
                    logger.exception("Evolution error [%s]", payload["error_id"])
                    await ws.send_json(payload)

            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error")


if __name__ == "__main__":
    mode_str = "REAL" if _has_real_keys() else "MOCK"
    print(f"Starting EvoHive Backend Server on http://0.0.0.0:8080 [{mode_str} MODE]")
    print("API root: http://0.0.0.0:8080/")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
