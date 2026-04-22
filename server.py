"""EvoHive WebSocket Server — bridges frontend visualization to core engine.

Mock mode: simulates the full 13-phase evolution pipeline using the real
event protocol, with fake LLM responses (no API keys needed).

Real mode: import the actual engine and run with real LLM calls.
"""

import asyncio
import json
import math
import os
import random
import re
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="EvoHive War Server")

ARTIFACT_ROOT = Path(__file__).parent / "evohive_runs"
DEFAULT_REAL_RUN_TIMEOUT_SEC = int(os.environ.get("EVOHIVE_WEB_TIMEOUT_SEC", "1800"))


def _slug(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "run").strip().lower()).strip("-")
    return (slug or "run")[:max_len].strip("-") or "run"


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


def _split_answer_sections(answer: str) -> dict:
    clean = (answer or "").strip()
    if not clean:
        return {"summary": "", "action_plan": [], "risks": [], "recommendation": "", "next_steps": []}

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean) if p.strip()]
    bullet_lines = []
    for line in clean.splitlines():
        stripped = line.strip()
        if re.match(r"^([-*]|\d+[.)])\s+", stripped):
            bullet_lines.append(re.sub(r"^([-*]|\d+[.)])\s+", "", stripped))

    summary = paragraphs[0] if paragraphs else clean
    risks = []
    actions = []
    for item in bullet_lines:
        lower = item.lower()
        if any(k in lower for k in ["risk", "constraint", "warning", "tradeoff", "caution", "limitation"]):
            risks.append(item)
        else:
            actions.append(item)

    if not actions and len(paragraphs) > 1:
        actions = paragraphs[1:4]
    if not risks:
        risk_sentences = re.findall(r"[^.!?]*(?:risk|constraint|warning|tradeoff|limitation)[^.!?]*[.!?]", clean, flags=re.I)
        risks = [s.strip() for s in risk_sentences[:3] if s.strip()]

    recommendation = actions[0] if actions else summary
    next_steps = actions[1:4] if len(actions) > 1 else paragraphs[1:4]

    return {
        "summary": summary,
        "action_plan": actions[:5],
        "risks": risks[:3],
        "recommendation": recommendation,
        "next_steps": next_steps[:3],
    }


def _build_structured_result(problem: str, answer: str, champion: dict, top5: list[dict]) -> dict:
    parsed = _split_answer_sections(answer)
    champion_label = champion.get("model") or champion.get("provider") or "unknown"
    alternatives = []
    for idx, unit in enumerate(top5[1:4], start=2):
        label = unit.get("model") or unit.get("provider") or unit.get("id") or "candidate"
        elo = round(unit.get("elo", 0), 1)
        fitness = round(unit.get("fitness", 0), 3)
        alternatives.append(f"#{idx} {label} finished behind the winner at ELO {elo} with fitness {fitness}.")
    winner_reason = (
        f"Top solution won because {champion_label} finished with the strongest final score "
        f"and survived the post-evolution pipeline."
    )
    return {
        "executive_summary": parsed["summary"] or f"EvoHive produced a best answer for: {problem}",
        "action_plan": parsed["action_plan"],
        "risks": parsed["risks"],
        "winner_reason": winner_reason,
        "recommendation": parsed["recommendation"] or winner_reason,
        "next_steps": parsed["next_steps"],
        "decision": f"Proceed with the {champion_label} champion response for: {problem}",
        "alternatives": alternatives,
    }


def _build_phase_stats(events: list[dict]) -> dict:
    phase_stats: dict[str, dict] = {}
    for event in events:
        phase = event.get("phase") or "unknown"
        ts = float(event.get("ts") or 0.0)
        stat = phase_stats.setdefault(phase, {"count": 0, "started_at": None, "ended_at": None})
        stat["count"] += 1
        if ts:
            stat["started_at"] = ts if stat["started_at"] is None else min(stat["started_at"], ts)
            stat["ended_at"] = ts if stat["ended_at"] is None else max(stat["ended_at"], ts)
    for stat in phase_stats.values():
        start = stat.get("started_at")
        end = stat.get("ended_at")
        stat["duration_sec"] = round(max(0.0, end - start), 3) if start and end else 0.0
    return phase_stats


def _build_model_roster(config: dict, results: dict) -> dict:
    roster: dict[str, dict] = {}
    configured = config.get("providers") or []
    for provider in configured:
        provider_meta = PROVIDERS.get(provider, {})
        roster[provider] = {
            "provider": provider,
            "name": provider_meta.get("name", provider),
            "configured": True,
            "models": [],
            "units": 0,
        }

    for unit in results.get("top5", []):
        provider = unit.get("provider") or "unknown"
        entry = roster.setdefault(provider, {
            "provider": provider,
            "name": PROVIDERS.get(provider, {}).get("name", provider),
            "configured": provider in configured,
            "models": [],
            "units": 0,
        })
        model = unit.get("model")
        if model and model not in entry["models"]:
            entry["models"].append(model)
        entry["units"] += 1
    return {"providers": list(roster.values())}


def _build_replay_summary(events: list[dict]) -> list[dict]:
    interesting = {
        "generation_started",
        "evaluation_complete",
        "elo_round_complete",
        "selection_complete",
        "crossover_complete",
        "mutation_complete",
        "red_team_attack",
        "debate_round",
        "refinement_complete",
        "run_complete",
    }
    replay = []
    for event in events:
        if event.get("event_type") not in interesting:
            continue
        replay.append({
            "ts": event.get("ts"),
            "event_type": event.get("event_type"),
            "phase": event.get("phase"),
            "data": event.get("data", {}),
        })
    return replay


def _build_run_telemetry(config: dict, results: dict, events: list[dict]) -> dict:
    return {
        "event_count": len(events),
        "phase_stats": _build_phase_stats(events),
        "model_roster": _build_model_roster(config, results),
        "generation_count": len(results.get("generations_data") or []),
        "api_calls": results.get("total_api_calls", 0),
        "estimated_cost": results.get("estimated_cost", 0),
    }


def _format_artifact_report(run_id: str, mode: str, results: dict, structured: dict, telemetry: dict) -> str:
    lines = [
        f"# EvoHive Web Run {run_id}",
        "",
        f"- Mode: {mode}",
        f"- Problem: {results.get('problem', '')}",
        f"- API Calls: {results.get('total_api_calls', 0)}",
        f"- Estimated Cost: ${results.get('estimated_cost', 0):.4f}",
        f"- Event Count: {telemetry.get('event_count', 0)}",
        "",
        "## Executive Summary",
        "",
        structured.get("executive_summary", ""),
        "",
        "## Recommendation",
        "",
        structured.get("recommendation", ""),
        "",
        "## Winner Reason",
        "",
        structured.get("winner_reason", ""),
        "",
        "## Action Plan",
        "",
    ]
    actions = structured.get("action_plan") or []
    lines.extend([f"- {item}" for item in actions] or ["- No structured action plan extracted."])
    lines.extend(["", "## Next Steps", ""])
    next_steps = structured.get("next_steps") or []
    lines.extend([f"- {item}" for item in next_steps] or ["- No explicit next steps extracted."])
    lines.extend(["", "## Risks", ""])
    risks = structured.get("risks") or []
    lines.extend([f"- {item}" for item in risks] or ["- No explicit risks extracted."])
    lines.extend(["", "## Alternatives", ""])
    alternatives = structured.get("alternatives") or []
    lines.extend([f"- {item}" for item in alternatives] or ["- No alternative summaries extracted."])
    lines.extend(["", "## Final Answer", "", results.get("evolved_answer", "") or ""])
    return "\n".join(lines).strip() + "\n"


def _persist_run_artifact(run_id: str, mode: str, config: dict, results: dict, events: list[dict]) -> dict:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = ARTIFACT_ROOT / f"{run_id}-{_slug(results.get('problem', 'run'))}"
    run_dir.mkdir(parents=True, exist_ok=True)

    structured = _build_structured_result(
        results.get("problem", ""),
        results.get("evolved_answer", ""),
        results.get("champion", {}),
        results.get("top5", []),
    )
    telemetry = _build_run_telemetry(config, results, events)
    replay = _build_replay_summary(events)

    payload = {
        "run_id": run_id,
        "mode": mode,
        "config": config,
        "results": results,
        "structured_result": structured,
        "telemetry": telemetry,
        "event_count": len(events),
    }

    json_path = run_dir / "run.json"
    events_path = run_dir / "events.json"
    replay_path = run_dir / "replay.json"
    telemetry_path = run_dir / "telemetry.json"
    report_path = run_dir / "report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    replay_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    telemetry_path.write_text(json.dumps(telemetry, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_format_artifact_report(run_id, mode, results, structured, telemetry), encoding="utf-8")

    return {
        "run_id": run_id,
        "dir": str(run_dir),
        "json_path": str(json_path),
        "events_path": str(events_path),
        "replay_path": str(replay_path),
        "telemetry_path": str(telemetry_path),
        "report_path": str(report_path),
        "structured_result": structured,
        "telemetry": telemetry,
    }


def _find_run_dir(run_id: str) -> Path | None:
    if not ARTIFACT_ROOT.exists():
        return None
    candidates = sorted(
        [path for path in ARTIFACT_ROOT.iterdir() if path.is_dir() and path.name.startswith(run_id)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _normalize_runtime_error(exc: Exception) -> dict:
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


# ── Serve frontend ──
FRONTEND_PATH = Path(__file__).parent / "evohive-war.html"
if not FRONTEND_PATH.exists():
    FRONTEND_PATH = Path(__file__).parent / "output" / "evohive-war.html"


@app.api_route("/", methods=["GET", "HEAD"])
async def index():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH, media_type="text/html")
    return HTMLResponse("<h1>evohive-war.html not found</h1>", status_code=404)


@app.get("/api/runs")
async def list_runs(limit: int = 20):
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
        payload = json.loads(run_json.read_text(encoding="utf-8"))
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
    return JSONResponse(content=json.loads(run_json.read_text(encoding="utf-8")))


@app.get("/api/runs/{run_id}/replay")
async def get_run_replay(run_id: str):
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        raise HTTPException(status_code=404, detail="run not found")
    replay_path = run_dir / "replay.json"
    if not replay_path.exists():
        raise HTTPException(status_code=404, detail="replay not found")
    return JSONResponse(content=json.loads(replay_path.read_text(encoding="utf-8")))


# ── Provider → LiteLLM model mapping ──
PROVIDER_MODEL_MAP = {
    "openai":      ["openai/gpt-4o-mini", "openai/gpt-4o"],
    "anthropic":   ["anthropic/claude-3-5-haiku-20241022", "anthropic/claude-sonnet-4-20250514"],
    "gemini":      ["gemini/gemini-2.0-flash", "gemini/gemini-2.5-flash"],
    "deepseek":    ["deepseek/deepseek-chat"],
    "groq":        ["groq/llama-3.3-70b-versatile"],
    "mistral":     ["mistral/mistral-large-latest"],
    "xai":         ["xai/grok-2"],
    "zhipuai":     ["zhipuai/glm-4-plus"],
    "siliconflow": ["siliconflow/Qwen/Qwen2.5-72B-Instruct"],
}

PROVIDERS = {
    "openai": {"color": "#e2e8f0", "name": "OpenAI", "models": ["gpt-4o", "gpt-4o-mini", "o3-mini"]},
    "anthropic": {"color": "#d4a574", "name": "Anthropic", "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]},
    "gemini": {"color": "#34d399", "name": "Google Gemini", "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]},
    "deepseek": {"color": "#4a9eff", "name": "DeepSeek", "models": ["deepseek-chat", "deepseek-reasoner"]},
    "groq": {"color": "#f97316", "name": "Groq", "models": ["llama-3.3-70b-versatile"]},
    "mistral": {"color": "#7c3aed", "name": "Mistral", "models": ["mistral-large-latest"]},
    "xai": {"color": "#ec4899", "name": "xAI (Grok)", "models": ["grok-2"]},
    "together": {"color": "#06b6d4", "name": "Together AI", "models": ["Llama-3.3-70B-Turbo"]},
    "fireworks": {"color": "#ef4444", "name": "Fireworks AI", "models": ["llama-v3.3-70b"]},
    "cohere": {"color": "#10b981", "name": "Cohere", "models": ["command-r-plus"]},
    "zhipuai": {"color": "#a855f7", "name": "ZhipuAI", "models": ["glm-4-plus"]},
    "siliconflow": {"color": "#22d3ee", "name": "SiliconFlow", "models": ["Qwen/Qwen2.5-72B-Instruct"]},
    "moonshot": {"color": "#f59e0b", "name": "Moonshot (Kimi)", "models": ["moonshot-v1-auto"]},
    "baichuan": {"color": "#ef4444", "name": "Baichuan", "models": ["Baichuan4"]},
    "yi": {"color": "#8b5cf6", "name": "Yi", "models": ["yi-large"]},
    "perplexity": {"color": "#3b82f6", "name": "Perplexity", "models": ["sonar-pro"]},
    "dashscope": {"color": "#f97316", "name": "DashScope", "models": ["qwen-max"]},
    "volcengine": {"color": "#22c55e", "name": "Volcengine", "models": ["doubao-pro-256k"]},
    "minimax": {"color": "#d946ef", "name": "Minimax", "models": ["MiniMax-Text-01"]},
}


# ═══════════════════════════════════════════════════════════════════
# REAL MODE — Bridge core EvoHive engine to WebSocket
# ═══════════════════════════════════════════════════════════════════

async def run_real_evolution(ws: WebSocket, config: dict):
    """Run the REAL EvoHive evolution engine, bridging events to WebSocket."""
    from evohive.models import EvolutionConfig
    from evohive.engine.evolution import run_evolution
    from evohive.engine.events import EventEmitter

    providers = config.get("providers", ["deepseek"])
    total = config.get("total", 15)
    gens = config.get("gens", 2)
    mode = config.get("mode", "fast")
    budget = float(config.get("budget", 0.5) or 0.5)
    run_timeout_sec = int(config.get("run_timeout_sec") or DEFAULT_REAL_RUN_TIMEOUT_SEC)
    enable_search = bool(config.get("enable_search", False))
    problem = config.get("problem",
        "Write a Python function `longest_palindrome(s: str) -> str` that "
        "returns the longest palindromic substring in s. "
        "Handle edge cases. Optimize for O(n^2) or better."
    )

    # Collect LiteLLM model strings from selected providers
    thinker_models = []
    for p in providers:
        if p in PROVIDER_MODEL_MAP:
            thinker_models.extend(PROVIDER_MODEL_MAP[p])
    if not thinker_models:
        thinker_models = ["deepseek/deepseek-chat"]

    # Use the first model as primary, rest as multi-model pool
    primary_model = thinker_models[0]

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
    )

    # Send war_started
    await ws.send_json({
        "type": "war_started",
        "config": {
            "total": evo_config.population_size,
            "gens": evo_config.generations,
            "mode": "REAL",
            "providers": providers,
            "problem": problem,
            "budget": budget,
            "run_timeout_sec": run_timeout_sec,
            "enable_search": enable_search,
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
    on_status(f"Starting REAL evolution with {len(thinker_models)} model(s): {', '.join(thinker_models)}")
    on_status(f"Problem: {problem[:100]}...")

    result = await asyncio.wait_for(
        run_evolution(
            config=evo_config,
            on_generation_complete=on_gen_complete,
            on_status=on_status,
            emitter=emitter,
            budget_limit=budget,
            save_results=False,
        ),
        timeout=run_timeout_sec,
    )

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
        "event_count": len(all_events),
        "generations_data": generations_data,
        "top5": top5_data if top5_data else [champion_data],
        "evolved_answer": result.refined_top_solution or (
            result.final_top_solutions[0].get("content", "") if result.final_top_solutions else ""
        ),
    }
    artifact = _persist_run_artifact(
        run_id=getattr(result, "id", uuid.uuid4().hex[:12]),
        mode="real",
        config={
            "providers": providers,
            "total": evo_config.population_size,
            "gens": evo_config.generations,
            "mode": mode,
            "budget": budget,
            "run_timeout_sec": run_timeout_sec,
            "enable_search": enable_search,
            "problem": problem,
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
    providers = config.get("providers", ["deepseek", "gemini"])
    total = config.get("total", 30)
    gens = config.get("gens", 3)
    mode = config.get("mode", "standard")

    SURVIVAL_RATE = 0.2
    ELITE_RATE = 0.05
    MUTATION_RATE = 0.3
    HOMOGENEITY_THRESHOLD = 0.6
    FRESH_BLOOD_INTERVAL = 2

    MockAgent._counter = 0
    api_calls = 0
    all_events = []

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
        "config": {"total": total, "gens": gens, "mode": mode, "providers": providers}
    })
    await asyncio.sleep(0.5)

    await emit("preflight_ok", "init", models=[f"{PROVIDERS[p]['name']}/{PROVIDERS[p]['models'][0]}" for p in providers if p in PROVIDERS])
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

    await emit("run_complete", "complete",
                total_api_calls=api_calls, estimated_cost=round(api_calls * 0.001, 3),
                duration=round(time.time() % 1000, 1))

    results_payload = {
        "problem": config.get("problem", "Mock evolution test"),
        "champion": champion.to_dict(),
        "total_api_calls": api_calls,
        "estimated_cost": round(api_calls * 0.001, 3),
        "event_count": len(all_events),
        "generations_data": generations_data,
        "top5": [a.to_dict() for a in agents[:5]],
        "evolved_answer": (
            f"Final EvoHive mock answer for: {config.get('problem', 'Mock evolution test')}\n\n"
            "1. Define a narrow target segment and position the offer around a measurable outcome.\n"
            "2. Start with a low-friction entry tier to maximize adoption and feedback volume.\n"
            "3. Reserve premium pricing for automation depth, team workflows, and advanced reliability.\n"
            "4. Use battle-tested proof points, benchmarks, and before/after comparisons in the final pitch.\n"
            "5. Keep iterating based on real user objections, not just internal assumptions."
        ),
    }
    artifact = _persist_run_artifact(
        run_id=f"mock-{uuid.uuid4().hex[:8]}",
        mode="mock",
        config=config,
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


# ── WebSocket endpoint ──

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            print(f"[WS] Received raw: {data[:200]}", flush=True)
            msg = json.loads(data)
            print(f"[WS] Parsed type: {msg.get('type')}", flush=True)

            if msg.get("type") == "start_war":
                config = msg.get("config", {})
                try:
                    real_mode = _has_real_keys()
                    if real_mode:
                        print(f"[REAL MODE] Starting evolution with config: {config}", flush=True)
                        await run_real_evolution(ws, config)
                    else:
                        print(f"[MOCK MODE] Starting evolution with config: {config}")
                        await run_mock_evolution(ws, config)
                except Exception as e:
                    tb = traceback.format_exc()
                    err = _normalize_runtime_error(e)
                    print(f"Evolution error: {tb}")
                    await ws.send_json({
                        "type": "war_error",
                        "error": err["error"],
                        "error_code": err["error_code"],
                        "retryable": err["retryable"],
                        "traceback": tb,
                    })

            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    mode_str = "REAL" if _has_real_keys() else "MOCK"
    print(f"Starting EvoHive War Server on http://0.0.0.0:8080 [{mode_str} MODE]")
    print(f"Frontend: {FRONTEND_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
