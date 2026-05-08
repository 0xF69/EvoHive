import json
import re
from pathlib import Path


def slug(text: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "run").strip().lower()).strip("-")
    return (value or "run")[:max_len].strip("-") or "run"


def split_answer_sections(answer: str) -> dict:
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


def build_structured_result(problem: str, answer: str, champion: dict, top5: list[dict]) -> dict:
    parsed = split_answer_sections(answer)
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


def build_phase_stats(events: list[dict]) -> dict:
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


def build_model_roster(config: dict, results: dict, providers_catalog: dict) -> dict:
    roster: dict[str, dict] = {}
    configured = config.get("providers") or []
    for provider in configured:
        provider_meta = providers_catalog.get(provider, {})
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
            "name": providers_catalog.get(provider, {}).get("name", provider),
            "configured": provider in configured,
            "models": [],
            "units": 0,
        })
        model = unit.get("model")
        if model and model not in entry["models"]:
            entry["models"].append(model)
        entry["units"] += 1
    return {"providers": list(roster.values())}


def build_replay_summary(events: list[dict]) -> list[dict]:
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


def build_run_telemetry(config: dict, results: dict, events: list[dict], providers_catalog: dict) -> dict:
    return {
        "event_count": len(events),
        "phase_stats": build_phase_stats(events),
        "model_roster": build_model_roster(config, results, providers_catalog),
        "generation_count": len(results.get("generations_data") or []),
        "api_calls": results.get("total_api_calls", 0),
        "estimated_cost": results.get("estimated_cost", 0),
    }


def format_artifact_report(run_id: str, mode: str, results: dict, structured: dict, telemetry: dict) -> str:
    lines = [
        f"# EvoHive Run {run_id}",
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


def persist_run_artifact(
    artifact_root: Path,
    providers_catalog: dict,
    run_id: str,
    mode: str,
    config: dict,
    results: dict,
    events: list[dict],
) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_dir = artifact_root / f"{run_id}-{slug(results.get('problem', 'run'))}"
    run_dir.mkdir(parents=True, exist_ok=True)

    structured = build_structured_result(
        results.get("problem", ""),
        results.get("evolved_answer", ""),
        results.get("champion", {}),
        results.get("top5", []),
    )
    telemetry = build_run_telemetry(config, results, events, providers_catalog)
    replay = build_replay_summary(events)

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
    report_path.write_text(format_artifact_report(run_id, mode, results, structured, telemetry), encoding="utf-8")

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
