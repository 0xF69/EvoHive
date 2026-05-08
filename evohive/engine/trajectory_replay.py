"""Build frontend-friendly trajectory replay payloads."""

from __future__ import annotations


def build_trajectory_replay(trajectory_log: list[dict]) -> dict:
    timeline = []
    phase_counts: dict[str, int] = {}
    for entry in sorted(trajectory_log or [], key=lambda item: item.get("seq", 0)):
        phase = entry.get("phase", "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        timeline.append({
            "seq": entry.get("seq", len(timeline) + 1),
            "time": entry.get("elapsed_sec", 0),
            "phase": phase,
            "actor": entry.get("actor", "system"),
            "action": entry.get("action", ""),
            "summary": entry.get("output_summary") or entry.get("input_summary") or "",
            "metrics": entry.get("metrics", {}),
            "checkpoint": phase_counts[phase] == 1,
        })
    return {
        "version": "trajectory-replay.v1",
        "step_count": len(timeline),
        "phase_count": len(phase_counts),
        "phase_counts": phase_counts,
        "timeline": timeline,
    }
