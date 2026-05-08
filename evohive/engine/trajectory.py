"""Trajectory logging for auditable agentic runs."""

from __future__ import annotations

from datetime import UTC, datetime
import time


class TrajectoryLog:
    """Compact action log for replay, provenance, and cost attribution."""

    def __init__(self):
        self._started_at = time.perf_counter()
        self._events: list[dict] = []
        self._counter = 0

    def record(
        self,
        *,
        phase: str,
        action: str,
        actor: str = "system",
        input_summary: str = "",
        output_summary: str = "",
        metrics: dict | None = None,
    ) -> dict:
        self._counter += 1
        entry = {
            "seq": self._counter,
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "elapsed_sec": round(time.perf_counter() - self._started_at, 4),
            "phase": phase,
            "actor": actor,
            "action": action,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "metrics": metrics or {},
        }
        self._events.append(entry)
        return entry

    def to_list(self) -> list[dict]:
        return list(self._events)

    def summary(self) -> dict:
        phases: dict[str, int] = {}
        actors: dict[str, int] = {}
        for entry in self._events:
            phases[entry["phase"]] = phases.get(entry["phase"], 0) + 1
            actors[entry["actor"]] = actors.get(entry["actor"], 0) + 1
        return {
            "event_count": len(self._events),
            "phases": phases,
            "actors": actors,
        }
