"""Checkpoint / Resume — 断点续跑

Saves intermediate evolution state after each generation,
allowing crashed runs to resume from the last completed generation.
"""

import glob
import json
import os
import re
from datetime import datetime
from typing import Any, Optional

from evohive.engine.logger import get_logger

_logger = get_logger("evohive.engine.checkpoint")

# Pattern: {run_id}_gen{generation}.ckpt.json
_CKPT_FILENAME_RE = re.compile(r"^(.+)_gen(\d+)\.ckpt\.json$")
_MAX_KEPT = 2  # Number of latest checkpoints to keep per run_id


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for datetime and other non-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_checkpoint(
    run_id: str,
    generation: int,
    population: list,
    evolution_memory: Any,
    config: Any,
    extra_state: Optional[dict] = None,
    checkpoint_dir: str = "evohive_checkpoints",
) -> str:
    """Save current evolution state as a checkpoint file.

    Uses atomic write (write to .tmp then os.rename) to prevent corruption.
    Keeps only the latest 2 checkpoints per run_id.

    Args:
        run_id: Unique identifier for this evolution run.
        generation: Current (completed) generation number.
        population: List of Solution objects (will be serialized via .model_dump()).
        evolution_memory: EvolutionMemory instance — its .memories list is saved.
        config: EvolutionConfig (will be serialized via .model_dump()).
        extra_state: Additional state dict (baseline_solution, generations_data,
                     cost tracker state, etc.).
        checkpoint_dir: Directory to store checkpoint files.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    filename = f"{run_id}_gen{generation}.ckpt.json"
    filepath = os.path.join(checkpoint_dir, filename)
    tmp_filepath = filepath + ".tmp"

    # Serialize population — skip large/transient fields
    serialized_population = []
    for sol in population:
        if hasattr(sol, "model_dump"):
            d = sol.model_dump()
        else:
            d = dict(sol) if isinstance(sol, dict) else {"content": str(sol)}
        # Remove embedding vectors to keep checkpoints small
        d.pop("embedding", None)
        serialized_population.append(d)

    # Serialize evolution memory
    memory_data = {
        "memories": getattr(evolution_memory, "memories", []),
        "max_memory_generations": getattr(evolution_memory, "max_memory_generations", 2),
    }

    # Serialize config
    if hasattr(config, "model_dump"):
        config_data = config.model_dump()
    else:
        config_data = dict(config) if isinstance(config, dict) else {}

    state = {
        "run_id": run_id,
        "generation": generation,
        "timestamp": datetime.now().isoformat(),
        "population": serialized_population,
        "evolution_memory": memory_data,
        "config": config_data,
        "extra_state": extra_state or {},
    }

    # Atomic write: write to .tmp, then rename
    with open(tmp_filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, default=_json_serializer)

    os.replace(tmp_filepath, filepath)

    # Clean up old checkpoints — keep only the latest _MAX_KEPT
    _cleanup_old_checkpoints(run_id, checkpoint_dir)

    return filepath


def load_checkpoint(
    run_id: str,
    checkpoint_dir: str = "evohive_checkpoints",
) -> Optional[dict]:
    """Load the latest checkpoint for a given run_id.

    Args:
        run_id: The run identifier to look up.
        checkpoint_dir: Directory where checkpoints are stored.

    Returns:
        Parsed state dict, or None if no checkpoint exists.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    # Find all checkpoints for this run_id
    candidates = []
    for fname in os.listdir(checkpoint_dir):
        m = _CKPT_FILENAME_RE.match(fname)
        if m and m.group(1) == run_id:
            gen = int(m.group(2))
            candidates.append((gen, fname))

    if not candidates:
        return None

    # Pick the one with the highest generation number
    candidates.sort(key=lambda x: x[0], reverse=True)
    latest_file = os.path.join(checkpoint_dir, candidates[0][1])

    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def list_checkpoints(
    checkpoint_dir: str = "evohive_checkpoints",
) -> list[dict]:
    """List all available checkpoints with metadata.

    Args:
        checkpoint_dir: Directory where checkpoints are stored.

    Returns:
        List of dicts with keys: run_id, generation, timestamp, filename.
    """
    results: list[dict] = []

    if not os.path.isdir(checkpoint_dir):
        return results

    for fname in os.listdir(checkpoint_dir):
        m = _CKPT_FILENAME_RE.match(fname)
        if not m:
            continue

        run_id = m.group(1)
        generation = int(m.group(2))

        # Try to read timestamp from the file
        filepath = os.path.join(checkpoint_dir, fname)
        timestamp = None
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            timestamp = data.get("timestamp")
        except (json.JSONDecodeError, OSError):
            pass

        results.append({
            "run_id": run_id,
            "generation": generation,
            "timestamp": timestamp,
            "filename": fname,
        })

    results.sort(key=lambda r: (r["run_id"], r["generation"]))
    return results


def cleanup_checkpoints(
    run_id: str,
    checkpoint_dir: str = "evohive_checkpoints",
) -> int:
    """Remove all checkpoint files for a completed run.

    Args:
        run_id: The run identifier whose checkpoints should be removed.
        checkpoint_dir: Directory where checkpoints are stored.

    Returns:
        Number of files removed.
    """
    if not os.path.isdir(checkpoint_dir):
        return 0

    removed = 0
    for fname in os.listdir(checkpoint_dir):
        m = _CKPT_FILENAME_RE.match(fname)
        if m and m.group(1) == run_id:
            filepath = os.path.join(checkpoint_dir, fname)
            try:
                os.remove(filepath)
                removed += 1
            except OSError:
                pass

    return removed


def _cleanup_old_checkpoints(
    run_id: str,
    checkpoint_dir: str,
) -> None:
    """Keep only the latest _MAX_KEPT checkpoints for a run_id."""
    candidates = []
    for fname in os.listdir(checkpoint_dir):
        m = _CKPT_FILENAME_RE.match(fname)
        if m and m.group(1) == run_id:
            gen = int(m.group(2))
            candidates.append((gen, fname))

    if len(candidates) <= _MAX_KEPT:
        return

    # Sort by generation descending; remove everything past the first _MAX_KEPT
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _, fname in candidates[_MAX_KEPT:]:
        filepath = os.path.join(checkpoint_dir, fname)
        try:
            os.remove(filepath)
        except OSError:
            pass
