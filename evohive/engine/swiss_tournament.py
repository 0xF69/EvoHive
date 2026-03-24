"""瑞士轮锦标赛 — O(N log N) 的高效 Elo 排名

替代全配对 O(N²) 的 Elo 锦标赛，保持排名准确性的同时大幅降低 API 调用量。
原理: 每轮将战绩相近的选手配对，经过 log₂(N) 轮即可近似全排序。
"""

import asyncio
import math
import random
from evohive.models import Solution
from evohive.engine.pairwise_judge import pairwise_compare, _elo_expected, _elo_update


async def run_swiss_tournament(
    solutions: list[Solution],
    problem: str,
    judge_models: list[str],
    dimensions_text: str,
    max_rounds: int = 0,
) -> dict[str, float]:
    """Run a Swiss-system Elo tournament.

    Swiss system pairs players with similar records each round.
    After ~log2(N) rounds, ranking is approximately correct.

    Args:
        solutions: Solutions to rank
        problem: Problem statement
        judge_models: Judge models
        dimensions_text: Evaluation dimensions
        max_rounds: Max rounds (0 = auto: ceil(log2(N)) + 2)

    Returns:
        Dict mapping solution_id to Elo rating
    """
    n = len(solutions)
    if n < 2:
        return {s.id: 1500.0 for s in solutions}

    # Auto-determine rounds
    if max_rounds <= 0:
        max_rounds = min(max(3, math.ceil(math.log2(n)) + 2), n - 1)

    # Initialize ratings
    ratings = {s.id: 1500.0 for s in solutions}
    sol_map = {s.id: s for s in solutions}

    # Track matchup history to avoid repeats
    played_pairs: set[tuple[str, str]] = set()

    sem = asyncio.Semaphore(10)

    async def compare_with_limit(sol_a, sol_b, model):
        async with sem:
            return await pairwise_compare(sol_a, sol_b, problem, model, dimensions_text)

    for round_num in range(max_rounds):
        # Sort by rating (descending)
        sorted_ids = sorted(ratings.keys(), key=lambda x: ratings[x], reverse=True)

        # Pair adjacent players (Swiss pairing)
        pairs = []
        used = set()

        for i in range(len(sorted_ids)):
            if sorted_ids[i] in used:
                continue

            # Find best opponent: next unused player we haven't played yet
            best_j = None
            for j in range(i + 1, len(sorted_ids)):
                if sorted_ids[j] in used:
                    continue
                pair_key = tuple(sorted([sorted_ids[i], sorted_ids[j]]))
                if pair_key not in played_pairs:
                    best_j = j
                    break

            # If all played, allow repeat with nearest
            if best_j is None:
                for j in range(i + 1, len(sorted_ids)):
                    if sorted_ids[j] not in used:
                        best_j = j
                        break

            if best_j is not None:
                id_a, id_b = sorted_ids[i], sorted_ids[best_j]
                pairs.append((sol_map[id_a], sol_map[id_b]))
                used.add(id_a)
                used.add(id_b)
                pair_key = tuple(sorted([id_a, id_b]))
                played_pairs.add(pair_key)

        if not pairs:
            break

        # Run all pairs for this round with each judge (concurrent)
        tasks = []
        for sol_a, sol_b in pairs:
            # Use one random judge per pair per round for efficiency
            model = random.choice(judge_models)
            tasks.append(compare_with_limit(sol_a, sol_b, model))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Update Elo
        for result in results:
            if isinstance(result, Exception) or not isinstance(result, dict):
                continue

            winner_id = result["winner_id"]
            loser_id = result["loser_id"]

            if winner_id not in ratings or loser_id not in ratings:
                continue

            expected_w = _elo_expected(ratings[winner_id], ratings[loser_id])
            expected_l = _elo_expected(ratings[loser_id], ratings[winner_id])

            ratings[winner_id] = _elo_update(ratings[winner_id], expected_w, 1.0)
            ratings[loser_id] = _elo_update(ratings[loser_id], expected_l, 0.0)

    return ratings
