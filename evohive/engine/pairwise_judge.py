"""Pairwise对比评审 + Elo Rating系统"""

import asyncio
import math
import random
from evohive.models import Solution
from evohive.llm import call_llm, extract_json
from evohive.prompts.pairwise_prompts import PAIRWISE_SYSTEM, PAIRWISE_USER


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A"""
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400))


def _elo_update(rating: float, expected: float, actual: float, k: float = 32) -> float:
    """Update Elo rating"""
    return rating + k * (actual - expected)


async def pairwise_compare(
    sol_a: Solution,
    sol_b: Solution,
    problem: str,
    model: str,
    dimensions_text: str,
    debias: bool = True,
) -> dict:
    """Compare two solutions using a single judge model.

    Position bias mitigation:
    当 debias=True 时，会进行两次比较 (A-B 和 B-A)，
    只有当两次结果一致时才确认胜者，否则视为平局 (confidence=0.5)。
    这消除了模型倾向于选择 "A" 或 "B" 位置的偏差。

    Returns:
        {"winner_id": str, "loser_id": str, "reason": str, "confidence": float}
    """
    async def _single_compare(first: Solution, second: Solution) -> dict:
        """单次比较，返回 winner label ('A' or 'B')"""
        system = PAIRWISE_SYSTEM.format(dimensions_text=dimensions_text)
        user = PAIRWISE_USER.format(
            problem=problem,
            solution_a=first.content[:3000],
            solution_b=second.content[:3000],
        )
        response = await call_llm(
            model=model,
            system_prompt=system,
            user_prompt=user,
            temperature=0.3,
            max_tokens=800,
            json_mode=True,
        )
        data = extract_json(response)
        if not data:
            return {"winner": "A", "confidence": 0.5, "reason": "评审解析失败"}
        return {
            "winner": data.get("winner", "A"),
            "confidence": float(data.get("confidence", 0.7)),
            "reason": data.get("reason", ""),
        }

    if not debias:
        # 旧逻辑: 单次比较 + 随机交换
        swapped = random.random() > 0.5
        if swapped:
            result = await _single_compare(sol_b, sol_a)
            # 翻转结果
            result["winner"] = "B" if result["winner"] == "A" else "A"
        else:
            result = await _single_compare(sol_a, sol_b)

        if result["winner"] == "A":
            return {"winner_id": sol_a.id, "loser_id": sol_b.id,
                    "reason": result["reason"], "confidence": result["confidence"]}
        else:
            return {"winner_id": sol_b.id, "loser_id": sol_a.id,
                    "reason": result["reason"], "confidence": result["confidence"]}

    # ── Debias 模式: 双向比较 ──
    # Round 1: A 在前, B 在后
    r1 = await _single_compare(sol_a, sol_b)
    # Round 2: B 在前, A 在后
    r2 = await _single_compare(sol_b, sol_a)

    # 判定: r1 中选的是谁? r2 中选的是谁?
    # r1.winner == "A" → sol_a wins round 1
    # r2.winner == "A" → sol_b wins round 2 (因为 sol_b 在 A 位置)
    r1_winner_id = sol_a.id if r1["winner"] == "A" else sol_b.id
    r2_winner_id = sol_b.id if r2["winner"] == "A" else sol_a.id

    if r1_winner_id == r2_winner_id:
        # 两轮结果一致 → 高置信度
        winner_id = r1_winner_id
        loser_id = sol_b.id if winner_id == sol_a.id else sol_a.id
        avg_confidence = (r1["confidence"] + r2["confidence"]) / 2
        return {
            "winner_id": winner_id,
            "loser_id": loser_id,
            "reason": r1["reason"],
            "confidence": min(avg_confidence + 0.1, 1.0),  # 一致性加分
        }
    else:
        # 两轮结果矛盾 → 位置偏差, 随机选但低置信度
        winner = random.choice([sol_a, sol_b])
        loser = sol_b if winner == sol_a else sol_a
        return {
            "winner_id": winner.id,
            "loser_id": loser.id,
            "reason": "两轮比较结果矛盾(位置偏差)，视为平局",
            "confidence": 0.5,
        }


async def run_elo_tournament(
    solutions: list[Solution],
    problem: str,
    judge_models: list[str],
    dimensions_text: str,
    rounds_per_pair: int = 1,
) -> dict[str, float]:
    """Run a full Elo tournament with multiple judge models.

    Each pair is compared by each judge model. Elo ratings are computed
    from the aggregated results.

    Args:
        solutions: List of solutions to rank
        problem: The problem statement
        judge_models: List of judge model identifiers
        dimensions_text: Text describing evaluation dimensions
        rounds_per_pair: How many times each pair is compared per model

    Returns:
        Dict mapping solution_id to Elo rating (initial 1500)
    """
    if len(solutions) < 2:
        return {s.id: 1500.0 for s in solutions}

    # Initialize Elo ratings
    ratings = {s.id: 1500.0 for s in solutions}
    sol_map = {s.id: s for s in solutions}

    # Generate all pairs
    pairs = []
    for i in range(len(solutions)):
        for j in range(i + 1, len(solutions)):
            pairs.append((solutions[i], solutions[j]))

    # For each judge model, compare all pairs
    sem = asyncio.Semaphore(10)

    async def compare_with_limit(sol_a, sol_b, model):
        async with sem:
            return await pairwise_compare(sol_a, sol_b, problem, model, dimensions_text)

    all_tasks = []
    for model in judge_models:
        for _ in range(rounds_per_pair):
            for sol_a, sol_b in pairs:
                all_tasks.append(compare_with_limit(sol_a, sol_b, model))

    results = await asyncio.gather(*all_tasks, return_exceptions=True)

    # Update Elo ratings
    for result in results:
        if isinstance(result, Exception) or not isinstance(result, dict):
            continue

        winner_id = result["winner_id"]
        loser_id = result["loser_id"]

        if winner_id not in ratings or loser_id not in ratings:
            continue

        expected_winner = _elo_expected(ratings[winner_id], ratings[loser_id])
        expected_loser = _elo_expected(ratings[loser_id], ratings[winner_id])

        ratings[winner_id] = _elo_update(ratings[winner_id], expected_winner, 1.0)
        ratings[loser_id] = _elo_update(ratings[loser_id], expected_loser, 0.0)

    return ratings


def apply_elo_to_solutions(solutions: list[Solution], elo_ratings: dict[str, float]):
    """Normalize Elo ratings to 0-1 and apply as fitness scores."""
    if not elo_ratings:
        return

    min_elo = min(elo_ratings.values())
    max_elo = max(elo_ratings.values())
    elo_range = max_elo - min_elo if max_elo > min_elo else 1.0

    for sol in solutions:
        if sol.id in elo_ratings:
            normalized = (elo_ratings[sol.id] - min_elo) / elo_range
            # Blend with existing fitness if available
            if sol.fitness > 0:
                sol.fitness = sol.fitness * 0.4 + normalized * 0.6
            else:
                sol.fitness = normalized
            sol.raw_fitness = sol.fitness


async def run_tournament(
    solutions: list[Solution],
    problem: str,
    judge_models: list[str],
    dimensions_text: str,
    use_swiss: bool = True,
) -> dict[str, float]:
    """Unified tournament interface.

    Automatically chooses Swiss tournament for large populations
    and full pairwise for small ones.

    Args:
        solutions: Solutions to rank
        problem: Problem statement
        judge_models: Judge model list
        dimensions_text: Evaluation dimensions text
        use_swiss: Force Swiss tournament (default True for efficiency)

    Returns:
        Dict mapping solution_id to Elo rating
    """
    n = len(solutions)

    # For small populations, full pairwise is fine
    if n <= 10 and not use_swiss:
        return await run_elo_tournament(solutions, problem, judge_models, dimensions_text)

    # For larger populations, use Swiss tournament
    from evohive.engine.swiss_tournament import run_swiss_tournament
    return await run_swiss_tournament(solutions, problem, judge_models, dimensions_text)
