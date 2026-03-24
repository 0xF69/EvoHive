"""强制辩论淘汰赛"""

import asyncio
import math
import random
from evohive.models import Solution
from evohive.llm import call_llm, extract_json
from evohive.prompts.debate_prompts import (
    DEBATE_ATTACK_SYSTEM, DEBATE_ATTACK_USER,
    DEBATE_DEFEND_SYSTEM, DEBATE_DEFEND_USER,
    DEBATE_JUDGE_SYSTEM, DEBATE_JUDGE_USER,
)


async def _attack(my_solution: str, opponent_solution: str, problem: str, model: str) -> list[str]:
    """一方攻击另一方"""
    response = await call_llm(
        model=model,
        system_prompt=DEBATE_ATTACK_SYSTEM,
        user_prompt=DEBATE_ATTACK_USER.format(
            my_solution=my_solution,
            opponent_solution=opponent_solution,
            problem=problem,
        ),
        temperature=0.5,
        max_tokens=1000,
        json_mode=True,
    )
    data = extract_json(response)
    if data and "attacks" in data:
        return data["attacks"][:5]
    return ["无法生成有效攻击"]


async def _defend(my_solution: str, attacks_text: str, problem: str, model: str) -> list[str]:
    """一方防守"""
    response = await call_llm(
        model=model,
        system_prompt=DEBATE_DEFEND_SYSTEM,
        user_prompt=DEBATE_DEFEND_USER.format(
            my_solution=my_solution,
            attacks_text=attacks_text,
            problem=problem,
        ),
        temperature=0.5,
        max_tokens=1000,
        json_mode=True,
    )
    data = extract_json(response)
    if data and "defenses" in data:
        return data["defenses"][:5]
    return ["无法生成有效防守"]


async def _judge_debate(
    solution_a: str, solution_b: str,
    a_attacks: list[str], a_defenses: list[str],
    b_attacks: list[str], b_defenses: list[str],
    problem: str, model: str,
) -> str:
    """裁判判定辩论胜者，返回 "A" 或 "B" """
    response = await call_llm(
        model=model,
        system_prompt=DEBATE_JUDGE_SYSTEM,
        user_prompt=DEBATE_JUDGE_USER.format(
            problem=problem,
            solution_a=solution_a[:1500],
            solution_b=solution_b[:1500],
            a_attacks="\n".join(f"- {a}" for a in a_attacks),
            b_attacks="\n".join(f"- {b}" for b in b_attacks),
            a_defenses="\n".join(f"- {d}" for d in a_defenses),
            b_defenses="\n".join(f"- {d}" for d in b_defenses),
        ),
        temperature=0.3,
        max_tokens=500,
        json_mode=True,
    )
    data = extract_json(response)
    if data and "winner" in data:
        return data["winner"]
    return random.choice(["A", "B"])


async def debate_pair(
    sol_a: Solution,
    sol_b: Solution,
    problem: str,
    thinker_model: str,
    judge_model: str,
) -> dict:
    """两个方案完整辩论一轮

    Returns:
        {"winner_id": str, "loser_id": str, "a_attacks": list, "b_attacks": list, ...}
    """
    # Phase 1: Both sides attack (concurrent)
    a_attacks, b_attacks = await asyncio.gather(
        _attack(sol_a.content, sol_b.content, problem, thinker_model),
        _attack(sol_b.content, sol_a.content, problem, thinker_model),
    )

    # Phase 2: Both sides defend (concurrent)
    a_attacks_text = "\n".join(f"- {a}" for a in b_attacks)  # A defends against B's attacks
    b_attacks_text = "\n".join(f"- {a}" for a in a_attacks)  # B defends against A's attacks

    a_defenses, b_defenses = await asyncio.gather(
        _defend(sol_a.content, a_attacks_text, problem, thinker_model),
        _defend(sol_b.content, b_attacks_text, problem, thinker_model),
    )

    # Phase 3: Judge decides
    winner_label = await _judge_debate(
        sol_a.content, sol_b.content,
        a_attacks, a_defenses,
        b_attacks, b_defenses,
        problem, judge_model,
    )

    if winner_label == "A":
        winner_id, loser_id = sol_a.id, sol_b.id
    else:
        winner_id, loser_id = sol_b.id, sol_a.id

    return {
        "winner_id": winner_id,
        "loser_id": loser_id,
        "a_attacks": a_attacks,
        "b_attacks": b_attacks,
        "a_defenses": a_defenses,
        "b_defenses": b_defenses,
    }


async def debate_tournament(
    solutions: list[Solution],
    problem: str,
    thinker_model: str,
    judge_models: list[str],
    top_n: int = 5,
) -> dict[str, float]:
    """辩论淘汰赛: Top N方案两两辩论，用Elo排名

    Args:
        solutions: 按fitness排序的方案列表
        problem: 问题描述
        thinker_model: 辩论用的模型
        judge_models: 裁判模型列表
        top_n: 参与辩论的方案数

    Returns:
        Dict mapping solution_id to debate Elo rating
    """
    targets = solutions[:top_n]
    if len(targets) < 2:
        return {s.id: 1500.0 for s in targets}

    # Elo ratings
    ratings = {s.id: 1500.0 for s in targets}

    # Generate pairs
    pairs = []
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            pairs.append((targets[i], targets[j]))

    # Run debates with each judge model
    sem = asyncio.Semaphore(3)  # Debates are expensive, limit concurrency

    async def debate_with_limit(sol_a, sol_b, judge_model):
        async with sem:
            return await debate_pair(sol_a, sol_b, problem, thinker_model, judge_model)

    tasks = []
    for judge_model in judge_models:
        for sol_a, sol_b in pairs:
            tasks.append(debate_with_limit(sol_a, sol_b, judge_model))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Update Elo ratings
    for result in results:
        if isinstance(result, Exception) or not isinstance(result, dict):
            continue

        winner_id = result["winner_id"]
        loser_id = result["loser_id"]

        if winner_id not in ratings or loser_id not in ratings:
            continue

        expected_w = 1.0 / (1.0 + math.pow(10, (ratings[loser_id] - ratings[winner_id]) / 400))
        expected_l = 1.0 - expected_w

        ratings[winner_id] += 32 * (1.0 - expected_w)
        ratings[loser_id] += 32 * (0.0 - expected_l)

    return ratings


def apply_debate_scores(solutions: list[Solution], debate_ratings: dict[str, float]):
    """将辩论Elo分数融合到fitness中"""
    if not debate_ratings:
        return

    min_r = min(debate_ratings.values())
    max_r = max(debate_ratings.values())
    r_range = max_r - min_r if max_r > min_r else 1.0

    for sol in solutions:
        if sol.id in debate_ratings:
            normalized = (debate_ratings[sol.id] - min_r) / r_range
            # Blend: 70% existing fitness + 30% debate score
            sol.fitness = sol.fitness * 0.7 + normalized * 0.3
