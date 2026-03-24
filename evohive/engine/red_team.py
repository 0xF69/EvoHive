"""红队攻击系统 — 专职找方案致命缺陷"""

import asyncio
from evohive.models import Solution
from evohive.llm import call_llm, extract_json
from evohive.prompts.red_team_prompts import RED_TEAM_SYSTEM, RED_TEAM_USER


async def red_team_attack(
    solution: Solution,
    problem: str,
    model: str,
) -> dict:
    """对单个方案发起红队攻击

    Returns:
        {
            "solution_id": str,
            "attacks": [{"target": str, "attack": str, "severity": int}],
            "overall_vulnerability": float,
        }
    """
    response = await call_llm(
        model=model,
        system_prompt=RED_TEAM_SYSTEM,
        user_prompt=RED_TEAM_USER.format(
            problem=problem,
            solution_content=solution.content,
        ),
        temperature=0.5,
        max_tokens=1500,
        json_mode=True,
    )

    data = extract_json(response)
    if not data:
        return {
            "solution_id": solution.id,
            "attacks": [],
            "overall_vulnerability": 0.5,
        }

    attacks = data.get("attacks", [])
    vulnerability = float(data.get("overall_vulnerability", 0.5))

    return {
        "solution_id": solution.id,
        "attacks": attacks,
        "overall_vulnerability": min(1.0, max(0.0, vulnerability)),
    }


async def red_team_batch(
    solutions: list[Solution],
    problem: str,
    models: list[str],
    top_n: int = 5,
) -> list[dict]:
    """对Top N方案执行红队攻击

    每个方案被每个红队模型各攻击一次，取平均脆弱度。

    Args:
        solutions: 按fitness排序的方案列表
        problem: 问题描述
        models: 红队模型列表
        top_n: 攻击前N个方案

    Returns:
        攻击结果列表，每个元素包含方案ID、攻击详情、综合脆弱度
    """
    targets = solutions[:top_n]

    sem = asyncio.Semaphore(8)

    async def attack_with_limit(sol, model):
        async with sem:
            return await red_team_attack(sol, problem, model)

    # Each target attacked by each model
    tasks = []
    for sol in targets:
        for model in models:
            tasks.append(attack_with_limit(sol, model))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate by solution
    sol_attacks = {}  # sol_id -> list of attack results
    for r in results:
        if isinstance(r, dict):
            sid = r["solution_id"]
            if sid not in sol_attacks:
                sol_attacks[sid] = []
            sol_attacks[sid].append(r)

    aggregated = []
    for sol in targets:
        attack_results = sol_attacks.get(sol.id, [])
        if not attack_results:
            aggregated.append({
                "solution_id": sol.id,
                "attacks": [],
                "overall_vulnerability": 0.5,
            })
            continue

        # Merge all attacks
        all_attacks = []
        total_vuln = 0.0
        for ar in attack_results:
            all_attacks.extend(ar.get("attacks", []))
            total_vuln += ar.get("overall_vulnerability", 0.5)

        avg_vuln = total_vuln / len(attack_results)

        aggregated.append({
            "solution_id": sol.id,
            "attacks": all_attacks,
            "overall_vulnerability": avg_vuln,
        })

    return aggregated


def apply_red_team_scores(solutions: list[Solution], attack_results: list[dict]):
    """根据红队攻击结果调整fitness

    能扛住攻击的方案加分，脆弱的方案降分。
    """
    vuln_map = {r["solution_id"]: r["overall_vulnerability"] for r in attack_results}

    for sol in solutions:
        if sol.id in vuln_map:
            vulnerability = vuln_map[sol.id]
            # Resilience bonus: low vulnerability = bonus, high vulnerability = penalty
            # vulnerability 0.0 -> +10% fitness
            # vulnerability 0.5 -> no change
            # vulnerability 1.0 -> -10% fitness
            adjustment = (0.5 - vulnerability) * 0.2
            sol.fitness = max(0.0, min(1.0, sol.fitness + adjustment))
