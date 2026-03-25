"""极端压力测试"""

import asyncio
import random
from evohive.models import Solution
from evohive.llm import call_llm, extract_json
from evohive.prompts.pressure_prompts import (
    PRESSURE_SYSTEM, PRESSURE_USER, PRESSURE_SCENARIOS,
)


async def pressure_test_solution(
    solution: Solution,
    problem: str,
    scenario: str,
    model: str,
) -> dict:
    """对单个方案施加一个极端压力场景

    Returns:
        {
            "solution_id": str,
            "scenario": str,
            "survives": bool,
            "impact_analysis": str,
            "broken_parts": list[str],
            "resilience_score": float,
        }
    """
    response = await call_llm(
        model=model,
        system_prompt=PRESSURE_SYSTEM,
        user_prompt=PRESSURE_USER.format(
            problem=problem,
            solution_content=solution.content,
            scenario=scenario,
        ),
        temperature=0.3,
        max_tokens=1000,
        json_mode=True,
    )

    data = extract_json(response)
    if not data:
        return {
            "solution_id": solution.id,
            "scenario": scenario,
            "survives": False,
            "impact_analysis": "评估失败",
            "broken_parts": [],
            "resilience_score": 0.5,
        }

    return {
        "solution_id": solution.id,
        "scenario": scenario,
        "survives": bool(data.get("survives", False)),
        "impact_analysis": data.get("impact_analysis", ""),
        "broken_parts": data.get("broken_parts", []),
        "resilience_score": float(data.get("resilience_score", 0.5)),
    }


async def pressure_test_batch(
    solutions: list[Solution],
    problem: str,
    model: str,
    n_scenarios: int = 3,
    top_n: int = 5,
) -> list[dict]:
    """对Top N方案进行极端压力测试

    Args:
        solutions: 按fitness排序的方案列表
        problem: 问题描述
        model: 使用的LLM模型
        n_scenarios: 每个方案测试几个场景
        top_n: 测试前N个方案

    Returns:
        列表，每个元素包含方案ID和综合韧性评分
    """
    targets = solutions[:top_n]

    # Select random scenarios for each solution
    sem = asyncio.Semaphore(15)

    tasks = []
    task_meta = []  # Track which solution each task belongs to

    for sol in targets:
        scenarios = random.sample(PRESSURE_SCENARIOS, min(n_scenarios, len(PRESSURE_SCENARIOS)))
        for scenario in scenarios:
            async def test_with_limit(s=sol, sc=scenario):
                async with sem:
                    return await pressure_test_solution(s, problem, sc, model)
            tasks.append(test_with_limit())
            task_meta.append(sol.id)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate by solution
    sol_results = {}
    for sol_id, result in zip(task_meta, results):
        if isinstance(result, dict):
            if sol_id not in sol_results:
                sol_results[sol_id] = []
            sol_results[sol_id].append(result)

    aggregated = []
    for sol in targets:
        test_results = sol_results.get(sol.id, [])
        if not test_results:
            aggregated.append({
                "solution_id": sol.id,
                "scenarios_tested": 0,
                "scenarios_survived": 0,
                "avg_resilience": 0.5,
                "details": [],
            })
            continue

        survived = sum(1 for r in test_results if r.get("survives", False))
        avg_resilience = sum(r.get("resilience_score", 0.5) for r in test_results) / len(test_results)

        aggregated.append({
            "solution_id": sol.id,
            "scenarios_tested": len(test_results),
            "scenarios_survived": survived,
            "avg_resilience": avg_resilience,
            "details": test_results,
        })

    return aggregated


def apply_pressure_scores(solutions: list[Solution], pressure_results: list[dict]):
    """将压力测试结果融合到fitness

    高韧性方案获得加分，低韧性方案降分。
    """
    resilience_map = {r["solution_id"]: r["avg_resilience"] for r in pressure_results}

    for sol in solutions:
        if sol.id in resilience_map:
            resilience = resilience_map[sol.id]
            # Resilience 0.5 = neutral, 1.0 = +5% fitness, 0.0 = -5% fitness
            adjustment = (resilience - 0.5) * 0.1
            sol.fitness = max(0.0, min(1.0, sol.fitness + adjustment))
