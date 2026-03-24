"""变异系统"""

import asyncio
import random
from evohive.models import Solution
from evohive.llm import call_llm
from evohive.prompts.mutation_strategies import (
    MUTATION_SYSTEM, MUTATION_STRATEGIES,
    ALTERNATIVE_AUDIENCES, RANDOM_CONSTRAINTS, SOURCE_INDUSTRIES,
)


def _fill_mutation_prompt(strategy_key: str, problem: str, content: str) -> str:
    """填充变异策略的prompt模板"""
    template = MUTATION_STRATEGIES[strategy_key]["prompt"]

    kwargs = {"problem": problem, "solution_content": content}

    if strategy_key == "audience_switch":
        kwargs["target"] = random.choice(ALTERNATIVE_AUDIENCES)
    elif strategy_key == "constraint_injection":
        kwargs["constraint"] = random.choice(RANDOM_CONSTRAINTS)
    elif strategy_key == "analogy_transfer":
        industry = random.choice(SOURCE_INDUSTRIES)
        kwargs["industry"] = industry

    return template.format(**kwargs)


async def maybe_mutate(
    solution: Solution,
    mutation_rate: float,
    problem: str,
    model: str,
) -> Solution:
    """以mutation_rate的概率对Solution施加变异"""
    if random.random() >= mutation_rate:
        return solution

    strategy_key = random.choice(list(MUTATION_STRATEGIES.keys()))
    strategy_name = MUTATION_STRATEGIES[strategy_key]["name"]

    prompt = _fill_mutation_prompt(strategy_key, problem, solution.content)

    response = await call_llm(
        model=model,
        system_prompt=MUTATION_SYSTEM,
        user_prompt=prompt,
        temperature=0.9,
        max_tokens=4000,
    )

    if response and len(response.strip()) >= 20 and not response.startswith("ERROR:"):
        solution.content = response
        solution.mutation_applied = strategy_name

    return solution


async def mutate_batch(
    solutions: list[Solution],
    mutation_rate: float,
    problem: str,
    model: str,
) -> list[Solution]:
    """批量执行变异（容错：单个变异失败不影响其他）"""
    sem = asyncio.Semaphore(20)

    async def do_mutate(sol):
        async with sem:
            try:
                return await maybe_mutate(sol, mutation_rate, problem, model)
            except Exception:
                # 变异失败时保留原始方案
                return sol

    return await asyncio.gather(*[do_mutate(s) for s in solutions])
