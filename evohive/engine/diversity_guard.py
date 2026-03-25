"""反同质化猎杀 + 新血注入"""

import asyncio
from evohive.models import Solution
from evohive.engine.genesis import generate_thinkers, generate_initial_solutions
from evohive.engine.judge import compute_population_similarity


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


def kill_homogeneous(
    new_solutions: list[Solution],
    existing_population: list[Solution],
    threshold: float = 0.6,
    elite_id: str | None = None,
    similarity_fn=None,
) -> tuple[list[Solution], list[Solution]]:
    """反同质化猎杀: 杀掉与现有种群过于相似的新方案

    Args:
        new_solutions: 新生成的子代方案
        existing_population: 现有种群（存活者）
        threshold: 相似度阈值，超过则杀掉
        elite_id: 精英ID，享受豁免
        similarity_fn: 自定义相似度函数 (text_a, text_b) -> float, 默认Jaccard

    Returns:
        (survivors, killed) - 存活的和被杀的方案
    """
    if similarity_fn is None:
        similarity_fn = _jaccard_similarity

    survivors = []
    killed = []
    reference = list(existing_population)

    for sol in new_solutions:
        if sol.id == elite_id:
            survivors.append(sol)
            reference.append(sol)
            continue

        is_too_similar = False
        for ref in reference:
            sim = similarity_fn(sol.content, ref.content)
            if sim > threshold:
                is_too_similar = True
                sol.is_alive = False
                sol.elimination_reason = f"反同质化猎杀 (相似度={sim:.2f} > {threshold})"
                killed.append(sol)
                break

        if not is_too_similar:
            survivors.append(sol)
            reference.append(sol)

    return survivors, killed


async def inject_fresh_blood(
    problem: str,
    model: str,
    count: int,
    generation: int,
    existing_population: list[Solution],
) -> list[Solution]:
    """新血注入: 生成全新的随机方案填充空位

    Args:
        problem: 问题描述
        model: LLM模型
        count: 需要注入的数量
        generation: 当前代数
        existing_population: 现有种群（用于确保新方案不重复）

    Returns:
        新生成的方案列表
    """
    if count <= 0:
        return []

    # Generate new thinkers with different personas
    thinkers = await generate_thinkers(problem, count, model)

    # Generate solutions
    new_solutions = await generate_initial_solutions(problem, thinkers)

    # Set generation
    for sol in new_solutions:
        sol.generation = generation

    return new_solutions


def should_inject_fresh_blood(
    population: list[Solution],
    generation: int,
    inject_interval: int = 2,
    similarity_threshold: float = 0.5,
    similarity_score: float | None = None,
) -> bool:
    """判断是否需要注入新血

    条件: 每N代检查一次，如果种群多样性下降则注入

    Args:
        population: 当前种群
        generation: 当前代数
        inject_interval: 检查间隔代数
        similarity_threshold: 相似度阈值
        similarity_score: 预计算的相似度分数（如来自async embedding），为None时使用Jaccard计算
    """
    if generation == 0:
        return False

    if generation % inject_interval != 0:
        return False

    # Use pre-computed similarity if provided, otherwise compute via Jaccard
    if similarity_score is not None:
        similarity = similarity_score
    else:
        similarity = compute_population_similarity(population)
    return similarity > similarity_threshold
