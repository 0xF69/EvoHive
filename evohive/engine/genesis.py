"""种群初始化 + 初始Solution生成（支持多模型）"""

import asyncio
import random

from evohive.models import Thinker, ThinkerDNA, Solution
from evohive.llm import call_llm, call_llm_batch, extract_json
from evohive.prompts.thinker_genesis import (
    GENERATE_THINKERS_SYSTEM, GENERATE_THINKERS_USER,
    THINKER_SOLVE_SYSTEM, THINKER_SOLVE_USER,
)


async def generate_thinkers(
    problem: str,
    count: int,
    model: str,
) -> list[Thinker]:
    """生成多样化的Thinker角色（单模型版）"""
    prompt = GENERATE_THINKERS_USER.format(count=count, problem=problem)
    response = await call_llm(
        model=model,
        system_prompt=GENERATE_THINKERS_SYSTEM,
        user_prompt=prompt,
        temperature=0.9,
        max_tokens=3000,
    )

    data = extract_json(response)
    if not data or not isinstance(data, list):
        raise ValueError(f"无法解析Thinker角色列表: {response[:200]}")

    thinkers = []
    for i, item in enumerate(data[:count]):
        thinker = Thinker(
            id=f"thinker_{i:03d}",
            dna=ThinkerDNA(
                persona=item.get("persona", f"通用分析师{i}"),
                knowledge_bias=item.get("knowledge_bias", "综合分析"),
                constraint=item.get("constraint", "无特殊约束"),
            ),
            model=model,
        )
        thinkers.append(thinker)

    return thinkers


async def generate_thinkers_multi_model(
    problem: str,
    model_assignments: list[tuple[str, int]],
) -> list[Thinker]:
    """生成多样化的Thinker角色（多模型版）

    Args:
        problem: 用户问题
        model_assignments: [(model_name, count), ...] 每个模型生成count个Thinker

    Returns:
        所有模型生成的Thinker合并列表
    """
    # 每个模型并发生成各自的Thinker
    tasks = []
    for model, count in model_assignments:
        if count <= 0:
            continue
        tasks.append(_generate_thinkers_for_model(problem, model, count))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_thinkers = []
    global_idx = 0
    for result in results:
        if isinstance(result, Exception):
            # 某个模型失败不影响其他模型
            continue
        for thinker in result:
            thinker.id = f"thinker_{global_idx:03d}"
            global_idx += 1
            all_thinkers.append(thinker)

    return all_thinkers


async def _generate_thinkers_for_model(
    problem: str,
    model: str,
    count: int,
) -> list[Thinker]:
    """为单个模型生成Thinker角色"""
    prompt = GENERATE_THINKERS_USER.format(count=count, problem=problem)
    response = await call_llm(
        model=model,
        system_prompt=GENERATE_THINKERS_SYSTEM,
        user_prompt=prompt,
        temperature=0.9,
        max_tokens=3000,
    )

    data = extract_json(response)
    if not data or not isinstance(data, list):
        raise ValueError(f"模型 {model} 无法解析Thinker角色列表: {response[:200]}")

    thinkers = []
    for i, item in enumerate(data[:count]):
        thinker = Thinker(
            id=f"temp_{i:03d}",  # 会在合并时重新编号
            dna=ThinkerDNA(
                persona=item.get("persona", f"通用分析师{i}"),
                knowledge_bias=item.get("knowledge_bias", "综合分析"),
                constraint=item.get("constraint", "无特殊约束"),
            ),
            model=model,
        )
        thinkers.append(thinker)

    return thinkers


async def generate_initial_solutions(
    problem: str,
    thinkers: list[Thinker],
) -> list[Solution]:
    """每个Thinker并发生成一个Solution"""
    calls = []
    for thinker in thinkers:
        system = THINKER_SOLVE_SYSTEM.format(
            persona=thinker.dna.persona,
            knowledge_bias=thinker.dna.knowledge_bias,
            constraint=thinker.dna.constraint,
        )
        user = THINKER_SOLVE_USER.format(problem=problem)
        calls.append({
            "model": thinker.model,
            "system_prompt": system,
            "user_prompt": user,
            "temperature": 0.9,
            "max_tokens": 4000,
        })

    responses = await call_llm_batch(calls, max_concurrent=20)

    solutions = []
    for i, (thinker, response) in enumerate(zip(thinkers, responses)):
        if response.startswith("ERROR:"):
            content = f"[生成失败: {response}]"
        else:
            content = response

        solution = Solution(
            content=content,
            generation=0,
            thinker_id=thinker.id,
        )
        solutions.append(solution)

    return solutions
