"""强制基因组合交叉重组"""

import asyncio
import random
from evohive.models import Solution, GeneRecord
from evohive.llm import call_llm, extract_json
from evohive.prompts.crossover_prompts import (
    EXTRACT_GENES_SYSTEM, EXTRACT_GENES_USER,
    COMBINE_GENES_SYSTEM, COMBINE_GENES_USER,
)


async def _extract_genes(solution: Solution, model: str) -> list[str]:
    """提取一个Solution的关键基因"""
    response = await call_llm(
        model=model,
        system_prompt=EXTRACT_GENES_SYSTEM,
        user_prompt=EXTRACT_GENES_USER.format(solution_content=solution.content),
        temperature=0.5,
        max_tokens=800,
    )

    data = extract_json(response)
    if data and isinstance(data, dict) and "genes" in data:
        return data["genes"][:5]

    # 回退: 把方案拆成几段作为基因
    lines = [l.strip() for l in solution.content.split("\n") if l.strip() and len(l.strip()) > 20]
    return lines[:4] if lines else ["方案核心内容"]


async def crossover_pair(
    parent_a: Solution,
    parent_b: Solution,
    problem: str,
    model: str,
    generation: int,
    memory_injection: str = "",
) -> Solution:
    """对一对parent执行强制基因组合

    Args:
        memory_injection: 遗传记忆文本，注入到组合prompt中避免重复犯错
    """
    # 1. 并发提取两个parent的基因
    genes_a, genes_b = await asyncio.gather(
        _extract_genes(parent_a, model),
        _extract_genes(parent_b, model),
    )

    # 2. 随机选择要组合的基因
    n_from_a = min(2, len(genes_a))
    n_from_b = min(2, len(genes_b))
    selected_a = random.sample(genes_a, n_from_a) if genes_a else []
    selected_b = random.sample(genes_b, n_from_b) if genes_b else []

    # 3. 强制组合
    genes_from_a_text = "\n".join(f"  - {g}" for g in selected_a)
    genes_from_b_text = "\n".join(f"  - {g}" for g in selected_b)

    # 构建user prompt，如果有遗传记忆则注入
    user_prompt = COMBINE_GENES_USER.format(
        problem=problem,
        genes_from_a=genes_from_a_text,
        genes_from_b=genes_from_b_text,
        parent_a_content=parent_a.content,
        parent_b_content=parent_b.content,
    )

    if memory_injection:
        user_prompt = user_prompt + "\n\n" + memory_injection

    response = await call_llm(
        model=model,
        system_prompt=COMBINE_GENES_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.8,
        max_tokens=4000,
    )

    # Content quality gate: discard obviously invalid outputs
    if not response or len(response.strip()) < 20 or response.strip().startswith("ERROR:"):
        # Fall back to the better parent's content
        response = parent_a.content if parent_a.fitness >= parent_b.fitness else parent_b.content

    return Solution(
        content=response,
        generation=generation,
        parent_a_id=parent_a.id,
        parent_b_id=parent_b.id,
        gene_record=GeneRecord(
            from_parent_a=selected_a,
            from_parent_b=selected_b,
        ),
    )


async def generate_next_generation(
    survivors: list[Solution],
    target_size: int,
    problem: str,
    model: str,
    generation: int,
    memory_injection: str = "",
) -> list[Solution]:
    """从存活者生成新一代

    Args:
        memory_injection: 遗传记忆文本，传递给交叉重组
    """
    # 存活者直接进入新一代
    new_generation = list(survivors)

    if len(survivors) < 2:
        return new_generation

    # 需要通过交叉产生的数量
    n_children = target_size - len(survivors)
    if n_children <= 0:
        return new_generation

    # 并发执行交叉重组
    sem = asyncio.Semaphore(20)

    async def do_crossover(_):
        async with sem:
            a, b = random.sample(survivors, 2)
            return await crossover_pair(a, b, problem, model, generation, memory_injection)

    children = await asyncio.gather(*[do_crossover(i) for i in range(n_children)])
    new_generation.extend(children)

    return new_generation
