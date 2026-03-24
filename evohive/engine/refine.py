"""深度扩写模块 — 将Top 1方案扩展为详尽的执行计划"""

from evohive.llm import call_llm
from evohive.prompts.refine_prompts import REFINE_SYSTEM, REFINE_USER


async def deep_refine(solution_content: str, problem: str, model: str) -> str:
    """将进化最优方案深度扩写为2000-3000字的完整执行计划。

    Args:
        solution_content: 进化产生的Top 1方案内容
        problem: 原始问题描述
        model: 使用的LLM模型

    Returns:
        扩写后的完整方案文本
    """
    return await call_llm(
        model=model,
        system_prompt=REFINE_SYSTEM,
        user_prompt=REFINE_USER.format(
            problem=problem,
            solution=solution_content,
        ),
        temperature=0.7,
        max_tokens=8000,
    )
