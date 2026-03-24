"""基线生成"""

from evohive.llm import call_llm


BASELINE_SYSTEM = "你是一个资深的策略顾问。请给出具体、可执行的方案。"

BASELINE_USER = """请针对以下问题，给出你认为最好的完整方案。
要求具体、可执行、有创意。像写专业咨询报告一样，结构清晰、细节充分。

问题: {problem}"""


async def generate_baseline(problem: str, model: str) -> str:
    """用同一个模型直接回答问题，作为对照基线"""
    return await call_llm(
        model=model,
        system_prompt=BASELINE_SYSTEM,
        user_prompt=BASELINE_USER.format(problem=problem),
        temperature=0.7,
        max_tokens=4000,
    )
