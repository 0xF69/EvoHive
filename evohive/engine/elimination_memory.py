"""淘汰反馈遗传记忆系统"""

import asyncio
from evohive.models import Solution
from evohive.llm import call_llm, extract_json
from evohive.prompts.elimination_prompts import (
    ELIMINATION_FEEDBACK_SYSTEM, ELIMINATION_FEEDBACK_USER,
)


class EvolutionMemory:
    """跨代遗传记忆 — 记录淘汰原因，传递给下一代"""

    def __init__(self, max_memory_generations: int = 2):
        self.memories: list[dict] = []  # {"generation": int, "reasons": list[str]}
        self.max_memory_generations = max_memory_generations

    def add_failures(self, generation: int, reasons: list[str]):
        """记录一代的失败原因"""
        if reasons:
            self.memories.append({
                "generation": generation,
                "reasons": reasons,
            })

    def get_active_memories(self, current_generation: int) -> list[str]:
        """获取当前仍活跃的记忆（仅保留最近N代）"""
        active = []
        for mem in self.memories:
            age = current_generation - mem["generation"]
            if age <= self.max_memory_generations:
                active.extend(mem["reasons"])
        return active

    def format_for_prompt(self, current_generation: int) -> str:
        """格式化为可注入prompt的文本"""
        memories = self.get_active_memories(current_generation)
        if not memories:
            return ""

        lines = ["前代方案的失败教训（你必须避免同样的问题）:"]
        for i, reason in enumerate(memories, 1):
            lines.append(f"  {i}. {reason}")
        return "\n".join(lines)


async def extract_failure_reasons(
    eliminated: list[Solution],
    problem: str,
    model: str,
    max_extract: int = 5,
) -> list[str]:
    """从被淘汰方案中提取失败原因

    Args:
        eliminated: 被淘汰的方案列表
        problem: 问题描述
        model: LLM模型
        max_extract: 最多提取几个方案的失败原因（控制成本）

    Returns:
        失败原因列表
    """
    if not eliminated:
        return []

    # 只取最差的几个
    to_analyze = sorted(eliminated, key=lambda s: s.fitness)[:max_extract]

    sem = asyncio.Semaphore(5)

    async def analyze_one(sol: Solution) -> list[str]:
        async with sem:
            response = await call_llm(
                model=model,
                system_prompt=ELIMINATION_FEEDBACK_SYSTEM,
                user_prompt=ELIMINATION_FEEDBACK_USER.format(
                    fitness=sol.fitness,
                    problem=problem,
                    solution_content=sol.content[:2000],  # Truncate to save tokens
                ),
                temperature=0.3,
                max_tokens=500,
                json_mode=True,
            )
            data = extract_json(response)
            if data and "failure_reasons" in data:
                return data["failure_reasons"][:3]
            return []

    results = await asyncio.gather(*[analyze_one(s) for s in to_analyze], return_exceptions=True)

    all_reasons = []
    for r in results:
        if isinstance(r, list):
            all_reasons.extend(r)

    # Deduplicate and limit
    seen = set()
    unique = []
    for reason in all_reasons:
        key = reason[:50]  # rough dedup
        if key not in seen:
            seen.add(key)
            unique.append(reason)

    return unique[:8]  # Max 8 reasons to avoid prompt bloat
