"""Executable Fitness Verification — 可执行适应度验证

When solutions contain code, actually RUN it against test cases
and combine the execution score with LLM-judge score.

Hybrid fitness = exec_weight * execution_score + (1 - exec_weight) * llm_judge_score

This catches the #1 weakness of LLM-as-judge: code that LOOKS correct
but doesn't actually work.
"""

import asyncio
import logging
from typing import Optional

from evohive.models import Solution

logger = logging.getLogger("evohive.executable_fitness")


async def evaluate_executable_fitness(
    solutions: list[Solution],
    test_cases: list[dict],
    *,
    exec_weight: float = 0.4,
    timeout_per_case: float = 5.0,
    max_memory_mb: int = 256,
) -> list[dict]:
    """Evaluate solutions by actually executing their code.

    For each solution:
    1. Extract code blocks from solution.content
    2. Run code against test_cases
    3. Compute execution_score (0-1 = passed/total)
    4. Blend with existing solution.fitness:
       new_fitness = exec_weight * exec_score + (1 - exec_weight) * solution.fitness
    5. Update solution.fitness in-place

    Args:
        solutions: Population of solutions to evaluate
        test_cases: List of {"input": str, "expected": str}
        exec_weight: Weight of execution score in final fitness (0-1)
        timeout_per_case: Timeout per test case in seconds
        max_memory_mb: Memory limit for execution

    Returns:
        List of execution report dicts for logging/display
    """
    # Lazy import to avoid circular imports and allow parallel creation of sandbox.py
    from evohive.engine.sandbox import (
        execute_with_test_cases,
        extract_code_blocks,
    )

    semaphore = asyncio.Semaphore(10)
    loop = asyncio.get_event_loop()

    async def _evaluate_one(solution: Solution) -> dict:
        """Evaluate a single solution's executable fitness."""
        report = {
            "solution_id": solution.id,
            "had_code": False,
            "exec_score": None,
            "passed": None,
            "total": len(test_cases),
            "error": None,
            "original_fitness": solution.fitness,
            "blended_fitness": None,
        }

        # Extract code blocks
        blocks = extract_code_blocks(solution.content)
        if not blocks:
            logger.debug(
                "Solution %s: no code blocks found, keeping LLM-judge score",
                solution.id,
            )
            return report

        report["had_code"] = True

        # Pick the longest Python block
        python_blocks = [b for b in blocks if b.get("language", "").lower() in ("python", "py", "")]
        if not python_blocks:
            # Fall back to all blocks if none explicitly tagged as python
            python_blocks = blocks

        code = max(python_blocks, key=lambda b: len(b.get("code", "")))["code"]

        try:
            # Run in executor since subprocess calls are blocking
            async with semaphore:
                result = await loop.run_in_executor(
                    None,
                    lambda: execute_with_test_cases(
                        code,
                        test_cases,
                        timeout_per_case=timeout_per_case,
                        max_memory_mb=max_memory_mb,
                    ),
                )

            exec_score = result.get("score", 0.0)
            passed = result.get("passed", 0)

            report["exec_score"] = exec_score
            report["passed"] = passed

            # Update solution execution metadata
            solution.execution_score = exec_score
            solution.execution_passed = passed
            solution.execution_total = len(test_cases)

            # Blend fitness: exec_weight * exec_score + (1 - exec_weight) * llm_judge_score
            original_fitness = solution.fitness
            blended = exec_weight * exec_score + (1 - exec_weight) * original_fitness
            solution.fitness = blended
            report["blended_fitness"] = blended

            logger.debug(
                "Solution %s: exec_score=%.3f, passed=%d/%d, "
                "fitness %.3f -> %.3f",
                solution.id,
                exec_score,
                passed,
                len(test_cases),
                original_fitness,
                blended,
            )

        except Exception as e:
            report["error"] = str(e)
            logger.warning(
                "Solution %s: execution failed: %s", solution.id, e
            )

        return report

    # Process all solutions concurrently (semaphore limits to 10)
    reports = await asyncio.gather(
        *[_evaluate_one(sol) for sol in solutions],
        return_exceptions=False,
    )

    # Log summary
    n_with_code = sum(1 for r in reports if r["had_code"])
    n_passed_all = sum(
        1 for r in reports
        if r["exec_score"] is not None and r["exec_score"] == 1.0
    )
    n_errors = sum(1 for r in reports if r["error"] is not None)

    logger.info(
        "Executable fitness: %d/%d solutions had code, "
        "%d passed all tests, %d had errors",
        n_with_code,
        len(solutions),
        n_passed_all,
        n_errors,
    )

    return reports


def has_executable_content(solution: Solution) -> bool:
    """Quick check if a solution likely contains executable code."""
    from evohive.engine.sandbox import extract_code_blocks

    blocks = extract_code_blocks(solution.content)
    return len(blocks) > 0
