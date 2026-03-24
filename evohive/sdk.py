"""EvoHive Python SDK -- one-line evolutionary collective intelligence.

Usage:
    from evohive import evolve

    result = await evolve("How to improve team efficiency?")
    print(result.best_solution)
    print(result.fitness)
"""

import asyncio
from typing import Optional, Callable

from evohive.models import EvolutionConfig, EvolutionRun


_DEFAULT_DIMENSIONS = [
    {"name": "feasibility", "weight": 0.3, "description": "feasibility"},
    {"name": "innovation", "weight": 0.25, "description": "innovation"},
    {"name": "specificity", "weight": 0.25, "description": "specificity"},
    {"name": "cost_efficiency", "weight": 0.2, "description": "cost efficiency"},
]


class EvolutionResult:
    """Friendly wrapper around EvolutionRun with convenient accessors."""

    def __init__(self, run: EvolutionRun):
        self._run = run

    @property
    def best_solution(self) -> str:
        """The top-ranked solution content.

        Returns the refined version if available, otherwise the raw top
        solution content.
        """
        if self._run.refined_top_solution:
            return self._run.refined_top_solution
        if self._run.final_top_solutions:
            return self._run.final_top_solutions[0].get("content", "")
        return ""

    @property
    def fitness(self) -> float:
        """Fitness score of the best solution."""
        if self._run.final_top_solutions:
            return self._run.final_top_solutions[0].get("fitness", 0.0)
        return 0.0

    @property
    def top_solutions(self) -> list[dict]:
        """Top 5 solutions with fitness scores."""
        return [
            {
                "rank": i + 1,
                "content": sol.get("content", ""),
                "fitness": sol.get("fitness", 0.0),
            }
            for i, sol in enumerate(self._run.final_top_solutions[:5])
        ]

    @property
    def cost(self) -> float:
        """Estimated cost in USD."""
        return self._run.estimated_cost

    @property
    def duration_seconds(self) -> float:
        """Total run time in seconds."""
        if self._run.finished_at and self._run.started_at:
            return (self._run.finished_at - self._run.started_at).total_seconds()
        return 0.0

    @property
    def generations_data(self):
        """Per-generation statistics."""
        return self._run.generations_data

    @property
    def raw(self) -> EvolutionRun:
        """Access the full EvolutionRun object."""
        return self._run

    def __str__(self) -> str:
        """Pretty print the result summary."""
        lines = [
            "=== EvoHive Evolution Result ===",
            f"Best fitness: {self.fitness:.3f}",
            f"Generations:  {len(self.generations_data)}",
            f"Duration:     {self.duration_seconds:.1f}s",
            f"API calls:    {self._run.total_api_calls}",
        ]
        if self.cost > 0:
            lines.append(f"Est. cost:    ${self.cost:.4f}")
        lines.append("")
        lines.append("--- Best Solution ---")
        # Truncate for display
        best = self.best_solution
        if len(best) > 500:
            lines.append(best[:500] + "...")
        else:
            lines.append(best)
        return "\n".join(lines)


async def evolve(
    problem: str,
    *,
    mode: str = "fast",
    population_size: int = 20,
    generations: int = 5,
    budget_limit: Optional[float] = None,
    thinker_model: str = "deepseek/deepseek-chat",
    judge_model: str = "deepseek/deepseek-chat",
    save_results: bool = True,
    output_dir: str = "evohive_results",
    on_generation: Optional[Callable] = None,
    **kwargs,
) -> EvolutionResult:
    """Run evolutionary optimization on a problem.

    This is the main entry point for the EvoHive SDK.

    Args:
        problem: The problem statement to optimize solutions for.
        mode: "fast" or "deep". Fast uses smaller population and fewer
            generations.
        population_size: Number of solutions per generation.
        generations: Number of evolutionary generations.
        budget_limit: Maximum budget in USD (None = unlimited).
        thinker_model: Model for generating solutions.
        judge_model: Model for evaluating solutions.
        save_results: Whether to save results to disk.
        output_dir: Directory for saved results.
        on_generation: Callback(generation, stats, best_solution) called
            after each generation completes.
        **kwargs: Additional EvolutionConfig parameters (e.g.
            survival_rate, mutation_rate, judge_dimensions,
            enable_swarm, etc.).

    Returns:
        EvolutionResult with convenient accessors for the best solution,
        cost, etc.
    """
    from evohive.engine.evolution import run_evolution

    # Build config, letting kwargs override defaults
    config_params = dict(
        problem=problem,
        mode=mode,
        population_size=population_size,
        generations=generations,
        thinker_model=thinker_model,
        judge_model=judge_model,
    )

    # Apply default dimensions if not provided
    if "judge_dimensions" not in kwargs:
        config_params["judge_dimensions"] = _DEFAULT_DIMENSIONS

    # Fast mode adjustments (matching CLI behaviour)
    if mode == "fast":
        config_params.setdefault("enable_debate", False)
        config_params.setdefault("enable_pressure_test", False)

    # Merge caller-supplied overrides
    config_params.update(kwargs)

    config = EvolutionConfig(**config_params)

    run = await run_evolution(
        config,
        on_generation_complete=on_generation,
        budget_limit=budget_limit,
        save_results=save_results,
        output_dir=output_dir,
    )

    return EvolutionResult(run)


def evolve_sync(problem: str, **kwargs) -> EvolutionResult:
    """Synchronous wrapper around evolve() for non-async contexts.

    Usage:
        from evohive import evolve_sync
        result = evolve_sync("How to improve team efficiency?")
    """
    return asyncio.run(evolve(problem, **kwargs))
