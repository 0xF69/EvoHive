"""Adaptive Parameter Controller — 自适应进化参数

Dynamically adjusts mutation_rate and survival_rate based on
population fitness convergence and diversity signals.

Inspired by self-adaptive EA literature:
- When population converges (fitness variance drops) → increase mutation_rate to explore
- When population is diverse (fitness variance high) → decrease mutation_rate to exploit
- When improvement stalls → expand survival_rate to keep more diverse candidates
- When improvement is rapid → tighten survival_rate to increase selection pressure
"""

from evohive.models.evolution_run import GenerationStats


class AdaptiveController:
    """Tracks generation-over-generation signals and recommends parameter adjustments."""

    def __init__(
        self,
        base_mutation_rate: float = 0.3,
        base_survival_rate: float = 0.2,
        # Bounds
        min_mutation_rate: float = 0.1,
        max_mutation_rate: float = 0.7,
        min_survival_rate: float = 0.1,
        max_survival_rate: float = 0.5,
        # Sensitivity
        adjustment_speed: float = 0.1,
    ):
        self.base_mutation_rate = base_mutation_rate
        self.base_survival_rate = base_survival_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.min_survival_rate = min_survival_rate
        self.max_survival_rate = max_survival_rate
        self.adjustment_speed = adjustment_speed

        self._mutation_rate = base_mutation_rate
        self._survival_rate = base_survival_rate

        # Internal history tracking
        self._history: list[dict] = []
        self._best_fitness_history: list[float] = []

    def update(self, generation_stats: GenerationStats, population_similarity: float) -> dict:
        """Called after each generation with new stats.

        Args:
            generation_stats: Stats from the just-completed generation
            population_similarity: Average pairwise similarity (0-1) of current population

        Returns:
            {"mutation_rate": float, "survival_rate": float, "reason": str}
            with the recommended parameters for the NEXT generation.
        """
        best = generation_stats.best_fitness
        worst = generation_stats.worst_fitness
        self._best_fitness_history.append(best)

        # 1. Fitness improvement rate
        if len(self._best_fitness_history) >= 2:
            prev_best = self._best_fitness_history[-2]
            if prev_best > 0:
                improvement_rate = (best - prev_best) / prev_best
            else:
                improvement_rate = 1.0 if best > 0 else 0.0
        else:
            improvement_rate = 0.0

        # 2. Fitness variance proxy: (best - worst) / best
        if best > 0:
            fitness_spread = (best - worst) / best
        else:
            fitness_spread = 0.0

        # 3. Decision matrix
        speed = self.adjustment_speed
        reason_parts: list[str] = []

        stagnation = improvement_rate < 0.01 and population_similarity > 0.6
        rapid_improvement = improvement_rate > 0.05
        high_diversity = population_similarity < 0.3

        if stagnation:
            # Boost mutation to escape local optima, widen survival to keep diversity
            self._mutation_rate += speed
            self._survival_rate += speed * 0.5
            reason_parts.append(
                f"stagnation detected (improvement={improvement_rate:.3f}, "
                f"similarity={population_similarity:.2f}): boosting mutation & widening survival"
            )
        elif rapid_improvement:
            # Exploit: slightly reduce mutation, tighten survival for selection pressure
            self._mutation_rate -= speed * 0.5
            self._survival_rate -= speed * 0.5
            reason_parts.append(
                f"rapid improvement ({improvement_rate:.3f}): reducing mutation & tightening survival"
            )
        elif high_diversity:
            # Already diverse enough, reduce mutation
            self._mutation_rate -= speed * 0.5
            reason_parts.append(
                f"high diversity (similarity={population_similarity:.2f}): reducing mutation"
            )
        else:
            # Normal: gradual drift toward base rates
            if self._mutation_rate > self.base_mutation_rate:
                self._mutation_rate -= speed * 0.25
            elif self._mutation_rate < self.base_mutation_rate:
                self._mutation_rate += speed * 0.25
            if self._survival_rate > self.base_survival_rate:
                self._survival_rate -= speed * 0.25
            elif self._survival_rate < self.base_survival_rate:
                self._survival_rate += speed * 0.25
            reason_parts.append("normal: drifting toward base rates")

        # 4. Clamp to bounds
        self._mutation_rate = max(self.min_mutation_rate, min(self.max_mutation_rate, self._mutation_rate))
        self._survival_rate = max(self.min_survival_rate, min(self.max_survival_rate, self._survival_rate))

        reason = "; ".join(reason_parts)
        record = {
            "generation": generation_stats.generation,
            "mutation_rate": round(self._mutation_rate, 4),
            "survival_rate": round(self._survival_rate, 4),
            "improvement_rate": round(improvement_rate, 4),
            "fitness_spread": round(fitness_spread, 4),
            "population_similarity": round(population_similarity, 4),
            "reason": reason,
        }
        self._history.append(record)

        return {
            "mutation_rate": self._mutation_rate,
            "survival_rate": self._survival_rate,
            "reason": reason,
        }

    @property
    def current_mutation_rate(self) -> float:
        return self._mutation_rate

    @property
    def current_survival_rate(self) -> float:
        return self._survival_rate

    def summary(self) -> dict:
        """Return history of all adjustments for logging/reporting."""
        return {
            "base_mutation_rate": self.base_mutation_rate,
            "base_survival_rate": self.base_survival_rate,
            "current_mutation_rate": round(self._mutation_rate, 4),
            "current_survival_rate": round(self._survival_rate, 4),
            "total_adjustments": len(self._history),
            "history": list(self._history),
        }
