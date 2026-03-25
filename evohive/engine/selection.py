"""锦标赛选择算法"""

import random
from evohive.models import Solution


def tournament_select(
    solutions: list[Solution],
    survival_rate: float,
    elite_rate: float = 0.05,
    tournament_size: int = 3,
) -> tuple[list[Solution], list[Solution]]:
    """锦标赛选择

    Returns:
        (survivors, eliminated)
    """
    if not solutions:
        return [], []

    n = len(solutions)
    target_survivors = max(2, int(n * survival_rate))

    # 1. 精英通道
    sorted_solutions = sorted(solutions, key=lambda s: s.fitness, reverse=True)
    n_elite = max(1, int(n * elite_rate))
    elite = sorted_solutions[:n_elite]

    survived_ids = {s.id for s in elite}
    survivors = list(elite)

    # 2. 锦标赛通道
    remaining = [s for s in solutions if s.id not in survived_ids]

    while len(survivors) < target_survivors and remaining:
        # 随机抽取tournament_size个
        k = min(tournament_size, len(remaining))
        contestants = random.sample(remaining, k)

        # 最高分者存活
        winner = max(contestants, key=lambda s: s.fitness)
        survivors.append(winner)
        survived_ids.add(winner.id)
        remaining = [s for s in remaining if s.id != winner.id]

    # 3. 标记淘汰者
    eliminated = []
    for s in solutions:
        if s.id not in survived_ids:
            s.is_alive = False
            s.elimination_reason = f"锦标赛淘汰 (fitness={s.fitness:.3f})"
            eliminated.append(s)

    return survivors, eliminated
