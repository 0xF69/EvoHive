"""测试锦标赛选择算法"""

from evohive.models import Solution
from evohive.engine.selection import tournament_select


def _make_solutions(n: int) -> list[Solution]:
    """创建n个带不同fitness的测试Solution"""
    solutions = []
    for i in range(n):
        s = Solution(
            id=f"sol_{i:03d}",
            content=f"方案{i}的内容",
            generation=0,
            fitness=(i + 1) / n,  # 0.1, 0.2, ..., 1.0
        )
        solutions.append(s)
    return solutions


def test_basic_selection():
    """基本选择：存活数量正确"""
    solutions = _make_solutions(10)
    survivors, eliminated = tournament_select(solutions, survival_rate=0.3)

    assert len(survivors) == 3
    assert len(eliminated) == 7
    assert len(survivors) + len(eliminated) == 10


def test_elite_always_survives():
    """精英方案必须存活"""
    solutions = _make_solutions(20)
    best = max(solutions, key=lambda s: s.fitness)

    # 跑100次，精英每次都应该存活
    for _ in range(100):
        # 重置状态
        for s in solutions:
            s.is_alive = True
            s.elimination_reason = None

        survivors, _ = tournament_select(solutions, survival_rate=0.2, elite_rate=0.05)
        survivor_ids = {s.id for s in survivors}
        assert best.id in survivor_ids, "最高分方案必须存活"


def test_eliminated_get_reason():
    """淘汰者应该有淘汰原因"""
    solutions = _make_solutions(10)
    _, eliminated = tournament_select(solutions, survival_rate=0.3)

    for s in eliminated:
        assert s.is_alive is False
        assert s.elimination_reason is not None
        assert "锦标赛淘汰" in s.elimination_reason


def test_small_population():
    """极小种群不应崩溃"""
    solutions = _make_solutions(2)
    survivors, eliminated = tournament_select(solutions, survival_rate=0.5)
    assert len(survivors) >= 1


def test_single_solution():
    """只有1个方案时不应崩溃"""
    solutions = _make_solutions(1)
    survivors, eliminated = tournament_select(solutions, survival_rate=0.5)
    assert len(survivors) == 1
    assert len(eliminated) == 0


if __name__ == "__main__":
    test_basic_selection()
    test_elite_always_survives()
    test_eliminated_get_reason()
    test_small_population()
    test_single_solution()
    print("All tests passed!")
