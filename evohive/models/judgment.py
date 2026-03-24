from pydantic import BaseModel


class DimensionScore(BaseModel):
    """单个维度的评分"""
    name: str
    score: int
    reason: str
    weight: float


class Judgment(BaseModel):
    """一次评审的完整结果"""
    solution_id: str
    scores: list[DimensionScore]
    raw_fitness: float


class StableJudgment(BaseModel):
    """去噪后的稳定评审结果"""
    solution_id: str
    median_scores: list[DimensionScore]
    raw_fitness: float
    diversity_bonus: float = 0.0
    final_fitness: float = 0.0
