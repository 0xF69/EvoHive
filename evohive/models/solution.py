from pydantic import BaseModel, Field
from typing import Optional
import uuid


class GeneRecord(BaseModel):
    """血统记录"""
    from_parent_a: list[str] = []
    from_parent_b: list[str] = []
    novel: list[str] = []


class Solution(BaseModel):
    """一个完整的方案"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    generation: int = 0
    thinker_id: Optional[str] = None
    parent_a_id: Optional[str] = None
    parent_b_id: Optional[str] = None
    gene_record: Optional[GeneRecord] = None
    fitness: float = 0.0
    raw_fitness: float = 0.0
    diversity_bonus: float = 0.0
    is_alive: bool = True
    elimination_reason: Optional[str] = None
    mutation_applied: Optional[str] = None
    elimination_feedback: Optional[list[str]] = None  # 淘汰反馈原因
    red_team_vulnerability: Optional[float] = None  # 红队攻击脆弱度
    pressure_resilience: Optional[float] = None  # 压力测试韧性
    debate_elo: Optional[float] = None  # 辩论Elo评分
    seed_content: Optional[str] = None  # Original seed (if from swarm)
    cluster_id: Optional[int] = None  # Cluster assignment
    embedding: Optional[list[float]] = Field(default=None, exclude=True)  # Cached embedding vector
    # Executable fitness verification
    execution_score: Optional[float] = None    # Score from code execution (0-1)
    execution_passed: Optional[int] = None     # Number of test cases passed
    execution_total: Optional[int] = None      # Total test cases
