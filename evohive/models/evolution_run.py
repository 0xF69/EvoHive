from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class GenerationStats(BaseModel):
    """一代的统计信息"""
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    alive_count: int
    eliminated_count: int


class EvolutionConfig(BaseModel):
    """一次进化运行的完整配置"""
    problem: str
    population_size: int = 50
    generations: int = 10
    survival_rate: float = 0.2
    mutation_rate: float = 0.3
    diversity_weight: float = 0.15
    elite_rate: float = 0.05
    tournament_size: int = 3
    judge_rounds: int = 1
    thinker_model: str = "deepseek/deepseek-chat"
    judge_model: str = "deepseek/deepseek-chat"
    judge_dimensions: list[dict] = []
    # v2.0 多模型支持
    judge_models: list[str] = []  # 多Judge模型列表
    red_team_models: list[str] = []  # 红队模型列表
    thinker_models: list[str] = []  # 多Thinker模型列表 [(model, count)展开后]
    # v2.0 功能开关
    enable_pairwise_judge: bool = True
    enable_elimination_memory: bool = True
    enable_diversity_guard: bool = True
    enable_fresh_blood: bool = True
    enable_red_team: bool = True
    enable_debate: bool = True
    enable_pressure_test: bool = True
    # v2.0 阈值
    homogeneity_threshold: float = 0.6
    fresh_blood_interval: int = 2
    convergence_threshold: float = 0.7
    memory_window: int = 2  # 淘汰记忆保留的代数
    # v3.0 Swarm & mode
    mode: str = "deep"  # "fast" or "deep"
    enable_swarm: bool = True  # Enable swarm layer
    swarm_count: int = 500  # Number of swarm seeds
    swarm_max_representatives: int = 50  # Max cluster representatives
    swarm_models: list[str] = []  # Models for swarm (Flash-tier)
    embedding_model: str = "text-embedding-3-small"  # Embedding model
    enable_web_search: bool = True  # Enable web search context
    enable_swiss_tournament: bool = True  # Use Swiss tournament instead of full pairwise
    # v3.1 Adaptive parameter control
    enable_adaptive: bool = True  # Enable adaptive mutation/survival rate adjustment
    # Executable fitness verification
    enable_executable_fitness: bool = False  # Off by default (opt-in)
    test_cases: list[dict] = []             # [{"input": "...", "expected": "..."}]
    exec_weight: float = 0.4               # Weight of execution score vs LLM judge
    exec_timeout: float = 5.0              # Timeout per test case


class EvolutionRun(BaseModel):
    """一次完整进化运行的所有数据"""
    id: str
    config: EvolutionConfig
    started_at: datetime
    finished_at: Optional[datetime] = None
    generations_data: list[GenerationStats] = []
    all_solutions: list[dict] = []
    final_top_solutions: list[dict] = []
    baseline_solution: str = ""
    total_api_calls: int = 0
    estimated_cost: float = 0.0
    cost_breakdown: dict = {}  # 按阶段拆分的成本明细
    budget_limit: Optional[float] = None  # 预算上限 (USD)
    refined_top_solution: str = ""
    early_stop_reason: Optional[str] = None
    # v2.0 新增数据
    red_team_results: list[dict] = []
    debate_results: dict = {}
    pressure_test_results: list[dict] = []
    elimination_memories: list[str] = []
    # v3.0 Swarm & mode
    swarm_stats: dict = {}  # Swarm phase statistics
    event_log: list[dict] = []  # Event log for replay
    mode: str = "deep"  # Which mode was used
    search_context: str = ""  # Search context used
    quality_comparison: dict = {}  # Baseline vs Top1 自动对比结果
    # v3.1 Adaptive parameter history
    adaptive_history: list[dict] = []  # History of adaptive parameter adjustments
    execution_results: list[dict] = []     # Execution verification results per generation
