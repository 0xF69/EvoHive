"""实时成本追踪与预算控制系统 — Real-time Cost Tracking & Budget Control

功能:
- 按 API 调用追踪 token 用量 (input / output)
- 基于 MODEL_REGISTRY 中的价格自动计算费用
- 维护运行时累计总成本
- 可设置预算上限, 超限时抛出 BudgetExceededError
- 线程安全 (使用 threading.Lock)
- 支持运行前成本预估
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from evohive.llm.model_registry import MODEL_REGISTRY, ModelInfo, detect_available_models
from evohive.models.evolution_run import EvolutionConfig


# ═══ 异常定义 ═══

class BudgetExceededError(Exception):
    """预算超限异常 — 当累计成本超过设定的 budget_limit 时抛出"""

    def __init__(self, current_cost: float, budget_limit: float, last_call_cost: float):
        self.current_cost = current_cost
        self.budget_limit = budget_limit
        self.last_call_cost = last_call_cost
        super().__init__(
            f"Budget exceeded: ${current_cost:.4f} > ${budget_limit:.4f} "
            f"(last call: ${last_call_cost:.4f})"
        )


# ═══ 单次调用记录 ═══

@dataclass
class CallRecord:
    """单次 API 调用的成本记录"""
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost: float
    phase: str = ""  # 调用所属阶段 (baseline / swarm / generation / red_team / ...)


# ═══ 模型价格查找表 (缓存) ═══

_MODEL_COST_MAP: dict[str, ModelInfo] = {}


def _get_model_info(model_id: str) -> Optional[ModelInfo]:
    """根据 model_id 查找注册表中的模型信息 (带缓存)"""
    if not _MODEL_COST_MAP:
        # 首次调用时构建查找表
        for m in MODEL_REGISTRY:
            _MODEL_COST_MAP[m.id] = m
    return _MODEL_COST_MAP.get(model_id)


def _calc_call_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """计算单次调用费用 (USD)

    公式: (input_tokens / 1000) * cost_per_1k_input
         + (output_tokens / 1000) * cost_per_1k_output
    """
    info = _get_model_info(model_id)
    if info is None:
        # 未注册的模型, 无法计算费用
        return 0.0
    return (
        (input_tokens / 1000.0) * info.cost_per_1k_input
        + (output_tokens / 1000.0) * info.cost_per_1k_output
    )


# ═══ 核心: CostTracker ═══

class CostTracker:
    """实时成本追踪器 — 线程安全

    Usage:
        tracker = CostTracker(budget_limit=5.0)
        tracker.record_call("openai/gpt-4o", 1000, 500, phase="generation")
        print(tracker.total_cost)
        print(tracker.format_report())
    """

    def __init__(self, budget_limit: Optional[float] = None):
        """
        Args:
            budget_limit: 预算上限 (USD), None 表示不限制
        """
        self.budget_limit = budget_limit
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        # 按 provider 汇总
        self._provider_costs: dict[str, float] = {}
        self._provider_calls: dict[str, int] = {}
        # 按 phase 汇总
        self._phase_costs: dict[str, float] = {}

    # ── 属性访问 ──

    @property
    def total_cost(self) -> float:
        """当前累计总成本 (USD)"""
        with self._lock:
            return self._total_cost

    @property
    def total_input_tokens(self) -> int:
        with self._lock:
            return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        with self._lock:
            return self._total_output_tokens

    @property
    def call_count(self) -> int:
        """总 API 调用次数"""
        with self._lock:
            return len(self._records)

    # ── 核心方法 ──

    def record_call(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        phase: str = "",
    ) -> float:
        """记录一次 API 调用并更新累计成本

        Args:
            model_id: 模型标识符 (如 "openai/gpt-4o")
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            phase: 所属阶段名

        Returns:
            本次调用的费用 (USD)

        Raises:
            BudgetExceededError: 累计成本超过 budget_limit
        """
        cost = _calc_call_cost(model_id, input_tokens, output_tokens)

        # 查找 provider
        info = _get_model_info(model_id)
        provider = info.provider if info else "unknown"

        record = CallRecord(
            model_id=model_id,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            phase=phase,
        )

        with self._lock:
            self._records.append(record)
            self._total_cost += cost
            self._total_input_tokens += input_tokens
            self._total_output_tokens += output_tokens

            # 按 provider 汇总
            self._provider_costs[provider] = (
                self._provider_costs.get(provider, 0.0) + cost
            )
            self._provider_calls[provider] = (
                self._provider_calls.get(provider, 0) + 1
            )

            # 按 phase 汇总
            if phase:
                self._phase_costs[phase] = (
                    self._phase_costs.get(phase, 0.0) + cost
                )

            current_total = self._total_cost

        # 预算检查 (在锁外抛出, 避免死锁)
        if self.budget_limit is not None and current_total > self.budget_limit:
            raise BudgetExceededError(current_total, self.budget_limit, cost)

        return cost

    def estimate_cost(
        self,
        population_size: int,
        generations: int,
        mode: str = "deep",
        n_models: int = 1,
    ) -> dict:
        """运行前成本预估 (基于平均模型价格)

        Args:
            population_size: 种群大小
            generations: 进化代数
            mode: "fast" or "deep"
            n_models: 可用模型数量

        Returns:
            {"estimated_min": float, "estimated_max": float, "breakdown": dict}
        """
        return _estimate_cost_internal(
            population_size=population_size,
            generations=generations,
            mode=mode,
            n_models=n_models,
        )

    def format_report(self) -> str:
        """格式化成本报告 (CLI 显示用)

        包含: 总成本, token 统计, 按 provider 分组明细, 按 phase 分组明细
        """
        with self._lock:
            total = self._total_cost
            total_in = self._total_input_tokens
            total_out = self._total_output_tokens
            n_calls = len(self._records)
            provider_costs = dict(self._provider_costs)
            provider_calls = dict(self._provider_calls)
            phase_costs = dict(self._phase_costs)

        lines = []
        lines.append("=" * 50)
        lines.append("  Cost Report / 成本报告")
        lines.append("=" * 50)
        lines.append(f"  Total Cost:    ${total:.4f}")
        lines.append(f"  API Calls:     {n_calls}")
        lines.append(f"  Input Tokens:  {total_in:,}")
        lines.append(f"  Output Tokens: {total_out:,}")

        if self.budget_limit is not None:
            remaining = max(0.0, self.budget_limit - total)
            pct = (total / self.budget_limit * 100) if self.budget_limit > 0 else 0
            lines.append(f"  Budget:        ${self.budget_limit:.4f} "
                         f"(used {pct:.1f}%, remaining ${remaining:.4f})")

        # 按 Provider 分组
        if provider_costs:
            lines.append("")
            lines.append("  Per-Provider Breakdown / 按 Provider 明细:")
            lines.append("  " + "-" * 46)
            # 按费用降序排列
            sorted_providers = sorted(
                provider_costs.items(), key=lambda x: x[1], reverse=True
            )
            for prov, cost in sorted_providers:
                calls = provider_calls.get(prov, 0)
                lines.append(f"    {prov:<20s}  ${cost:.4f}  ({calls} calls)")

        # 按 Phase 分组
        if phase_costs:
            lines.append("")
            lines.append("  Per-Phase Breakdown / 按阶段明细:")
            lines.append("  " + "-" * 46)
            sorted_phases = sorted(
                phase_costs.items(), key=lambda x: x[1], reverse=True
            )
            for phase, cost in sorted_phases:
                lines.append(f"    {phase:<20s}  ${cost:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def reset(self):
        """重置所有累计数据"""
        with self._lock:
            self._records.clear()
            self._total_cost = 0.0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._provider_costs.clear()
            self._provider_calls.clear()
            self._phase_costs.clear()


# ═══ 预估逻辑 (内部) ═══

def _get_avg_cost(models: list[ModelInfo]) -> tuple[float, float]:
    """计算可用模型的平均 input / output 单价 (per 1k tokens)

    Returns:
        (avg_cost_per_1k_input, avg_cost_per_1k_output)
    """
    if not models:
        # 如果没有可用模型, 使用全注册表的平均值
        models = MODEL_REGISTRY

    total_in = sum(m.cost_per_1k_input for m in models)
    total_out = sum(m.cost_per_1k_output for m in models)
    n = len(models)
    return (total_in / n, total_out / n)


def _estimate_tokens_cost(
    n_calls: int,
    avg_output_tokens: int,
    avg_input_tokens: int,
    avg_in_price: float,
    avg_out_price: float,
) -> float:
    """预估 n 次调用的总费用

    假设每次调用的 input 约 avg_input_tokens, output 约 avg_output_tokens
    """
    per_call = (
        (avg_input_tokens / 1000.0) * avg_in_price
        + (avg_output_tokens / 1000.0) * avg_out_price
    )
    return n_calls * per_call


def _estimate_cost_internal(
    population_size: int,
    generations: int,
    mode: str = "deep",
    n_models: int = 1,
    # 以下为可选的 config 细节, 用于更精确预估
    enable_swarm: bool = True,
    swarm_count: int = 500,
    swarm_max_representatives: int = 50,
    enable_pairwise_judge: bool = True,
    enable_red_team: bool = True,
    enable_debate: bool = True,
    enable_pressure_test: bool = True,
    n_red_team_models: int = 3,
) -> dict:
    """内部预估逻辑 — 基于各阶段的调用量模型

    预估假设:
    - 输入 token 默认 ~1000 (prompt + context)
    - 输出 token 按阶段不同而不同

    Returns:
        {"estimated_min": float, "estimated_max": float, "breakdown": dict}
    """
    available = detect_available_models()
    avg_in_price, avg_out_price = _get_avg_cost(available)

    # 默认 input tokens per call (prompt 平均长度)
    DEFAULT_INPUT = 1000
    breakdown: dict[str, float] = {}

    # ── 1. Baseline: 1 call, ~2000 tokens output ──
    baseline_cost = _estimate_tokens_cost(1, 2000, DEFAULT_INPUT, avg_in_price, avg_out_price)
    breakdown["baseline"] = baseline_cost

    # ── 2. Swarm Seeds: swarm_count calls, ~100 tokens output each ──
    swarm_seed_cost = 0.0
    if enable_swarm:
        swarm_seed_cost = _estimate_tokens_cost(
            swarm_count, 100, DEFAULT_INPUT, avg_in_price, avg_out_price
        )
    breakdown["swarm_seeds"] = swarm_seed_cost

    # ── 3. Swarm Expansion: swarm_max_representatives calls, ~1000 tokens output each ──
    swarm_expand_cost = 0.0
    if enable_swarm:
        swarm_expand_cost = _estimate_tokens_cost(
            swarm_max_representatives, 1000, DEFAULT_INPUT, avg_in_price, avg_out_price
        )
    breakdown["swarm_expansion"] = swarm_expand_cost

    # ── 4. Per-Generation 循环 ──
    per_gen_cost = 0.0

    # 4a. 评估: population_size evaluations, ~500 tokens each
    eval_cost = _estimate_tokens_cost(
        population_size, 500, DEFAULT_INPUT, avg_in_price, avg_out_price
    )

    # 4b. Elo / Swiss 对战: 约 population_size * log2(population_size) 次比较
    import math
    n_comparisons = int(population_size * max(1, math.log2(max(2, population_size))))
    if enable_pairwise_judge:
        elo_cost = _estimate_tokens_cost(
            n_comparisons, 300, DEFAULT_INPUT, avg_in_price, avg_out_price
        )
    else:
        elo_cost = 0.0

    # 4c. Crossover: ~population_size * survival_rate 次交叉
    crossover_calls = max(1, int(population_size * 0.5))
    crossover_cost = _estimate_tokens_cost(
        crossover_calls, 1000, DEFAULT_INPUT, avg_in_price, avg_out_price
    )

    # 4d. Mutation: ~population_size * mutation_rate 次变异
    mutation_calls = max(1, int(population_size * 0.3))
    mutation_cost = _estimate_tokens_cost(
        mutation_calls, 800, DEFAULT_INPUT, avg_in_price, avg_out_price
    )

    per_gen_cost = eval_cost + elo_cost + crossover_cost + mutation_cost
    total_gen_cost = per_gen_cost * generations
    breakdown["generations"] = total_gen_cost

    # ── 5. Post-Evolution: Red Team, Debate, Pressure Test ──
    # Red Team: top-5 方案 * n_red_team_models 个模型
    red_team_cost = 0.0
    if enable_red_team:
        red_team_calls = 5 * n_red_team_models
        red_team_cost = _estimate_tokens_cost(
            red_team_calls, 800, DEFAULT_INPUT, avg_in_price, avg_out_price
        )
    breakdown["red_team"] = red_team_cost

    # Debate: 约 10 轮 debate (top-5 方案两两辩论)
    debate_cost = 0.0
    if enable_debate:
        debate_calls = 10
        debate_cost = _estimate_tokens_cost(
            debate_calls, 1000, DEFAULT_INPUT, avg_in_price, avg_out_price
        )
    breakdown["debate"] = debate_cost

    # Pressure Test: top-5 方案
    pressure_cost = 0.0
    if enable_pressure_test:
        pressure_calls = 5
        pressure_cost = _estimate_tokens_cost(
            pressure_calls, 800, DEFAULT_INPUT, avg_in_price, avg_out_price
        )
    breakdown["pressure_test"] = pressure_cost

    # ── 6. Refinement: ~7 calls, ~2000 tokens each ──
    refine_cost = _estimate_tokens_cost(7, 2000, DEFAULT_INPUT, avg_in_price, avg_out_price)
    breakdown["refinement"] = refine_cost

    # ── 汇总 ──
    total = sum(breakdown.values())

    # Min / Max 估计: 实际消耗在 0.5x ~ 2.0x 之间波动
    # (取决于 prompt 复杂度、模型选择、early stop 等)
    estimated_min = total * 0.5
    estimated_max = total * 2.0

    return {
        "estimated_min": round(estimated_min, 4),
        "estimated_max": round(estimated_max, 4),
        "breakdown": {k: round(v, 4) for k, v in breakdown.items()},
    }


# ═══ 便捷入口: estimate_run_cost ═══

def estimate_run_cost(config: EvolutionConfig, n_available_models: int) -> dict:
    """根据 EvolutionConfig 预估整次运行的成本

    Args:
        config: 进化配置
        n_available_models: 当前可用模型数量

    Returns:
        {
            "estimated_min": float,   # 最低预估 (USD)
            "estimated_max": float,   # 最高预估 (USD)
            "breakdown": {            # 按阶段拆分
                "baseline": float,
                "swarm_seeds": float,
                "swarm_expansion": float,
                "generations": float,
                "red_team": float,
                "debate": float,
                "pressure_test": float,
                "refinement": float,
            }
        }
    """
    # fast 模式只跑 1 代
    effective_generations = 1 if config.mode == "fast" else config.generations

    n_red_team_models = len(config.red_team_models) if config.red_team_models else 3

    return _estimate_cost_internal(
        population_size=config.population_size,
        generations=effective_generations,
        mode=config.mode,
        n_models=n_available_models,
        enable_swarm=config.enable_swarm,
        swarm_count=config.swarm_count,
        swarm_max_representatives=config.swarm_max_representatives,
        enable_pairwise_judge=config.enable_pairwise_judge,
        enable_red_team=config.enable_red_team,
        enable_debate=config.enable_debate,
        enable_pressure_test=config.enable_pressure_test,
        n_red_team_models=n_red_team_models,
    )
