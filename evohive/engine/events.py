"""事件流系统 — 为实时可视化和 WebSocket 推送准备

每个进化阶段发出结构化事件，前端/CLI 通过回调消费。
事件协议轻量，不引入消息队列依赖。
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Any


@dataclass
class EvolutionEvent:
    """进化过程中的一个事件"""
    type: str           # 事件类型
    phase: str          # 所属阶段: "swarm", "evolution", "post_evolution", "refinement"
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class EventEmitter:
    """事件发射器 — 管理事件回调"""

    def __init__(self):
        self._callbacks: list[Callable[[EvolutionEvent], None]] = []
        self._event_log: list[EvolutionEvent] = []
        self._log_events: bool = True

    def on_event(self, callback: Callable[[EvolutionEvent], None]):
        """Register an event callback"""
        self._callbacks.append(callback)

    def emit(self, event_type: str, phase: str, **data):
        """Emit an event to all registered callbacks"""
        event = EvolutionEvent(type=event_type, phase=phase, data=data)

        if self._log_events:
            self._event_log.append(event)

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass  # Never let a callback crash the evolution

    def get_event_log(self) -> list[EvolutionEvent]:
        """Get all logged events"""
        return list(self._event_log)

    def clear_log(self):
        """Clear event log"""
        self._event_log.clear()


# ── 预定义事件类型常量 ──

# Swarm phase
SWARM_STARTED = "swarm_started"
SWARM_SEEDS_GENERATING = "swarm_seeds_generating"
SWARM_SEED_GENERATED = "swarm_seed_generated"
SWARM_SEEDS_COMPLETE = "swarm_seeds_complete"
SWARM_CLUSTERING = "swarm_clustering"
SWARM_CLUSTERS_FORMED = "swarm_clusters_formed"
SWARM_EXPANDING = "swarm_expanding"
SWARM_COMPLETE = "swarm_complete"

# Evolution phase
EVOLUTION_STARTED = "evolution_started"
GENERATION_STARTED = "generation_started"
EVALUATION_STARTED = "evaluation_started"
EVALUATION_COMPLETE = "evaluation_complete"
ELO_TOURNAMENT_STARTED = "elo_tournament_started"
ELO_TOURNAMENT_COMPLETE = "elo_tournament_complete"
CONVERGENCE_DETECTED = "convergence_detected"
SELECTION_COMPLETE = "selection_complete"
CROSSOVER_STARTED = "crossover_started"
CROSSOVER_COMPLETE = "crossover_complete"
MUTATION_COMPLETE = "mutation_complete"
HOMOGENEITY_CULLED = "homogeneity_culled"
FRESH_BLOOD_INJECTED = "fresh_blood_injected"
GENERATION_COMPLETE = "generation_complete"
EVOLUTION_COMPLETE = "evolution_complete"

# Post-evolution phase
RED_TEAM_STARTED = "red_team_started"
RED_TEAM_ATTACK = "red_team_attack"
RED_TEAM_COMPLETE = "red_team_complete"
DEBATE_STARTED = "debate_started"
DEBATE_ROUND = "debate_round"
DEBATE_COMPLETE = "debate_complete"
PRESSURE_TEST_STARTED = "pressure_test_started"
PRESSURE_TEST_COMPLETE = "pressure_test_complete"

# Refinement phase
REFINEMENT_STARTED = "refinement_started"
REFINEMENT_CHAPTER = "refinement_chapter"
REFINEMENT_COMPLETE = "refinement_complete"

# Overall
RUN_STARTED = "run_started"
RUN_COMPLETE = "run_complete"

# Pre-flight model checks
PREFLIGHT_OK = "preflight_ok"
PREFLIGHT_PARTIAL = "preflight_partial"
