"""用户自定义模型配置"""

import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field


class ModelEntry(BaseModel):
    """单个模型配置"""
    model: str
    count: int = 1


class EvoHiveConfig(BaseModel):
    """完整的EvoHive配置"""
    judges: list[ModelEntry] = Field(default_factory=lambda: [ModelEntry(model="deepseek/deepseek-chat")])
    thinkers: list[ModelEntry] = Field(default_factory=lambda: [ModelEntry(model="deepseek/deepseek-chat")])
    red_team: list[ModelEntry] = Field(default_factory=lambda: [ModelEntry(model="deepseek/deepseek-chat")])

    # Evolution parameters
    population_size: int = 10
    generations: int = 3
    survival_rate: float = 0.3
    mutation_rate: float = 0.3

    # Swarm configuration
    mode: str = "deep"
    enable_swarm: bool = True
    swarm_count: int = 500
    swarm_max_representatives: int = 50
    swarm_models: list[ModelEntry] = Field(default_factory=list)
    embedding_model: str = "text-embedding-3-small"
    enable_web_search: bool = True
    enable_swiss_tournament: bool = True

    # Feature toggles
    enable_pairwise_judge: bool = True
    enable_elimination_memory: bool = True
    enable_diversity_guard: bool = True
    enable_fresh_blood: bool = True
    enable_red_team: bool = True
    enable_debate: bool = True
    enable_pressure_test: bool = True
    enable_dialogue: bool = True

    # Thresholds
    homogeneity_threshold: float = 0.6
    fresh_blood_interval: int = 2
    fresh_blood_similarity_threshold: float = 0.5
    convergence_threshold: float = 0.7

    def get_judge_models(self) -> list[str]:
        """获取所有Judge模型列表"""
        return [j.model for j in self.judges]

    def get_thinker_models(self) -> list[str]:
        """获取所有Thinker模型列表（按count展开）"""
        models = []
        for t in self.thinkers:
            models.extend([t.model] * t.count)
        return models

    def get_red_team_models(self) -> list[str]:
        """获取所有红队模型列表"""
        return [r.model for r in self.red_team]

    def get_swarm_models(self) -> list[str]:
        """获取所有Swarm模型列表，若为空则回退到第一个thinker模型"""
        if self.swarm_models:
            models = []
            for s in self.swarm_models:
                models.extend([s.model] * s.count)
            return models
        # Fallback to first thinker model
        if self.thinkers:
            return [self.thinkers[0].model]
        return []


def load_config(path: str | Path | None = None) -> EvoHiveConfig:
    """从YAML文件加载配置

    Args:
        path: 配置文件路径。如果为None，尝试以下顺序:
              1. ./evohive.yaml
              2. ~/.evohive.yaml
              3. 使用默认配置

    Returns:
        EvoHiveConfig实例
    """
    if path:
        p = Path(path)
        if p.exists():
            return _parse_config(p)
        raise FileNotFoundError(f"配置文件不存在: {path}")

    # Auto-discover
    for candidate in [
        Path("evohive.yaml"),
        Path("evohive.yml"),
        Path.home() / ".evohive.yaml",
        Path.home() / ".evohive.yml",
    ]:
        if candidate.exists():
            return _parse_config(candidate)

    # Default config
    return EvoHiveConfig()


def _parse_config(path: Path) -> EvoHiveConfig:
    """Parse a YAML config file into EvoHiveConfig"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or not isinstance(raw, dict):
        return EvoHiveConfig()

    config_dict = {}

    # Parse model lists
    for key in ["judges", "thinkers", "red_team", "swarm_models"]:
        if key in raw and isinstance(raw[key], list):
            entries = []
            for item in raw[key]:
                if isinstance(item, str):
                    entries.append(ModelEntry(model=item))
                elif isinstance(item, dict):
                    entries.append(ModelEntry(**item))
            config_dict[key] = entries

    # Parse simple fields
    simple_fields = [
        "population_size", "generations", "survival_rate", "mutation_rate",
        "enable_pairwise_judge", "enable_elimination_memory",
        "enable_diversity_guard", "enable_fresh_blood",
        "enable_red_team", "enable_debate", "enable_pressure_test",
        "enable_dialogue", "homogeneity_threshold", "fresh_blood_interval",
        "fresh_blood_similarity_threshold", "convergence_threshold",
        "mode", "enable_swarm", "swarm_count", "swarm_max_representatives",
        "embedding_model", "enable_web_search", "enable_swiss_tournament",
    ]
    for field in simple_fields:
        if field in raw:
            config_dict[field] = raw[field]

    return EvoHiveConfig(**config_dict)
