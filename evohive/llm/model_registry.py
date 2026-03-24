"""全局模型注册表 + API Key 自动检测

覆盖市面上所有主流大模型 Provider:
- 检测环境变量中已配置的 API Key
- 自动分配到 Thinker / Judge / Red Team / Swarm 角色
- 按模型能力分级: flagship (旗舰) / standard (标准) / flash (轻量)
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelInfo:
    """单个模型的注册信息"""
    id: str                # LiteLLM 模型标识符
    provider: str          # Provider 名称
    env_var: str           # 需要的环境变量
    tier: str              # "flagship" | "standard" | "flash"
    display_name: str      # 显示名称
    cost_per_1k_input: float = 0.0   # 千 token 输入价格 (USD, 近似)
    cost_per_1k_output: float = 0.0  # 千 token 输出价格 (USD, 近似)
    supports_json_mode: bool = True   # 是否支持 JSON mode
    max_context: int = 128000         # 最大上下文长度


# ═══ 模型注册表 ═══
# 按 Provider 分组，每个 Provider 注册多个模型

MODEL_REGISTRY: list[ModelInfo] = [

    # ── OpenAI ──
    ModelInfo(
        id="openai/gpt-4o",
        provider="openai", env_var="OPENAI_API_KEY",
        tier="flagship", display_name="GPT-4o",
        cost_per_1k_input=0.0025, cost_per_1k_output=0.01,
    ),
    ModelInfo(
        id="openai/gpt-4o-mini",
        provider="openai", env_var="OPENAI_API_KEY",
        tier="flash", display_name="GPT-4o Mini",
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
    ),
    ModelInfo(
        id="openai/gpt-4.1",
        provider="openai", env_var="OPENAI_API_KEY",
        tier="flagship", display_name="GPT-4.1",
        cost_per_1k_input=0.002, cost_per_1k_output=0.008,
    ),
    ModelInfo(
        id="openai/gpt-4.1-mini",
        provider="openai", env_var="OPENAI_API_KEY",
        tier="standard", display_name="GPT-4.1 Mini",
        cost_per_1k_input=0.0004, cost_per_1k_output=0.0016,
    ),
    ModelInfo(
        id="openai/gpt-4.1-nano",
        provider="openai", env_var="OPENAI_API_KEY",
        tier="flash", display_name="GPT-4.1 Nano",
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0004,
    ),
    ModelInfo(
        id="openai/o3-mini",
        provider="openai", env_var="OPENAI_API_KEY",
        tier="standard", display_name="o3-mini",
        cost_per_1k_input=0.0011, cost_per_1k_output=0.0044,
    ),

    # ── Anthropic (Claude) ──
    ModelInfo(
        id="anthropic/claude-sonnet-4-20250514",
        provider="anthropic", env_var="ANTHROPIC_API_KEY",
        tier="flagship", display_name="Claude Sonnet 4",
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
    ),
    ModelInfo(
        id="anthropic/claude-3-5-haiku-20241022",
        provider="anthropic", env_var="ANTHROPIC_API_KEY",
        tier="flash", display_name="Claude 3.5 Haiku",
        cost_per_1k_input=0.0008, cost_per_1k_output=0.004,
    ),

    # ── Google Gemini ──
    ModelInfo(
        id="gemini/gemini-2.5-pro",
        provider="gemini", env_var="GEMINI_API_KEY",
        tier="flagship", display_name="Gemini 2.5 Pro",
        cost_per_1k_input=0.00125, cost_per_1k_output=0.01,
    ),
    ModelInfo(
        id="gemini/gemini-2.5-flash",
        provider="gemini", env_var="GEMINI_API_KEY",
        tier="standard", display_name="Gemini 2.5 Flash",
        cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
    ),
    ModelInfo(
        id="gemini/gemini-2.0-flash",
        provider="gemini", env_var="GEMINI_API_KEY",
        tier="flash", display_name="Gemini 2.0 Flash",
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0004,
    ),

    # ── DeepSeek ──
    ModelInfo(
        id="deepseek/deepseek-chat",
        provider="deepseek", env_var="DEEPSEEK_API_KEY",
        tier="standard", display_name="DeepSeek V3",
        cost_per_1k_input=0.00027, cost_per_1k_output=0.0011,
    ),
    ModelInfo(
        id="deepseek/deepseek-reasoner",
        provider="deepseek", env_var="DEEPSEEK_API_KEY",
        tier="flagship", display_name="DeepSeek R1",
        cost_per_1k_input=0.00055, cost_per_1k_output=0.0022,
    ),

    # ── Groq (免费/快速推理) ──
    ModelInfo(
        id="groq/llama-3.3-70b-versatile",
        provider="groq", env_var="GROQ_API_KEY",
        tier="standard", display_name="Llama 3.3 70B (Groq)",
        cost_per_1k_input=0.00059, cost_per_1k_output=0.00079,
    ),
    ModelInfo(
        id="groq/llama-3.1-8b-instant",
        provider="groq", env_var="GROQ_API_KEY",
        tier="flash", display_name="Llama 3.1 8B (Groq)",
        cost_per_1k_input=0.00005, cost_per_1k_output=0.00008,
    ),
    ModelInfo(
        id="groq/gemma2-9b-it",
        provider="groq", env_var="GROQ_API_KEY",
        tier="flash", display_name="Gemma2 9B (Groq)",
        cost_per_1k_input=0.0002, cost_per_1k_output=0.0002,
    ),

    # ── Mistral ──
    ModelInfo(
        id="mistral/mistral-large-latest",
        provider="mistral", env_var="MISTRAL_API_KEY",
        tier="flagship", display_name="Mistral Large",
        cost_per_1k_input=0.002, cost_per_1k_output=0.006,
    ),
    ModelInfo(
        id="mistral/mistral-small-latest",
        provider="mistral", env_var="MISTRAL_API_KEY",
        tier="standard", display_name="Mistral Small",
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0003,
    ),

    # ── xAI (Grok) ──
    ModelInfo(
        id="xai/grok-3",
        provider="xai", env_var="XAI_API_KEY",
        tier="flagship", display_name="Grok 3",
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
    ),
    ModelInfo(
        id="xai/grok-3-mini",
        provider="xai", env_var="XAI_API_KEY",
        tier="standard", display_name="Grok 3 Mini",
        cost_per_1k_input=0.0003, cost_per_1k_output=0.0005,
    ),

    # ── Together AI ──
    ModelInfo(
        id="together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        provider="together_ai", env_var="TOGETHERAI_API_KEY",
        tier="standard", display_name="Llama 3.3 70B (Together)",
        cost_per_1k_input=0.00088, cost_per_1k_output=0.00088,
    ),
    ModelInfo(
        id="together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        provider="together_ai", env_var="TOGETHERAI_API_KEY",
        tier="standard", display_name="Qwen 2.5 72B (Together)",
        cost_per_1k_input=0.0012, cost_per_1k_output=0.0012,
    ),

    # ── Fireworks AI ──
    ModelInfo(
        id="fireworks_ai/accounts/fireworks/models/llama-v3p3-70b-instruct",
        provider="fireworks_ai", env_var="FIREWORKS_API_KEY",
        tier="standard", display_name="Llama 3.3 70B (Fireworks)",
        cost_per_1k_input=0.0009, cost_per_1k_output=0.0009,
    ),

    # ── Cohere ──
    ModelInfo(
        id="cohere/command-r-plus",
        provider="cohere", env_var="COHERE_API_KEY",
        tier="flagship", display_name="Command R+",
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
    ),
    ModelInfo(
        id="cohere/command-r",
        provider="cohere", env_var="COHERE_API_KEY",
        tier="standard", display_name="Command R",
        cost_per_1k_input=0.0005, cost_per_1k_output=0.0015,
    ),

    # ── ZhipuAI (智谱) ──
    ModelInfo(
        id="zhipuai/glm-4-plus",
        provider="zhipuai", env_var="ZHIPUAI_API_KEY",
        tier="flagship", display_name="GLM-4 Plus",
        cost_per_1k_input=0.0007, cost_per_1k_output=0.0007,
    ),
    ModelInfo(
        id="zhipuai/glm-4-flash",
        provider="zhipuai", env_var="ZHIPUAI_API_KEY",
        tier="flash", display_name="GLM-4 Flash",
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0001,
    ),

    # ── SiliconFlow (硅基流动) ──
    ModelInfo(
        id="siliconflow/Qwen/Qwen2.5-72B-Instruct",
        provider="siliconflow", env_var="SILICONFLOW_API_KEY",
        tier="standard", display_name="Qwen 2.5 72B (SiliconFlow)",
        cost_per_1k_input=0.0006, cost_per_1k_output=0.0006,
    ),
    ModelInfo(
        id="siliconflow/Qwen/Qwen2.5-7B-Instruct",
        provider="siliconflow", env_var="SILICONFLOW_API_KEY",
        tier="flash", display_name="Qwen 2.5 7B (SiliconFlow)",
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0001,
    ),

    # ── Moonshot (月之暗面 / Kimi) ──
    ModelInfo(
        id="moonshot/moonshot-v1-auto",
        provider="moonshot", env_var="MOONSHOT_API_KEY",
        tier="standard", display_name="Kimi (Moonshot)",
        cost_per_1k_input=0.001, cost_per_1k_output=0.001,
    ),

    # ── Baichuan (百川) ──
    ModelInfo(
        id="baichuan/Baichuan4",
        provider="baichuan", env_var="BAICHUAN_API_KEY",
        tier="standard", display_name="Baichuan 4",
        cost_per_1k_input=0.001, cost_per_1k_output=0.001,
    ),

    # ── Yi / Lingyiwanwu (零一万物) ──
    ModelInfo(
        id="yi/yi-large",
        provider="yi", env_var="YI_API_KEY",
        tier="flagship", display_name="Yi Large",
        cost_per_1k_input=0.003, cost_per_1k_output=0.003,
    ),

    # ── Perplexity ──
    ModelInfo(
        id="perplexity/sonar-pro",
        provider="perplexity", env_var="PERPLEXITYAI_API_KEY",
        tier="flagship", display_name="Sonar Pro",
        cost_per_1k_input=0.003, cost_per_1k_output=0.015,
    ),
    ModelInfo(
        id="perplexity/sonar",
        provider="perplexity", env_var="PERPLEXITYAI_API_KEY",
        tier="standard", display_name="Sonar",
        cost_per_1k_input=0.001, cost_per_1k_output=0.001,
    ),

    # ── DashScope / 阿里云 (通义千问) ──
    ModelInfo(
        id="dashscope/qwen-max",
        provider="dashscope", env_var="DASHSCOPE_API_KEY",
        tier="flagship", display_name="Qwen Max (DashScope)",
        cost_per_1k_input=0.002, cost_per_1k_output=0.006,
    ),
    ModelInfo(
        id="dashscope/qwen-plus",
        provider="dashscope", env_var="DASHSCOPE_API_KEY",
        tier="standard", display_name="Qwen Plus (DashScope)",
        cost_per_1k_input=0.0004, cost_per_1k_output=0.0012,
    ),
    ModelInfo(
        id="dashscope/qwen-turbo",
        provider="dashscope", env_var="DASHSCOPE_API_KEY",
        tier="flash", display_name="Qwen Turbo (DashScope)",
        cost_per_1k_input=0.0001, cost_per_1k_output=0.0003,
    ),

    # ── Volcengine / 火山引擎 (豆包) ──
    ModelInfo(
        id="volcengine/doubao-pro-256k",
        provider="volcengine", env_var="VOLCENGINE_API_KEY",
        tier="standard", display_name="豆包 Pro",
        cost_per_1k_input=0.0007, cost_per_1k_output=0.0013,
    ),

    # ── Minimax ──
    ModelInfo(
        id="minimax/MiniMax-Text-01",
        provider="minimax", env_var="MINIMAX_API_KEY",
        tier="standard", display_name="MiniMax Text 01",
        cost_per_1k_input=0.0004, cost_per_1k_output=0.0016,
    ),
]

# ── Custom provider 注册 (需要 api_base 的 Provider) ──
# 这些 Provider 在 llm/provider.py 的 CUSTOM_PROVIDERS 中也需要注册
CUSTOM_PROVIDER_CONFIGS = {
    "siliconflow": {
        "api_base": "https://api.siliconflow.cn/v1",
        "litellm_prefix": "openai",
    },
    "zhipuai": {
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "litellm_prefix": "openai",
    },
    "moonshot": {
        "api_base": "https://api.moonshot.cn/v1",
        "litellm_prefix": "openai",
    },
    "baichuan": {
        "api_base": "https://api.baichuan-ai.com/v1",
        "litellm_prefix": "openai",
    },
    "yi": {
        "api_base": "https://api.lingyiwanwu.com/v1",
        "litellm_prefix": "openai",
    },
    "dashscope": {
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "litellm_prefix": "openai",
    },
    "volcengine": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "litellm_prefix": "openai",
    },
    "minimax": {
        "api_base": "https://api.minimax.chat/v1",
        "litellm_prefix": "openai",
    },
}


def detect_available_models() -> list[ModelInfo]:
    """扫描环境变量，返回所有可用模型列表"""
    available = []
    for model in MODEL_REGISTRY:
        key = os.environ.get(model.env_var, "").strip()
        if key:
            available.append(model)
    return available


def detect_available_providers() -> dict[str, list[ModelInfo]]:
    """按 Provider 分组返回可用模型"""
    models = detect_available_models()
    providers: dict[str, list[ModelInfo]] = {}
    for m in models:
        if m.provider not in providers:
            providers[m.provider] = []
        providers[m.provider].append(m)
    return providers


@dataclass
class AutoAssignment:
    """自动分配结果"""
    thinker_models: list[str] = field(default_factory=list)
    judge_models: list[str] = field(default_factory=list)
    red_team_models: list[str] = field(default_factory=list)
    swarm_models: list[str] = field(default_factory=list)
    embedding_model: str = "text-embedding-3-small"
    available_providers: list[str] = field(default_factory=list)
    total_models: int = 0


def auto_assign_models(
    population_size: int = 10,
    mode: str = "deep",
) -> AutoAssignment:
    """根据检测到的 API Key 自动分配模型到各角色.

    分配策略:
    - Thinker: 所有可用的 standard + flagship 模型 (多样性最重要)
    - Judge: 所有可用的 standard + flagship 模型 (多角度评审)
    - Red Team: flagship 模型优先 (攻击力需要最强)
    - Swarm: flash 模型优先 (成本敏感，批量操作)
    - Embedding: 如果有 OpenAI key 用 text-embedding-3-small
    """
    available = detect_available_models()
    if not available:
        # 没有检测到任何 API Key
        return AutoAssignment()

    providers = set(m.provider for m in available)

    # 按 tier 分组
    flagships = [m for m in available if m.tier == "flagship"]
    standards = [m for m in available if m.tier == "standard"]
    flashes = [m for m in available if m.tier == "flash"]

    # ── Judge: standard + flagship (每个 Provider 选 1 个) ──
    judge_candidates = flagships + standards
    judge_models = _pick_diverse(judge_candidates, max_count=7)

    # ── Thinker: standard + flagship, 按 population_size 分配 ──
    thinker_candidates = standards + flagships
    if not thinker_candidates:
        thinker_candidates = flashes
    thinker_models = _distribute_for_population(thinker_candidates, population_size)

    # ── Red Team: flagship 优先 ──
    red_team_candidates = flagships + standards
    red_team_models = _pick_diverse(red_team_candidates, max_count=3)

    # ── Swarm: flash 优先 (最便宜) ──
    swarm_candidates = flashes + standards
    if not swarm_candidates:
        swarm_candidates = flagships
    swarm_models = _pick_diverse(swarm_candidates, max_count=3)

    # ── Embedding ──
    embedding_model = "text-embedding-3-small"
    # OpenAI embedding 需要 OPENAI_API_KEY
    # 如果没有，用 Gemini 或其他支持 embedding 的
    if not os.environ.get("OPENAI_API_KEY"):
        if os.environ.get("GEMINI_API_KEY"):
            embedding_model = "gemini/text-embedding-004"
        elif os.environ.get("COHERE_API_KEY"):
            embedding_model = "cohere/embed-english-v3.0"

    # 如果某个角色没分配到模型，用全局 fallback
    fallback = available[0].id
    if not judge_models:
        judge_models = [fallback]
    if not thinker_models:
        thinker_models = [fallback] * min(population_size, 5)
    if not red_team_models:
        red_team_models = [fallback]
    if not swarm_models:
        swarm_models = [fallback]

    return AutoAssignment(
        thinker_models=thinker_models,
        judge_models=judge_models,
        red_team_models=red_team_models,
        swarm_models=swarm_models,
        embedding_model=embedding_model,
        available_providers=sorted(providers),
        total_models=len(available),
    )


def _pick_diverse(candidates: list[ModelInfo], max_count: int) -> list[str]:
    """从候选模型中选出多样化的子集 (每个 Provider 最多 1 个)"""
    seen_providers = set()
    picked = []
    for m in candidates:
        if m.provider not in seen_providers:
            picked.append(m.id)
            seen_providers.add(m.provider)
        if len(picked) >= max_count:
            break
    return picked


def _distribute_for_population(
    candidates: list[ModelInfo],
    population_size: int,
) -> list[str]:
    """将模型按种群大小平均分配

    例如: 3 个模型, population=12 → 每个模型 4 个
    """
    if not candidates:
        return []

    # 每个 Provider 选一个代表
    diverse = []
    seen = set()
    for m in candidates:
        if m.provider not in seen:
            diverse.append(m)
            seen.add(m.provider)

    if not diverse:
        return []

    # 均匀分配
    result = []
    per_model = max(1, population_size // len(diverse))
    for m in diverse:
        result.extend([m.id] * per_model)

    # 补齐到 population_size
    while len(result) < population_size:
        result.append(diverse[len(result) % len(diverse)].id)

    return result[:population_size]


def format_detection_report(assignment: AutoAssignment) -> str:
    """格式化自动检测报告 (用于 CLI 显示)"""
    if not assignment.available_providers:
        return "未检测到任何 API Key"

    lines = []
    lines.append(f"检测到 {len(assignment.available_providers)} 个 Provider, "
                  f"{assignment.total_models} 个可用模型")
    lines.append(f"Provider: {', '.join(assignment.available_providers)}")
    lines.append("")

    # 去重显示
    lines.append(f"Thinker: {', '.join(sorted(set(assignment.thinker_models)))}")
    lines.append(f"Judge:   {', '.join(assignment.judge_models)}")
    lines.append(f"Red Team:{', '.join(assignment.red_team_models)}")
    lines.append(f"Swarm:   {', '.join(assignment.swarm_models)}")
    lines.append(f"Embed:   {assignment.embedding_model}")

    return "\n".join(lines)
