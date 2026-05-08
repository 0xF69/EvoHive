# EvoHive — Evolutionary Answer Engine

EvoHive 是一个可进化、可验证、可控成本、可复盘的多模型答案引擎。

它不是让一个大模型直接给出一次回答，而是把同一个问题拆成多条候选答案路径，让多个 Thinker 生成方案，再经过评审、选择、交叉、变异、验证和预算控制，最终产出一个更可靠、可追踪的结果。

适合处理单次 LLM 回答容易浅、容易偏、难以验证的问题，例如产品策略、代码方案、研究分析、决策推演和复杂执行计划。

---

## 当前核心能力

### 答案进化引擎
- 多 Thinker 候选方案生成
- 锦标赛选择、精英保留、交叉重组、变异
- 反同质化、多样性控制、新血注入
- 支持多代进化，也支持 quick 模式快速探索

### 多模型编排
- 支持 Thinker、Judge、Red Team、Swarm 等角色分工
- 支持 OpenAI、Anthropic、Gemini、DeepSeek、Groq、Mistral、xAI、Together、Fireworks、Cohere、Perplexity、SiliconFlow、Moonshot 等 LiteLLM 兼容 Provider
- 支持预飞行检查、Provider 断路器、失败重试和跨 Provider fallback

### 评审与排名
- 支持自定义 Judge 维度
- 支持多轮评分取中位数
- 支持 Pairwise Elo 排名和 Swiss Tournament，减少全配对评审成本
- 支持 baseline vs evolution 质量对比

### 可验证输出
- 自动从最终答案中抽取 claims
- 生成 verification report 和 claim verification report
- 标记高风险 claim、弱证据 claim、需要外部验证的 claim
- 可选开启搜索验证，用 Tavily / Serper 为弱 claim 补充证据

### 图结构与复盘
- `lineage_graph`: 记录候选方案的父代、交叉、变异和淘汰路径
- `answer_graph`: 把问题、答案量子、最终答案、claims、验证节点组织成图结构
- `trajectory_log`: 记录每个阶段、模型角色、动作和摘要
- `trajectory_replay`: 为 API 消费方提供可复盘的时间线数据

### 成本与预算控制
- 记录每次 LLM 调用的 input/output tokens、调用次数和估算成本
- 生成 `cost_breakdown`、`resource_report`、`token_budget_report`
- 支持用户切换 token 预算控制: `off`、`auto`、`relaxed`、`strict`
- 预算紧张时可以裁剪输出、跳过昂贵阶段或提前停止

### 后端服务能力
- FastAPI HTTP API
- WebSocket 实时事件流
- Python SDK
- CLI
- 运行结果 artifact 持久化
- checkpoint 保存与恢复

---

## 不属于当前版本的内容

- 不是模型训练平台
- 不是图片生成产品
- 不是已经完成的前端可视化产品
- 不保证答案一定正确，而是提供更强的生成、比较、验证和复盘机制

---

## 快速开始

### 环境要求

- Python 3.10+
- 至少一个 LLM API Key

### 安装

```bash
git clone https://github.com/0xF69/EvoHive.git
cd EvoHive
pip install -e .
```

### 设置环境变量

```bash
# 必须 — 至少一个 LLM
export DEEPSEEK_API_KEY="sk-xxx"

# 推荐 — Embedding (语义相似度)
export OPENAI_API_KEY="sk-xxx"

# 可选 — 多模型 + Web Search
export GROQ_API_KEY="gsk_xxx"
export GEMINI_API_KEY="AIzaSyxxx"
export TAVILY_API_KEY="tvly-xxx"
```

### 运行

```bash
# 快速模式（1-2 分钟）
evohive evolve "你的问题" -m fast

# 深度模式（8-15 分钟，默认）
evohive evolve "你的问题" -m deep

# 使用 YAML 配置文件
evohive evolve "你的问题" -c evohive.yaml

# 查看所有支持的模型及可用性
evohive models

# 查看历史进化运行
evohive history
```

### 完整参数示例

```bash
evohive evolve \
  "给一个新茶饮品牌设计三线城市市场进入策略" \
  -m deep \
  -n 50 \
  -g 3 \
  --swarm-count 1000 \
  --budget 5.0 \
  --judge-models "deepseek/deepseek-chat,gemini/gemini-2.0-flash" \
  --red-team-models "groq/llama-3.3-70b-versatile"
```

### Judge 可靠性测试

```bash
evohive judge-test -p "你的问题" --rounds 5
```

### Python SDK

```python
from evohive.models import EvolutionConfig
from evohive.engine.evolution import run_evolution
import asyncio

config = EvolutionConfig(
    problem="你的问题",
    population_size=50,
    generations=3,
    thinker_models=["deepseek/deepseek-chat"],
    judge_models=["deepseek/deepseek-chat"],
    mode="deep",
    token_budget_control="auto",  # off | auto | relaxed | strict
    enable_token_budget_control=True,
)

result = asyncio.run(run_evolution(config, budget_limit=5.0))
print(result.final_top_solutions)
```

---

## 架构概览

```
用户问题
    ↓
[Web Search] → 获取真实数据上下文
    ↓
[Swarm 层] 1000个轻量Agent → 语义聚类 → 50个策略方向
    ↓
[进化引擎] 50个完整方案 × 3代进化
    ├─ 绝对评审 + 瑞士轮Elo排名
    ├─ 锦标赛淘汰 + 淘汰记忆
    ├─ 交叉重组 + 变异
    └─ 反同质化 + 新血注入
    ↓
[后进化三层对抗]
    ├─ 红队攻击 (Top 5)
    ├─ 辩论淘汰赛 (Top 5)
    └─ 极端压力测试 (Top 5)
    ↓
[深度扩写] Top 1 → 多章节执行计划 (含真实数据)
    ↓
[交互对话] 针对最优方案追问
```

## 项目结构

```
evohive/
├── cli.py                    # CLI入口 (Typer + Rich)
├── config.py                 # YAML配置系统
├── sdk.py                    # Python SDK接口
├── engine/
│   ├── evolution.py          # 进化主循环 (v3.0 编排)
│   ├── swarm.py              # Swarm千级Agent种子探索
│   ├── embedding.py          # 语义Embedding相似度
│   ├── swiss_tournament.py   # 瑞士轮锦标赛
│   ├── events.py             # 事件流系统 (30+事件类型)
│   ├── web_search.py         # Web Search工具
│   ├── cost_tracker.py       # 实时成本追踪与预算控制
│   ├── checkpoint.py         # 检查点与崩溃恢复
│   ├── adaptive.py           # 自适应参数控制
│   ├── logger.py             # 结构化JSON日志
│   ├── genesis.py            # Thinker + 初始种群生成
│   ├── selection.py          # 锦标赛选择
│   ├── crossover.py          # 交叉重组
│   ├── mutation.py           # 5种变异策略
│   ├── judge.py              # 评审系统 (含Embedding多样性)
│   ├── pairwise_judge.py     # Pairwise Elo + 统一锦标赛接口
│   ├── elimination_memory.py # 淘汰反馈遗传记忆
│   ├── diversity_guard.py    # 反同质化 + 新血注入
│   ├── red_team.py           # 红队攻击
│   ├── debate.py             # 辩论淘汰赛
│   ├── pressure_test.py      # 极端压力测试
│   ├── tool_refine.py        # 多章节工具增强扩写
│   ├── baseline.py           # 基线对照
│   └── dialogue.py           # 交互对话
├── models/                   # Pydantic数据模型
├── prompts/                  # Prompt模板
├── llm/
│   ├── provider.py           # LiteLLM封装 + 断路器 + 自动降级
│   └── model_registry.py     # 30+模型定义与自动检测
└── tests/                    # 测试套件 (94 tests)
```

---

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| 语言 | Python 3.10+ |
| CLI | Typer + Rich |
| LLM 调用 | LiteLLM (统一多模型接口) |
| 数据模型 | Pydantic v2 |
| 异步 | asyncio |
| Web Search | Tavily / Serper API |
| Embedding | OpenAI text-embedding-3-small |
| 配置 | PyYAML |

---

## 许可证

[MIT](LICENSE)
