# EvoHive v3.0 — 进化式集体智能引擎

**让 1000 个 AI 互相厮杀，活下来的就是最好的方案。**

将自然选择算法应用于 LLM Agent 群体。通过 Swarm 千级策略探索、多代竞争进化、交叉重组、变异、三层对抗测试，产出远超单一模型直接回答的专业方案。

---

## v3.0 新特性

### Swarm 层 — 千级 Agent 策略探索
- 一次生成 500-1000 个轻量策略种子（每个 50-100 token）
- Embedding 语义聚类，自动发现 30-50 个策略方向
- 选出代表性种子扩展为完整方案，送入进化引擎
- 成本极低：1000 seeds ≈ $0.05-0.10

### 语义 Embedding 相似度
- 替代原始 Jaccard 词袋相似度
- 基于 Embedding Cosine Similarity 的收敛检测、同质剔除、多样性评分
- 失败时自动回退到 Jaccard

### 瑞士轮锦标赛
- O(N log N) 替代 O(N²) 全配对 Elo
- 50 个 Agent 的评估从 1225 次降到 ~175 次（减少 85%）
- 国际象棋赛制，数学上证明可靠

### 事件流架构
- 每个进化阶段发出结构化事件
- 为前端 WebSocket 实时可视化做好准备
- 30+ 种预定义事件类型

### Web Search 真实数据
- 支持 Tavily / Serper 搜索 API
- 方案生成和深度扩写均可注入真实市场数据
- 未配置 API 时自动跳过，不中断流程

### 快速/深度双模式
| 模式 | Swarm | 进化 | 后进化 | 成本 | 时间 |
|------|-------|------|--------|------|------|
| fast | 200 seeds | 1 代 | 仅红队 | ~$0.5 | 1-2 分钟 |
| deep | 500+ seeds | 3 代 | 红队+辩论+压力 | ~$3-5 | 8-15 分钟 |

### 实时成本追踪与预算控制
- 每次 API 调用实时记录 token 消耗和费用
- 按 Provider / 阶段细分成本报表
- `--budget` 设置预算上限，超出自动停止
- 预运行成本估算

### 检查点与崩溃恢复
- 每代进化后原子写入检查点
- 程序崩溃后可从最后完成的代恢复继续
- 保留最近 2 个检查点

### 断路器与自动降级
- Per-Provider 断路器（3 次失败触发熔断，60 秒后探测恢复）
- 主模型失败自动切换到其他 Provider 的可用模型
- 预飞行检查 + 交互式确认不可用模型
- 按错误类型区分重试策略（Rate Limit / Server / Timeout / Auth）

### 自适应进化参数
- 检测停滞 → 自动提升变异率
- 检测快速改进 → 降低变异率利用当前方向
- mutation_rate / survival_rate 动态调整

### 30+ 模型支持
- **14+ Provider**: OpenAI / Anthropic / Gemini / DeepSeek / Groq / Mistral / xAI / Together AI / Fireworks / Cohere / ZhipuAI / SiliconFlow / Moonshot / Perplexity 等
- 自动检测环境变量中已配置的 API Key
- 按角色需求（Thinker / Judge / Red Team / Swarm）智能分配模型

---

## 核心进化引擎

- 多模型 Thinker 角色生成（persona / knowledge_bias / constraint）
- 锦标赛选择 + 精英保留
- 强制基因提取 + 交叉重组（禁止折中，保持锐度）
- 5 种变异策略：假设反转 / 受众切换 / 极端放大 / 约束注入 / 类比迁移
- 多 Judge 陪审团 + Pairwise Elo 排名
- 淘汰反馈遗传记忆（2 代衰减窗口）
- 反同质化猎杀 + 新鲜血液注入
- 三层对抗：红队攻击 → 辩论淘汰赛 → 极端压力测试
- 多章节工具增强深度扩写
- 交互式方案对话模式

---

## 快速开始

### 环境要求

- Python 3.10+
- 至少一个 LLM API Key

### 安装

```bash
git clone https://github.com/0xF69/evohive1.0.git
cd evohive1.0
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
