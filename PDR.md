# EvoHive - Product Design Record (PDR)

## 项目概述

EvoHive: 进化式集体智能引擎。将自然选择算法应用于LLM Agent群体，通过多角色竞争、交叉重组、变异进化，产出超越单一模型直接回答的专业方案。

---

## 已完成功能

### M1: 核心进化引擎（v0.1.0）
- [x] Thinker角色多样化生成（persona / knowledge_bias / constraint）
- [x] 初始种群并发生成
- [x] 锦标赛选择 + 精英保留
- [x] 强制基因提取 + 交叉重组（禁止折中，保持锐度）
- [x] 5种变异策略（假设反转/受众切换/极端放大/约束注入/类比迁移）
- [x] 趋同检测（Jaccard相似度 > 0.7 自动终止）
- [x] 多样性加分（diversity_bonus 防止种群坍缩）

### M1.5: Judge可靠性（v0.1.0）
- [x] Judge可靠性预实验（多轮评审一致性检测）
- [x] 严苛Judge Prompt（强制分差、评分锚定、自检机制）
- [x] 多轮评审取中位数去噪

### M2: 质量提升（v0.2.0）
- [x] 深度扩写模块（Top 1 方案 → 完整执行计划）
- [x] 移除所有字数限制（300-500字 → 不限篇幅，要求专业咨询报告水准）
- [x] max_tokens 全面提升（1500 → 4000）
- [x] Prompt去元语言（禁止"基因""方案A/B"等内部术语泄露）
- [x] Baseline对照（同模型直接回答 vs 进化结果对比）

### M3: 多Judge陪审团制度（v2.0.0）
- [x] 多模型独立评审: 支持N个不同模型各自独立评分（数量无限制）
- [x] Pairwise对比评审: 两两对比替代绝对打分，LLM对比判断更准确
- [x] Elo Rating排名: 从两两对比结果计算Elo分数（国际象棋排名算法）
- [x] 叠加方案: 多模型 × 对比评审 = N个Judge各自做pairwise comparison，综合Elo排名
- [x] 位置偏差消除: 随机交换AB位置，防止LLM的位置偏好

### M4: 淘汰反馈遗传记忆（v2.0.0）
- [x] Judge淘汰时提取具体失败原因（每个方案最多3条）
- [x] 失败原因作为遗传记忆注入下一代crossover prompt
- [x] EvolutionMemory类管理跨代记忆，支持记忆衰减（默认2代）
- [x] 下一代强制规避: "前代方案因为X/Y/Z原因被淘汰，你必须避免同样的问题"
- [x] 最多提取5个最差方案的失败原因（控制API成本）

### M5: 用户自定义模型配置（v2.0.0）
- [x] YAML配置文件支持（evohive.yaml）
- [x] Judge模型完全自定义（数量无限制）
- [x] Thinker模型完全自定义（数量无限制，支持count参数）
- [x] 红队模型完全自定义
- [x] CLI参数支持: `--judge-models`, `--red-team-models`（逗号分隔）
- [x] 功能开关: 所有v2.0功能均可通过CLI单独禁用（`--no-red-team`, `--no-debate`等）
- [x] 向下兼容: 不配置则默认单模型DeepSeek

### M6: 红队攻击系统（v2.0.0）
- [x] 专职破坏者Agent: 独立于Judge，唯一职责是找方案致命缺陷
- [x] 攻击维度: 核心假设、数据支撑、可行性、成本估算、竞争壁垒
- [x] 多模型红队: 每个方案被每个红队模型各攻击一次
- [x] 脆弱度评分: 0-1分，影响最终fitness（低脆弱度加分，高脆弱度降分）
- [x] 仅对Top 5方案执行（控制成本）

### M7: 强制辩论淘汰赛（v2.0.0）
- [x] 两两对决: Top 5方案随机配对，强制面对面辩论
- [x] 完整攻防回合: A攻击B → B防守 → B攻击A → A防守
- [x] 多Judge裁判: 不同Judge模型各自独立判定辩论胜负
- [x] 辩论Elo积分: 从多轮辩论结果计算综合排名
- [x] 辩论分数融合: 70%原始fitness + 30%辩论Elo

### M8: 极端压力测试（v2.0.0）
- [x] 8种预设压力场景（预算砍90%/团队只剩1人/竞争对手抄袭/市场萎缩50%等）
- [x] 每个Top 5方案随机施加3个极端条件
- [x] 韧性评分: 多场景综合韧性，影响最终fitness
- [x] 详细输出: 每个场景的存活判定 + 影响分析 + 失效部分

### M9: 新血注入 + 反同质化猎杀（v2.0.0）
- [x] 反同质化猎杀: 新方案与现有种群Jaccard相似度>0.6直接杀掉
- [x] 精英豁免: 仅Top 1享受猎杀豁免
- [x] 新血注入: 被杀掉的位置由全新随机方案填充
- [x] 自动触发: 每2代检测种群多样性，相似度>0.5时强制注入
- [x] 与猎杀联动: 被猎杀的数量=新血注入的数量

### M10: 方案对话模式（v2.0.0）
- [x] 进化完成后自动进入对话模式
- [x] 基于Top 1方案的交互式追问（局部修改/风险追问/场景适配/细节展开）
- [x] 完整对话历史管理（多轮上下文）
- [x] 携带原始问题 + 进化最优方案作为上下文
- [x] 可通过 `--no-dialogue` 跳过

### M11: 多工具分章节深度扩写（v2.0.0）
- [x] 章节大纲自动生成（3-6章）
- [x] 逐章节并发展开（每章节独立LLM调用）
- [x] 工具接口架构: WebSearchTool / CompetitorAnalysisTool / CostEstimationTool（占位实现）
- [x] 回退机制: 大纲生成失败时自动回退到原始单次扩写

### 基础设施
- [x] CLI命令行界面（Typer + Rich）
- [x] LiteLLM统一模型调用
- [x] 异步并发架构（asyncio + Semaphore限流）
- [x] JSON结果持久化
- [x] YAML配置文件支持（evohive.yaml）
- [x] GitHub仓库: https://github.com/0xF69/evohive1.0 (private)

---

## 未完成功能

### 🟡 P1: 进化过程可视化

**设计方案**:
- 进化树: 哪些方案交叉产生了更强的后代
- 适应度曲线: 每代最优/平均/最差的变化趋势
- 实时淘汰展示: 弱方案标记淘汰，强方案存活

---

### 🟡 P1: A/B对比输出

**设计方案**: 最终输出并排展示 Baseline（同模型直接回答）vs 进化方案，差异高亮，让用户一眼看到进化带来的增量价值。

---

## 📌 接下来要做的

### 🔴 P0: Web前端界面【接下来要做】

**问题**: 纯CLI工具无法吸引大量用户，开源项目没有视觉反馈很难获得星标。MiroFish 35k stars的核心原因之一就是有前端。

**设计方案**:
1. **进化过程实时可视化**: 每代方案的适应度曲线、淘汰/存活的实时动画、进化树（谁和谁交叉产生了更强后代）
2. **分步向导式体验**: 输入问题 → 观看进化过程 → 查看结果 → 交互追问
3. **方案对比面板**: 多个方案并排展示，差异高亮
4. **技术选型**: Vue/React前端 + Python后端API（FastAPI）

**预期效果**: 让用户"看到"进化过程的震撼，大幅提升项目吸引力和传播性。

---

### 🔴 P0: 多工具深度扩写接入真实API【接下来要做】

**问题**: 当前多工具扩写的工具为占位实现，无法实际联网搜索。需要接入真实的搜索API。

**设计方案**:
1. 接入Serper/Tavily/Exa等搜索API，实现WebSearchTool
2. 扩写时自动搜索真实案例、竞品信息、行业数据
3. 最终输出标注信息来源，区分"LLM推理"和"真实数据"

**预期效果**: 最终输出有据可查，真正超越GPT直接回答。

---

### 🟡 P1: 发布到PyPI【接下来要做】

**问题**: 当前安装流程繁琐，门槛太高。

**设计方案**:
1. **发布到PyPI**: 用户只需 `pip install evohive`
2. **完整使用流程简化为3行**:
   ```bash
   pip install evohive
   export DEEPSEEK_API_KEY=sk-xxx
   evohive evolve -p "你的问题"
   ```
3. **自动化发布**: GitHub Actions CI/CD，push tag自动发布新版本

**预期效果**: 安装门槛降到最低。

---

## 技术栈

| 组件 | 技术选型 |
|---|---|
| 语言 | Python 3.10+ |
| CLI | Typer + Rich |
| LLM调用 | LiteLLM（统一多模型接口） |
| 数据模型 | Pydantic v2 |
| 异步 | asyncio |
| 配置 | YAML (PyYAML) |
| 默认模型 | DeepSeek (deepseek-chat) |
| 支持模型 | 任何LiteLLM兼容模型（DeepSeek/Qwen/GLM/Gemini/Llama/GPT/Claude...） |

---

## 项目结构（v2.0.0）

```
evohive/
├── cli.py                          # CLI入口（v2.0 含对话模式）
├── config.py                       # YAML配置文件加载
├── engine/
│   ├── evolution.py                # 进化主循环（v2.0 集成所有新功能）
│   ├── genesis.py                  # 初始种群生成
│   ├── selection.py                # 锦标赛选择
│   ├── crossover.py                # 交叉重组（v2.0 支持遗传记忆注入）
│   ├── mutation.py                 # 变异策略
│   ├── judge.py                    # 绝对评审系统
│   ├── pairwise_judge.py           # [v2.0] Pairwise对比 + Elo Rating
│   ├── elimination_memory.py       # [v2.0] 淘汰反馈遗传记忆
│   ├── diversity_guard.py          # [v2.0] 反同质化猎杀 + 新血注入
│   ├── red_team.py                 # [v2.0] 红队攻击系统
│   ├── debate.py                   # [v2.0] 强制辩论淘汰赛
│   ├── pressure_test.py            # [v2.0] 极端压力测试
│   ├── dialogue.py                 # [v2.0] 方案对话模式
│   ├── tool_refine.py              # [v2.0] 多工具分章节深度扩写
│   ├── baseline.py                 # 基线对照
│   └── refine.py                   # 原始深度扩写
├── models/                         # Pydantic数据模型
├── prompts/                        # Prompt模板（14个模板文件）
│   ├── pairwise_prompts.py         # [v2.0] 对比评审
│   ├── elimination_prompts.py      # [v2.0] 淘汰反馈
│   ├── red_team_prompts.py         # [v2.0] 红队攻击
│   ├── debate_prompts.py           # [v2.0] 辩论
│   ├── pressure_prompts.py         # [v2.0] 压力测试
│   ├── dialogue_prompts.py         # [v2.0] 对话模式
│   └── ...                         # 其他已有模板
├── llm/                            # LLM调用接口
└── templates/
```

---

## CLI使用方式

### 基础用法（所有v2.0功能默认开启）
```bash
evohive evolve -p "你的问题"
```

### 多Judge模型
```bash
evohive evolve -p "问题" --judge-models "deepseek/deepseek-chat,qwen/qwen-turbo,google/gemini-2.0-flash"
```

### 独立红队模型
```bash
evohive evolve -p "问题" --red-team-models "qwen/qwen-turbo,google/gemini-2.0-flash"
```

### 按需关闭功能
```bash
evohive evolve -p "问题" --no-red-team --no-debate --no-pressure --no-dialogue
```

### YAML配置文件
```bash
evohive evolve -p "问题" --config evohive.yaml
```

### YAML配置文件示例
```yaml
judges:
  - model: deepseek/deepseek-chat
  - model: qwen/qwen-turbo
  - model: google/gemini-2.0-flash
thinkers:
  - model: deepseek/deepseek-chat
    count: 5
  - model: qwen/qwen-turbo
    count: 5
red_team:
  - model: qwen/qwen-turbo
  - model: google/gemini-2.0-flash

enable_pairwise_judge: true
enable_elimination_memory: true
enable_diversity_guard: true
enable_fresh_blood: true
enable_red_team: true
enable_debate: true
enable_pressure_test: true

homogeneity_threshold: 0.6
fresh_blood_interval: 2
convergence_threshold: 0.7
```

---

## 版本历史

| 版本 | 日期 | 关键变更 |
|---|---|---|
| v0.1.0 | 2026-03-17 | 核心进化引擎 + CLI + Judge可靠性 |
| v0.2.0 | 2026-03-19 | 移除字数限制 + 深度扩写 + Prompt去元语言 + 质量提升 |
| v2.0.0 | 2026-03-20 | 多Judge陪审团(Pairwise+Elo) + 淘汰遗传记忆 + 红队攻击 + 辩论淘汰赛 + 极端压力测试 + 反同质化猎杀 + 新血注入 + 方案对话模式 + 多工具分章节扩写 + YAML配置 + 用户自定义模型 |
