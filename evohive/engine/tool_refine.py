"""多工具深度扩写 — Tool-Augmented Refinement"""

import asyncio
from evohive.llm import call_llm
from evohive.prompts.refine_prompts import REFINE_SYSTEM, REFINE_USER


# ── Tool接口定义 ──────────────────────────────

class RefineTool:
    """扩写工具基类"""
    name: str = "base_tool"
    description: str = "基础工具"

    async def execute(self, query: str) -> str:
        raise NotImplementedError


class WebSearchTool(RefineTool):
    """联网搜索工具 — 使用真实搜索API

    支持 Tavily/Serper API。未配置时返回空结果。
    """
    name = "web_search"
    description = "搜索互联网获取真实数据和案例"

    async def execute(self, query: str) -> str:
        try:
            from evohive.engine.web_search import web_search
            results = await web_search(query, max_results=3)
            if not results:
                return ""
            lines = []
            for r in results:
                lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')}")
            return "\n".join(lines)
        except Exception:
            return ""


class CompetitorAnalysisTool(RefineTool):
    """竞品分析工具（占位实现）"""
    name = "competitor_analysis"
    description = "分析竞品信息和市场格局"

    async def execute(self, query: str) -> str:
        return f"[竞品分析工具未配置] 无法分析: {query}。"


class CostEstimationTool(RefineTool):
    """成本估算工具（占位实现）"""
    name = "cost_estimation"
    description = "估算执行成本和预算"

    async def execute(self, query: str) -> str:
        return f"[成本估算工具未配置] 无法估算: {query}。"


# ── 工具注册表 ──────────────────────────────

AVAILABLE_TOOLS: list[RefineTool] = [
    WebSearchTool(),
    CompetitorAnalysisTool(),
    CostEstimationTool(),
]


# ── 分章节扩写 ──────────────────────────────

OUTLINE_SYSTEM = "你是一个方案架构师。请为方案生成一个章节大纲。"

OUTLINE_USER = """以下是一份进化最优方案，请为它设计一个深度扩写的章节大纲。

原始问题: {problem}

方案内容:
---
{solution}
---

请输出3-6个章节标题，每个章节应该是方案的一个核心模块。
格式要求（严格JSON）:
{{"chapters": ["章节1标题", "章节2标题", ...]}}"""

CHAPTER_SYSTEM = """你是一位顶级战略咨询顾问。请深度展开方案的一个章节。

要求:
- 具体的执行步骤和操作流程
- 可量化的衡量指标和成功标准（KPI）
- 时间节点和里程碑
- 潜在风险及对应的预防/应对措施
- 篇幅充分，不要省略任何重要细节"""

CHAPTER_USER = """原始问题: {problem}

完整方案概要:
---
{solution}
---

请深度展开以下章节: {chapter_title}

直接输出该章节的完整内容，使用Markdown格式。"""


async def tool_augmented_refine(
    solution_content: str,
    problem: str,
    model: str,
    search_context: str = "",
) -> str:
    """多工具分章节深度扩写

    1. 生成章节大纲
    2. 逐章节展开（每章节可调用工具获取支撑信息）
    3. 合并为完整报告

    Args:
        solution_content: 进化最优方案
        problem: 原始问题
        model: 使用的LLM模型
        search_context: 可选的预获取搜索上下文

    Returns:
        扩写后的完整方案
    """
    from evohive.llm import extract_json

    # Step 1: Generate outline
    outline_response = await call_llm(
        model=model,
        system_prompt=OUTLINE_SYSTEM,
        user_prompt=OUTLINE_USER.format(problem=problem, solution=solution_content),
        temperature=0.5,
        max_tokens=500,
        json_mode=True,
    )

    data = extract_json(outline_response)
    if data and "chapters" in data:
        chapters = data["chapters"]
    else:
        # Fallback: use original refine
        return await _fallback_refine(solution_content, problem, model, search_context)

    if not chapters:
        return await _fallback_refine(solution_content, problem, model, search_context)

    # Step 2: Expand each chapter concurrently
    sem = asyncio.Semaphore(3)

    async def expand_chapter(title: str) -> str:
        async with sem:
            # Search for chapter-relevant data
            chapter_context = ""
            if search_context:
                chapter_context = f"\n\n参考数据:\n{search_context}"
            else:
                try:
                    from evohive.engine.web_search import search_for_chapter
                    chapter_data = await search_for_chapter(title, problem)
                    if chapter_data:
                        chapter_context = f"\n\n{chapter_data}"
                except Exception:
                    pass

            user = CHAPTER_USER.format(
                problem=problem,
                solution=solution_content,
                chapter_title=title,
            )
            if chapter_context:
                user = user + chapter_context

            response = await call_llm(
                model=model,
                system_prompt=CHAPTER_SYSTEM,
                user_prompt=user,
                temperature=0.7,
                max_tokens=4000,
            )
            return response

    chapter_contents = await asyncio.gather(*[expand_chapter(ch) for ch in chapters])

    # Step 3: Assemble final document
    parts = []
    for title, content in zip(chapters, chapter_contents):
        if not content.startswith("#"):
            parts.append(f"## {title}\n\n{content}")
        else:
            parts.append(content)

    return "\n\n---\n\n".join(parts)


async def _fallback_refine(solution_content: str, problem: str, model: str, search_context: str = "") -> str:
    """回退到原始单次扩写"""
    user_prompt = REFINE_USER.format(problem=problem, solution=solution_content)
    if search_context:
        user_prompt = user_prompt + f"\n\n参考数据:\n{search_context}"
    return await call_llm(
        model=model,
        system_prompt=REFINE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=8000,
    )
