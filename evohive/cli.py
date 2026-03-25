"""EvoHive CLI入口 (v3.0)"""

__cli_build__ = "2026-03-22a"  # 用于验证是否部署了最新代码

import asyncio
import sys
import io
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from pathlib import Path


# ═══ 过滤LiteLLM的stderr垃圾输出 ═══
class _StderrFilter:
    """拦截stderr，过滤LiteLLM的调试信息（仅过滤明确的垃圾行，保留错误信息）"""
    _SPAM_PREFIXES = [
        "Give Feedback",
        "LiteLLM.Info:",
        "\nLiteLLM.Info:",
        "Provider List:",
        "\x1b[92m\x1b[1m",  # LiteLLM的ANSI彩色前缀
    ]
    _SPAM_EXACT = [
        "docs.litellm.ai",
        "BerriAI/litellm",
        "litellm.set_verbose",
        "litellm._turn_on_debug",
        "Set Verbose=True",
    ]

    def __init__(self, original):
        self._original = original

    def write(self, text):
        if not text or not text.strip():
            return self._original.write(text)
        stripped = text.strip()
        # Only filter lines that clearly start with LiteLLM spam
        if any(stripped.startswith(p) for p in self._SPAM_PREFIXES):
            return len(text)
        # Filter lines that are exact LiteLLM info strings
        if any(kw in stripped for kw in self._SPAM_EXACT):
            return len(text)
        return self._original.write(text)

    def flush(self):
        return self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name):
        return getattr(self._original, name)

sys.stderr = _StderrFilter(sys.stderr)

# 同时拦截stdout（LiteLLM有时也会通过print()输出到stdout）
class _StdoutFilter:
    """拦截stdout中LiteLLM的垃圾输出，保留正常输出"""
    _SPAM_PREFIXES = [
        "Give Feedback",
        "LiteLLM.Info:",
        "\nLiteLLM.Info:",
        "Provider List:",
        "\x1b[92m\x1b[1m",
    ]
    _SPAM_EXACT = [
        "docs.litellm.ai",
        "BerriAI/litellm",
        "litellm.set_verbose",
        "litellm._turn_on_debug",
    ]

    def __init__(self, original):
        self._original = original

    def write(self, text):
        if not text or not text.strip():
            return self._original.write(text)
        stripped = text.strip()
        if any(stripped.startswith(p) for p in self._SPAM_PREFIXES):
            return len(text)
        if any(kw in stripped for kw in self._SPAM_EXACT):
            return len(text)
        return self._original.write(text)

    def flush(self):
        return self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)

sys.stdout = _StdoutFilter(sys.stdout)

from evohive.models import EvolutionConfig, GenerationStats, Solution

app = typer.Typer(
    name="evohive",
    help="进化式集体智能引擎 — 让数百个AI大脑通过自然选择进化出最优方案 (Swarm支持)",
    no_args_is_help=True,
)
console = Console()


def _parse_judges(judges_str: str) -> list[dict]:
    """解析评审维度字符串"""
    dims = []
    for part in judges_str.split(","):
        part = part.strip()
        if ":" in part:
            name, weight = part.rsplit(":", 1)
            dims.append({
                "name": name.strip(),
                "weight": float(weight.strip()),
                "description": name.strip(),
            })
        else:
            dims.append({
                "name": part.strip(),
                "weight": 1.0,
                "description": part.strip(),
            })
    total = sum(d["weight"] for d in dims)
    if total > 0:
        for d in dims:
            d["weight"] = d["weight"] / total
    return dims


def _parse_model_list(models_str: str) -> list[str]:
    """解析逗号分隔的模型列表"""
    if not models_str:
        return []
    return [m.strip() for m in models_str.split(",") if m.strip()]


@app.command()
def evolve(
    problem: str = typer.Argument(..., help="要解决的问题"),
    population: int = typer.Option(10, "--population", "-n", help="种群大小"),
    generations: int = typer.Option(3, "--generations", "-g", help="进化代数"),
    survival_rate: float = typer.Option(0.3, "--survival-rate", help="存活率"),
    mutation_rate: float = typer.Option(0.3, "--mutation-rate", help="变异概率"),
    thinker_model: str = typer.Option("deepseek/deepseek-chat", "--thinker-model", help="Thinker使用的LLM"),
    judge_model: str = typer.Option("deepseek/deepseek-chat", "--judge-model", help="Judge使用的LLM"),
    judge_models: str = typer.Option("", "--judge-models", help="多Judge模型（逗号分隔）"),
    red_team_models: str = typer.Option("", "--red-team-models", help="红队模型（逗号分隔）"),
    judges: str = typer.Option(
        "可行性:0.3,创新性:0.25,具体性:0.25,成本效率:0.2",
        "--judges", "-j",
        help="评审维度（格式: 名称:权重,名称:权重）"
    ),
    config_file: str = typer.Option("", "--config", "-c", help="YAML配置文件路径"),
    no_pairwise: bool = typer.Option(False, "--no-pairwise", help="禁用Pairwise Elo评审"),
    no_memory: bool = typer.Option(False, "--no-memory", help="禁用淘汰反馈遗传记忆"),
    no_diversity: bool = typer.Option(False, "--no-diversity-guard", help="禁用反同质化猎杀"),
    no_red_team: bool = typer.Option(False, "--no-red-team", help="禁用红队攻击"),
    no_debate: bool = typer.Option(False, "--no-debate", help="禁用辩论淘汰赛"),
    no_pressure: bool = typer.Option(False, "--no-pressure", help="禁用极端压力测试"),
    dialogue: bool = typer.Option(True, "--dialogue/--no-dialogue", help="进化完成后进入对话模式"),
    mode: str = typer.Option("deep", "--mode", "-m", help="运行模式: fast (快速) 或 deep (深度)"),
    no_swarm: bool = typer.Option(False, "--no-swarm", help="禁用Swarm层"),
    swarm_count: int = typer.Option(500, "--swarm-count", help="Swarm种子数量"),
    no_search: bool = typer.Option(False, "--no-search", help="禁用Web搜索"),
    budget: float = typer.Option(0.0, "--budget", "-b", help="预算上限(USD), 0=不限制"),
    save: bool = typer.Option(True, "--save/--no-save", help="自动保存结果到文件"),
    output_dir: str = typer.Option("evohive_results", "--output-dir", help="结果保存目录"),
):
    """运行进化 — 让AI方案通过自然选择进化出最优方案 (v3.0)"""
    from evohive.engine.evolution import run_evolution

    # 加载YAML配置
    auto_detected = False
    if config_file:
        from evohive.config import load_config
        cfg = load_config(config_file)
        judge_models_list = cfg.get_judge_models()
        red_team_models_list = cfg.get_red_team_models()
        thinker_models_list = cfg.get_thinker_models()
        swarm_models_list = cfg.get_swarm_models()
        # 如果YAML指定了thinkers，用其总数作为种群大小
        if thinker_models_list:
            population = len(thinker_models_list)
    else:
        judge_models_list = _parse_model_list(judge_models) if judge_models else []
        red_team_models_list = _parse_model_list(red_team_models) if red_team_models else []
        thinker_models_list = []
        swarm_models_list = []

        # ── 自动检测: 当用户未手动指定模型时，扫描环境变量自动分配 ──
        if not judge_models_list and not red_team_models_list:
            from evohive.llm.model_registry import auto_assign_models, format_detection_report
            assignment = auto_assign_models(population_size=population, mode=mode)
            if assignment.available_providers:
                auto_detected = True
                thinker_models_list = assignment.thinker_models
                judge_models_list = assignment.judge_models
                red_team_models_list = assignment.red_team_models
                swarm_models_list = assignment.swarm_models
                # 用检测到的第一个 thinker 作为默认 thinker_model
                if thinker_models_list:
                    thinker_model = thinker_models_list[0]
                    population = len(thinker_models_list)

    dimensions = _parse_judges(judges)

    config = EvolutionConfig(
        problem=problem,
        population_size=population,
        generations=generations,
        survival_rate=survival_rate,
        mutation_rate=mutation_rate,
        thinker_model=thinker_model,
        judge_model=judge_model,
        judge_dimensions=dimensions,
        judge_models=judge_models_list,
        red_team_models=red_team_models_list,
        thinker_models=thinker_models_list,
        swarm_models=swarm_models_list,
        enable_pairwise_judge=not no_pairwise,
        enable_elimination_memory=not no_memory,
        enable_diversity_guard=not no_diversity,
        enable_fresh_blood=not no_diversity,
        enable_red_team=not no_red_team,
        enable_debate=not no_debate,
        enable_pressure_test=not no_pressure,
        mode=mode,
        enable_swarm=not no_swarm,
        swarm_count=swarm_count,
        enable_web_search=not no_search,
        enable_swiss_tournament=True,
    )

    if mode == "fast":
        config.generations = 1
        config.swarm_count = min(200, config.swarm_count)
        config.enable_debate = False
        config.enable_pressure_test = False
        generations = 1  # update local var used in display
        swarm_count = config.swarm_count  # update local var used in display

    # 打印配置
    console.print()

    # 自动检测报告
    if auto_detected:
        from evohive.llm.model_registry import format_detection_report
        console.print(Panel(
            format_detection_report(assignment),
            title="[bold cyan]自动检测到的 API Key 与模型分配[/bold cyan]",
            border_style="cyan",
        ))
        console.print()

    features_on = []
    if config.enable_pairwise_judge:
        features_on.append("Pairwise Elo")
    if config.enable_elimination_memory:
        features_on.append("遗传记忆")
    if config.enable_diversity_guard:
        features_on.append("反同质化")
    if config.enable_red_team:
        features_on.append("红队攻击")
    if config.enable_debate:
        features_on.append("辩论淘汰赛")
    if config.enable_pressure_test:
        features_on.append("压力测试")

    jm_display = ", ".join(judge_models_list) if judge_models_list else judge_model
    tm_display = ", ".join(sorted(set(thinker_models_list))) if thinker_models_list else thinker_model
    console.print(Panel(
        f"[bold]问题:[/bold] {problem}\n"
        f"[bold]种群:[/bold] {population} agents × {generations} 代\n"
        f"[bold]存活率:[/bold] {survival_rate:.0%}  [bold]变异率:[/bold] {mutation_rate:.0%}\n"
        f"[bold]Thinker模型:[/bold] {tm_display}\n"
        f"[bold]Judge模型:[/bold] {jm_display}\n"
        f"[bold]评审维度:[/bold] {', '.join(d['name'] for d in dimensions)}\n"
        f"[bold]v3.0功能:[/bold] {' | '.join(features_on)}\n"
        f"[bold]运行模式:[/bold] {'⚡ 快速' if mode == 'fast' else '🔬 深度'}\n"
        f"[bold]Swarm:[/bold] {'开启' if not no_swarm else '关闭'}"
        + (f" ({swarm_count} 种子)" if not no_swarm else ""),
        title="[bold green]EvoHive v3.0 进化配置[/bold green]",
        border_style="green",
    ))
    console.print()

    # 进度回调
    def on_gen_complete(gen: int, stats: GenerationStats, best: Solution):
        console.print(
            f"  [bold cyan]Generation {gen}/{generations}[/bold cyan]  "
            f"最高: {stats.best_fitness:.3f}  "
            f"平均: {stats.avg_fitness:.3f}  "
            f"存活: {stats.alive_count}  "
            f"淘汰: {stats.eliminated_count}"
        )

    def on_status(msg: str):
        console.print(f"  [dim]{msg}[/dim]")

    def on_preflight_confirm(ok_models: list[str], failed_info: list[dict]) -> bool:
        """Interactive confirmation when some models fail pre-flight check."""
        console.print()
        # Show failed models
        console.print("[bold red]以下模型不可用:[/bold red]")
        for f in failed_info:
            # Extract short error reason
            err = str(f.get("error", "unknown"))
            if "NotFoundError" in err or "not found" in err.lower():
                reason = "模型不存在或已下线"
            elif "insufficient" in err.lower() or "quota" in err.lower() or "balance" in err.lower():
                reason = "余额不足或配额用尽"
            elif "401" in err or "auth" in err.lower() or "API key" in err.lower():
                reason = "API Key 无效"
            elif "RateLimitError" in err or "429" in err:
                reason = "请求频率超限"
            else:
                reason = err[:80]
            console.print(f"  [red]✗[/red] {f['model']}  — {reason}")

        console.print()
        # Show available models
        console.print("[bold green]可用模型:[/bold green]")
        for m in ok_models:
            console.print(f"  [green]✓[/green] {m}")

        console.print()
        try:
            answer = console.input("[bold yellow]是否使用可用模型继续? (y/n): [/bold yellow]")
            return answer.strip().lower() in ("y", "yes", "是", "")
        except (EOFError, KeyboardInterrupt):
            return False

    # 运行进化
    console.print(f"[dim]build: {__cli_build__}[/dim]")
    console.print("[bold yellow]>>> 开始进化...[/bold yellow]")
    console.print()

    budget_limit = budget if budget > 0 else None

    try:
        result = asyncio.run(run_evolution(
            config,
            on_generation_complete=on_gen_complete,
            on_status=on_status,
            budget_limit=budget_limit,
            save_results=save,
            output_dir=output_dir,
            on_preflight_confirm=on_preflight_confirm,
        ))
    except Exception as e:
        if "User aborted" in str(e):
            console.print("[dim]已取消进化。[/dim]")
            return
        if "Budget exceeded" in str(e) or "BudgetExceededError" in type(e).__name__:
            console.print(Panel(
                str(e),
                title="[bold red]预算超限 — 进化已终止[/bold red]",
                border_style="red",
            ))
            return
        console.print(Panel(
            f"[bold]{type(e).__name__}[/bold]: {e}",
            title="[bold red]进化过程出错[/bold red]",
            border_style="red",
        ))
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        return

    # 输出结果
    console.print()
    console.print("[bold green]>>> 进化完成![/bold green]")
    console.print()

    # 趋同警告
    if result.early_stop_reason:
        console.print(Panel(
            result.early_stop_reason,
            title="[bold red]提前终止: 种群趋同[/bold red]",
            border_style="red",
        ))
        console.print()

    # 基线
    console.print(Panel(
        result.baseline_solution,
        title="[bold blue]基线回答（直接问LLM）[/bold blue]",
        border_style="blue",
    ))
    console.print()

    # Top方案
    for i, sol_data in enumerate(result.final_top_solutions):
        fitness = sol_data.get("fitness", 0)
        content = sol_data.get("content", "")
        mutation = sol_data.get("mutation_applied", None)
        parent_a = sol_data.get("parent_a_id", None)
        vulnerability = sol_data.get("red_team_vulnerability", None)
        resilience = sol_data.get("pressure_resilience", None)
        debate_elo = sol_data.get("debate_elo", None)

        title_parts = [f"Top {i+1}", f"fitness={fitness:.3f}"]
        if mutation:
            title_parts.append(f"变异: {mutation}")
        if parent_a:
            title_parts.append("(交叉产物)")
        if vulnerability is not None:
            title_parts.append(f"脆弱度={vulnerability:.2f}")
        if resilience is not None:
            title_parts.append(f"韧性={resilience:.2f}")
        if debate_elo is not None:
            title_parts.append(f"辩论Elo={debate_elo:.0f}")

        console.print(Panel(
            content,
            title=f"[bold green]{' | '.join(title_parts)}[/bold green]",
            border_style="green",
        ))
        console.print()

    # 红队攻击结果
    if result.red_team_results:
        console.print(Panel(
            _format_red_team_results(result.red_team_results),
            title="[bold red]红队攻击结果[/bold red]",
            border_style="red",
        ))
        console.print()

    # 压力测试结果
    if result.pressure_test_results:
        console.print(Panel(
            _format_pressure_results(result.pressure_test_results),
            title="[bold yellow]极端压力测试结果[/bold yellow]",
            border_style="yellow",
        ))
        console.print()

    # 淘汰记忆
    if result.elimination_memories:
        memories_text = "\n".join(f"  - {m}" for m in result.elimination_memories[:10])
        console.print(Panel(
            memories_text,
            title="[bold cyan]遗传记忆 — 累积的失败教训[/bold cyan]",
            border_style="cyan",
        ))
        console.print()

    # 深度扩写版 Top 1
    if result.refined_top_solution:
        console.print(Panel(
            result.refined_top_solution,
            title="[bold magenta]深度扩写版 Top 1 — 完整执行计划[/bold magenta]",
            border_style="magenta",
        ))
        console.print()

    # 统计摘要
    if result.generations_data:
        table = Table(title="进化统计")
        table.add_column("代数", style="cyan")
        table.add_column("最高适应度", style="green")
        table.add_column("平均适应度", style="yellow")
        table.add_column("存活", style="blue")
        table.add_column("淘汰", style="red")

        for stats in result.generations_data:
            table.add_row(
                str(stats.generation),
                f"{stats.best_fitness:.3f}",
                f"{stats.avg_fitness:.3f}",
                str(stats.alive_count),
                str(stats.eliminated_count),
            )

        console.print(table)

    console.print(f"\n[dim]总API调用: {result.total_api_calls}次[/dim]")
    duration = (result.finished_at - result.started_at).total_seconds()
    console.print(f"[dim]总耗时: {duration:.1f}秒[/dim]")
    if result.estimated_cost > 0:
        console.print(f"[dim]预估成本: ${result.estimated_cost:.4f}[/dim]")
    if save:
        console.print(f"[dim]结果已保存至: {output_dir}/[/dim]")

    # 对话模式
    if dialogue and config.mode != "fast" and result.refined_top_solution:
        _enter_dialogue_mode(
            result.refined_top_solution,
            problem,
            thinker_model,
        )


def _format_red_team_results(results: list[dict]) -> str:
    """格式化红队攻击结果"""
    lines = []
    for r in results:
        sid = r.get("solution_id", "?")[:8]
        vuln = r.get("overall_vulnerability", 0.5)
        attacks = r.get("attacks", [])
        lines.append(f"方案 {sid} (脆弱度: {vuln:.2f}):")
        for att in attacks[:3]:
            if isinstance(att, dict):
                target = att.get("target", "")
                attack = att.get("attack", "")
                severity = att.get("severity", "?")
                lines.append(f"  [{severity}/10] {target}: {attack}")
            else:
                lines.append(f"  - {att}")
    return "\n".join(lines) if lines else "无攻击结果"


def _format_pressure_results(results: list[dict]) -> str:
    """格式化压力测试结果"""
    lines = []
    for r in results:
        sid = r.get("solution_id", "?")[:8]
        tested = r.get("scenarios_tested", 0)
        survived = r.get("scenarios_survived", 0)
        resilience = r.get("avg_resilience", 0.5)
        lines.append(f"方案 {sid}: {survived}/{tested} 场景存活 (韧性: {resilience:.2f})")
        for detail in r.get("details", [])[:3]:
            scenario = detail.get("scenario", "")
            surv = "+" if detail.get("survives", False) else "-"
            lines.append(f"  {surv} {scenario}")
    return "\n".join(lines) if lines else "无测试结果"


def _enter_dialogue_mode(solution_content: str, problem: str, model: str):
    """进入方案对话模式"""
    from evohive.engine.dialogue import dialogue_turn

    console.print()
    console.print(Panel(
        "进化完成。你现在可以针对Top 1方案进行追问。\n"
        "输入你的问题，或输入 'quit' / 'exit' / 'q' 退出。",
        title="[bold cyan]方案对话模式[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    chat_history = []

    while True:
        try:
            user_input = console.input("[bold green]你: [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]退出对话模式[/dim]")
            break

        response = asyncio.run(dialogue_turn(
            user_input, solution_content, problem, chat_history, model,
        ))

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response})

        console.print()
        console.print(Panel(
            response,
            title="[bold cyan]EvoHive[/bold cyan]",
            border_style="cyan",
        ))
        console.print()


@app.command()
def models():
    """查看支持的模型列表及当前可用状态"""
    from evohive.llm.model_registry import MODEL_REGISTRY, detect_available_providers
    import os

    available_providers = detect_available_providers()

    table = Table(title="EvoHive 支持的模型")
    table.add_column("模型", style="cyan")
    table.add_column("Provider", style="blue")
    table.add_column("级别", style="yellow")
    table.add_column("环境变量", style="dim")
    table.add_column("状态", style="green")

    for m in MODEL_REGISTRY:
        has_key = m.provider in available_providers
        status = "[green]✓ 可用[/green]" if has_key else "[red]✗ 未配置[/red]"
        tier_display = {"flagship": "旗舰", "standard": "标准", "flash": "轻量"}.get(m.tier, m.tier)
        table.add_row(m.id, m.provider, tier_display, m.env_var, status)

    console.print(table)
    console.print()
    if available_providers:
        console.print(f"[green]已配置 {len(available_providers)} 个 Provider: {', '.join(sorted(available_providers))}[/green]")
    else:
        console.print("[red]未检测到任何 API Key。请 export 对应的环境变量。[/red]")
    console.print()
    console.print("[dim]使用 --thinker-model 和 --judge-model 指定模型，例如:[/dim]")
    console.print("[dim]  evohive evolve \"你的问题\" --thinker-model deepseek/deepseek-chat[/dim]")


@app.command()
def history(
    output_dir: str = typer.Option("evohive_results", "--output-dir", help="结果存储目录"),
    limit: int = typer.Option(20, "--limit", "-n", help="显示最近N条记录"),
):
    """查看历史进化运行记录"""
    from evohive.engine.persistence import list_previous_runs

    runs = list_previous_runs(output_dir)
    if not runs:
        console.print("[dim]暂无历史记录[/dim]")
        return

    table = Table(title="历史运行记录")
    table.add_column("ID", style="cyan")
    table.add_column("时间", style="green")
    table.add_column("模式", style="yellow")
    table.add_column("问题", style="white")

    for r in runs[:limit]:
        table.add_row(
            r["id"],
            r["timestamp"][:19] if r["timestamp"] else "N/A",
            r["mode"],
            r["problem"],
        )

    console.print(table)
    console.print(f"\n[dim]共 {len(runs)} 条记录, 存储于: {output_dir}/[/dim]")


@app.command()
def judge_test(
    problem: str = typer.Option(
        "给一个面向独立开发者的AI代码审查工具设计定价策略",
        "--problem", "-p", help="测试问题",
    ),
    model: str = typer.Option("deepseek/deepseek-chat", "--model", "-m", help="要测试的Judge模型"),
    judges: str = typer.Option(
        "可行性:0.3,创新性:0.25,具体性:0.25,成本效率:0.2",
        "--judges", "-j",
        help="评审维度",
    ),
    solutions: int = typer.Option(5, "--solutions", "-n", help="测试方案数量"),
    rounds: int = typer.Option(5, "--rounds", "-r", help="评审轮次"),
):
    """M1.5预实验: 测试Judge模型的评分可靠性"""
    from evohive.engine.judge import judge_reliability_test

    dimensions = _parse_judges(judges)

    console.print()
    console.print(Panel(
        f"[bold]测试模型:[/bold] {model}\n"
        f"[bold]测试方案数:[/bold] {solutions}\n"
        f"[bold]评审轮次:[/bold] {rounds}\n"
        f"[bold]评审维度:[/bold] {', '.join(d['name'] for d in dimensions)}",
        title="[bold yellow]Judge可靠性测试 (M1.5)[/bold yellow]",
        border_style="yellow",
    ))
    console.print()
    console.print("[dim]正在生成测试方案并进行多轮评审...[/dim]")
    console.print()

    def _print_progress(msg: str):
        console.print(f"  [dim]{msg}[/dim]")

    result = asyncio.run(judge_reliability_test(
        problem=problem,
        model=model,
        dimensions=dimensions,
        n_solutions=solutions,
        n_rounds=rounds,
        on_progress=_print_progress,
    ))

    verdict = result["verdict"]
    score = result["consistency_score"]
    if verdict == "reliable":
        verdict_display = "[bold green]可靠 (RELIABLE)[/bold green]"
        border = "green"
    elif verdict == "marginal":
        verdict_display = "[bold yellow]勉强可用 (MARGINAL)[/bold yellow]"
        border = "yellow"
    else:
        verdict_display = "[bold red]不可靠 (UNRELIABLE)[/bold red]"
        border = "red"

    console.print(Panel(
        f"[bold]综合一致性:[/bold] {score:.3f}\n"
        f"[bold]评分方差:[/bold] {result['score_variance']:.4f}\n"
        f"[bold]排名一致性:[/bold] {result['rank_agreement']:.3f}\n"
        f"\n"
        f"[bold]判定:[/bold] {verdict_display}\n"
        f"\n"
        f"[dim]一致性 > 0.7 = 可靠 | 0.5-0.7 = 勉强可用 | < 0.5 = 不可靠[/dim]\n"
        f"[dim]如果不可靠，建议换用更强的Judge模型（如claude或gpt-4o）[/dim]",
        title="[bold]测试结果[/bold]",
        border_style=border,
    ))


if __name__ == "__main__":
    app()
