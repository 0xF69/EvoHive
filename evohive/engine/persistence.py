"""结果持久化与导出模块 — 保存/加载进化运行结果，生成Markdown报告

Persistence & export for evolution run results.
Supports JSON serialization, Markdown report generation, and run history listing.
"""

import json
import os
from datetime import datetime
from typing import Optional

from evohive.models import EvolutionRun, EvolutionConfig, GenerationStats


# ── JSON datetime 序列化 ──

def _json_serializer(obj):
    """自定义JSON序列化：处理datetime等不可序列化类型"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── 保存与加载 ──

def save_run_result(run: EvolutionRun, output_dir: str = "evohive_results") -> str:
    """保存一次进化运行的完整结果（JSON + Markdown）。

    Save full evolution run to JSON and a human-readable Markdown summary.

    Args:
        run: 完整的进化运行数据
        output_dir: 输出目录，默认 evohive_results

    Returns:
        JSON 文件的路径
    """
    # 创建输出目录 / Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存JSON / Save full run as JSON
    json_path = os.path.join(output_dir, f"{run.id}.json")
    run_data = run.model_dump()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, ensure_ascii=False, indent=2, default=_json_serializer)

    # 2. 保存Markdown报告 / Save Markdown report
    md_path = os.path.join(output_dir, f"{run.id}.md")
    md_content = format_markdown_report(run)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path


def load_run_result(path: str) -> EvolutionRun:
    """从JSON文件加载进化运行结果。

    Load an EvolutionRun from a previously saved JSON file.

    Args:
        path: JSON文件路径

    Returns:
        反序列化后的 EvolutionRun 对象
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return EvolutionRun(**data)


# ── Markdown 报告生成 ──

def format_markdown_report(run: EvolutionRun) -> str:
    """生成完整的 Markdown 格式进化报告。

    Generate a comprehensive Markdown report for the evolution run.

    Args:
        run: 完整的进化运行数据

    Returns:
        Markdown格式的报告字符串
    """
    lines: list[str] = []

    # ── 头部信息 / Header ──
    duration_str = _format_duration(run.started_at, run.finished_at)
    lines.append(f"# EvoHive Evolution Report")
    lines.append("")
    lines.append(f"- **Run ID**: `{run.id}`")
    lines.append(f"- **Started**: {run.started_at.isoformat()}")
    lines.append(f"- **Finished**: {run.finished_at.isoformat() if run.finished_at else 'N/A'}")
    lines.append(f"- **Mode**: {run.mode}")
    lines.append(f"- **Duration**: {duration_str}")
    if run.early_stop_reason:
        lines.append(f"- **Early Stop**: {run.early_stop_reason}")
    lines.append("")

    # ── 问题描述 / Problem Statement ──
    lines.append("## Problem Statement")
    lines.append("")
    lines.append(run.config.problem)
    lines.append("")

    # ── 配置概要 / Configuration Summary ──
    cfg = run.config
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Population Size | {cfg.population_size} |")
    lines.append(f"| Generations | {cfg.generations} |")
    lines.append(f"| Mode | {cfg.mode} |")
    lines.append(f"| Survival Rate | {cfg.survival_rate} |")
    lines.append(f"| Mutation Rate | {cfg.mutation_rate} |")
    lines.append(f"| Thinker Model | `{cfg.thinker_model}` |")
    lines.append(f"| Judge Model | `{cfg.judge_model}` |")
    if cfg.judge_models:
        lines.append(f"| Judge Models | {', '.join(f'`{m}`' for m in cfg.judge_models)} |")
    if cfg.thinker_models:
        lines.append(f"| Thinker Models | {', '.join(f'`{m}`' for m in cfg.thinker_models)} |")
    if cfg.red_team_models:
        lines.append(f"| Red Team Models | {', '.join(f'`{m}`' for m in cfg.red_team_models)} |")
    if cfg.swarm_models:
        lines.append(f"| Swarm Models | {', '.join(f'`{m}`' for m in cfg.swarm_models)} |")
    lines.append("")

    # ── Swarm 统计 / Swarm Stats ──
    if run.swarm_stats:
        lines.append("## Swarm Stats")
        lines.append("")
        for key, value in run.swarm_stats.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    # ── 进化统计表 / Evolution Statistics ──
    if run.generations_data:
        lines.append("## Evolution Statistics")
        lines.append("")
        lines.append("| Generation | Best Fitness | Avg Fitness | Worst Fitness | Alive | Eliminated |")
        lines.append("|------------|-------------|-------------|---------------|-------|------------|")
        for g in run.generations_data:
            lines.append(
                f"| {g.generation} | {g.best_fitness:.4f} | {g.avg_fitness:.4f} "
                f"| {g.worst_fitness:.4f} | {g.alive_count} | {g.eliminated_count} |"
            )
        lines.append("")

    # ── Baseline vs Top 1 对比 / Baseline vs Top 1 Comparison ──
    lines.append("## Baseline vs Top 1")
    lines.append("")
    if run.baseline_solution:
        lines.append("### Baseline Solution")
        lines.append("")
        lines.append(run.baseline_solution)
        lines.append("")
    else:
        lines.append("*No baseline solution recorded.*")
        lines.append("")

    if run.final_top_solutions:
        top1 = run.final_top_solutions[0]
        lines.append("### Top 1 Solution")
        lines.append("")
        # 方案内容可能在 content 或 solution 字段
        top1_content = top1.get("content", top1.get("solution", str(top1)))
        top1_fitness = top1.get("fitness", "N/A")
        lines.append(f"**Fitness**: {top1_fitness}")
        lines.append("")
        lines.append(top1_content)
        lines.append("")

    # ── Top 5 方案 / Top 5 Solutions ──
    lines.append("## Top 5 Solutions")
    lines.append("")
    top5 = run.final_top_solutions[:5]
    if top5:
        for i, sol in enumerate(top5, 1):
            fitness = sol.get("fitness", "N/A")
            sol_id = sol.get("id", f"solution-{i}")
            content = sol.get("content", sol.get("solution", str(sol)))
            lines.append(f"### #{i} (Fitness: {fitness}, ID: {sol_id})")
            lines.append("")
            lines.append(content)
            lines.append("")
    else:
        lines.append("*No final solutions recorded.*")
        lines.append("")

    # ── 红队结果 / Red Team Results ──
    if run.red_team_results:
        lines.append("## Red Team Results")
        lines.append("")
        for i, rt in enumerate(run.red_team_results, 1):
            attack = rt.get("attack", rt.get("challenge", "N/A"))
            model = rt.get("model", "N/A")
            survived = rt.get("survived", rt.get("passed", "N/A"))
            lines.append(f"**Attack #{i}** (model: `{model}`, survived: {survived})")
            lines.append("")
            lines.append(f"> {attack}")
            lines.append("")

    # ── 辩论结果 / Debate Results ──
    if run.debate_results:
        lines.append("## Debate Results")
        lines.append("")
        for key, value in run.debate_results.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    # ── 压力测试 / Pressure Test Results ──
    if run.pressure_test_results:
        lines.append("## Pressure Test Results")
        lines.append("")
        for i, pt in enumerate(run.pressure_test_results, 1):
            scenario = pt.get("scenario", pt.get("test", "N/A"))
            passed = pt.get("passed", pt.get("survived", "N/A"))
            lines.append(f"- **Test #{i}**: {scenario} — passed: {passed}")
        lines.append("")

    # ── 淘汰记忆 / Elimination Memories ──
    if run.elimination_memories:
        lines.append("## Elimination Memories")
        lines.append("")
        for mem in run.elimination_memories:
            lines.append(f"- {mem}")
        lines.append("")

    # ── 精炼方案 / Refined Top Solution ──
    if run.refined_top_solution:
        lines.append("## Refined Top Solution")
        lines.append("")
        lines.append(run.refined_top_solution)
        lines.append("")

    # ── 成本摘要 / Cost Summary ──
    lines.append("## Cost Summary")
    lines.append("")
    lines.append(f"- **Estimated Cost**: ${run.estimated_cost:.4f}")
    lines.append(f"- **Total API Calls**: {run.total_api_calls}")
    lines.append(f"- **Duration**: {duration_str}")
    lines.append("")

    return "\n".join(lines)


# ── 历史运行列表 ──

def list_previous_runs(output_dir: str = "evohive_results") -> list[dict]:
    """列出所有已保存的进化运行记录。

    List previously saved runs with basic metadata.

    Args:
        output_dir: 结果存储目录

    Returns:
        包含 id, timestamp, problem, mode 的字典列表
    """
    results: list[dict] = []

    if not os.path.isdir(output_dir):
        return results

    for filename in os.listdir(output_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            run_id = data.get("id", filename.replace(".json", ""))
            timestamp = data.get("started_at", "")
            problem = data.get("config", {}).get("problem", "")
            mode = data.get("mode", "deep")

            results.append({
                "id": run_id,
                "timestamp": timestamp,
                "problem": problem[:50],
                "mode": mode,
            })
        except (json.JSONDecodeError, KeyError, OSError):
            # 跳过无法解析的文件 / Skip files that can't be parsed
            continue

    # 按时间戳倒序排列 / Sort by timestamp descending
    results.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return results


# ── 辅助函数 / Helpers ──

def _format_duration(started_at: datetime, finished_at: Optional[datetime]) -> str:
    """计算并格式化运行时长 / Format run duration as human-readable string."""
    if not finished_at:
        return "N/A"
    delta = finished_at - started_at
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
