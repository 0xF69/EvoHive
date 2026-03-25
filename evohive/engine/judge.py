"""Judge评审系统"""

from evohive.models import Solution, DimensionScore, Judgment, StableJudgment
from evohive.llm import call_llm, call_llm_batch, extract_json
from evohive.prompts.judge_evaluate import JUDGE_SYSTEM, JUDGE_USER
from statistics import median


# ─────────────── 趋同检测 ───────────────

def compute_population_similarity(solutions: list[Solution]) -> float:
    """计算种群内方案的平均Jaccard相似度 (0-1)。

    返回值越高说明越趋同:
    - < 0.3: 健康多样性
    - 0.3-0.5: 多样性开始下降
    - > 0.5: 严重趋同，进化可能已失去意义
    - > 0.7: 种群坍缩，应中止
    """
    if len(solutions) < 2:
        return 0.0

    keyword_sets = []
    for sol in solutions:
        # 提取内容关键词集合（去除短词和常见停用词）
        words = {w for w in sol.content.split() if len(w) > 1}
        keyword_sets.append(words)

    total_sim = 0.0
    pair_count = 0
    for i in range(len(keyword_sets)):
        for j in range(i + 1, len(keyword_sets)):
            if not keyword_sets[i] or not keyword_sets[j]:
                continue
            intersection = len(keyword_sets[i] & keyword_sets[j])
            union = len(keyword_sets[i] | keyword_sets[j])
            if union > 0:
                total_sim += intersection / union
                pair_count += 1

    return total_sim / pair_count if pair_count > 0 else 0.0


async def compute_population_similarity_async(
    solutions: list[Solution],
    embedding_model: str = "text-embedding-3-small",
) -> float:
    """Async version using embedding similarity (preferred)."""
    try:
        from evohive.engine.embedding import compute_population_similarity_embedding
        return await compute_population_similarity_embedding(solutions, embedding_model)
    except Exception:
        # Fallback to Jaccard
        return compute_population_similarity(solutions)


async def judge_reliability_test(
    problem: str,
    model: str,
    dimensions: list[dict],
    n_solutions: int = 5,
    n_rounds: int = 5,
    on_progress: callable = None,
) -> dict:
    """Judge可靠性预实验 (M1.5)。

    对同一批方案进行多轮评审，检测评分一致性。
    返回:
        consistency_score: 评分一致性 (0-1), >0.7可用, <0.5不可靠
        score_variance: 各轮评分的平均方差
        rank_agreement: 不同轮次排名一致性 (Kendall-like)
        verdict: "reliable" | "marginal" | "unreliable"
    """
    from evohive.engine.genesis import generate_thinkers, generate_initial_solutions

    def _log(msg):
        if on_progress:
            on_progress(msg)

    # 1. 生成少量测试方案
    _log(f"[1/3] 生成 {n_solutions} 个Thinker人格...")
    thinkers = await generate_thinkers(problem, n_solutions, model)
    _log(f"  ✓ {len(thinkers)} 个Thinker就绪")
    for i, t in enumerate(thinkers):
        _log(f"    Thinker {i+1}: {t.persona[:40]}...")

    _log(f"[1/3] 各Thinker独立生成方案...")
    solutions = await generate_initial_solutions(problem, thinkers)
    if not solutions:
        return {"consistency_score": 0, "verdict": "unreliable", "error": "无法生成测试方案"}
    _log(f"  ✓ {len(solutions)} 个方案就绪")
    for i, s in enumerate(solutions):
        preview = s.content[:60].replace("\n", " ")
        _log(f"    方案{i+1}: {preview}...")

    # 2. 对每个方案进行n_rounds轮独立评审
    _log(f"\n[2/3] 开始 {n_rounds} 轮独立评审 ({len(solutions)}方案 × {n_rounds}轮 = {len(solutions)*n_rounds}次API调用)...")
    dims_text = _build_dimensions_text(dimensions)
    all_round_scores = []  # [round][solution_idx] = raw_fitness

    for round_idx in range(n_rounds):
        _log(f"\n  ── 第 {round_idx+1}/{n_rounds} 轮评审 ──")
        round_scores = []
        for sol_idx, sol in enumerate(solutions):
            user_prompt = JUDGE_USER.format(
                dimensions_text=dims_text,
                solution_content=sol.content,
            )
            resp = await call_llm(
                model=model,
                system_prompt=JUDGE_SYSTEM,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1000,
                json_mode=True,
            )
            j = _parse_judgment(resp, sol.id, dimensions)
            score = j.raw_fitness if j else 0.5
            round_scores.append(score)
            dim_detail = ""
            if j:
                dim_detail = " | ".join(f"{sc.name}={sc.score}" for sc in j.scores)
            _log(f"    方案{sol_idx+1} → fitness={score:.3f}  [{dim_detail}]")
        all_round_scores.append(round_scores)
        _log(f"  第{round_idx+1}轮排名: {' > '.join(f'方案{i+1}({s:.3f})' for i, s in sorted(enumerate(round_scores), key=lambda x: -x[1]))}")

    # 3. 计算一致性指标
    _log(f"\n[3/3] 计算一致性指标...")
    n = len(solutions)

    # 平均方差: 每个方案在各轮的评分方差，取平均
    variances = []
    for sol_idx in range(n):
        scores = [all_round_scores[r][sol_idx] for r in range(n_rounds)]
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        variances.append(var)
    avg_variance = sum(variances) / len(variances)
    _log(f"  各方案评分方差: {', '.join(f'方案{i+1}={v:.4f}' for i, v in enumerate(variances))}")
    _log(f"  平均方差: {avg_variance:.4f}")

    # 排名一致性: 比较每两轮的排名是否一致 (pairwise agreement)
    rank_agreements = []
    for r1 in range(n_rounds):
        for r2 in range(r1 + 1, n_rounds):
            concordant = 0
            total_pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    diff1 = all_round_scores[r1][i] - all_round_scores[r1][j]
                    diff2 = all_round_scores[r2][i] - all_round_scores[r2][j]
                    if diff1 * diff2 > 0:  # 同向
                        concordant += 1
                    total_pairs += 1
            if total_pairs > 0:
                rank_agreements.append(concordant / total_pairs)

    avg_rank_agreement = sum(rank_agreements) / len(rank_agreements) if rank_agreements else 0.0
    _log(f"  排名一致性: {avg_rank_agreement:.3f}")

    # 综合一致性分数
    consistency_score = avg_rank_agreement * 0.6 + max(0, 1 - avg_variance * 20) * 0.4

    # 判定
    if consistency_score >= 0.7:
        verdict = "reliable"
    elif consistency_score >= 0.5:
        verdict = "marginal"
    else:
        verdict = "unreliable"

    return {
        "consistency_score": round(consistency_score, 3),
        "score_variance": round(avg_variance, 4),
        "rank_agreement": round(avg_rank_agreement, 3),
        "n_solutions": n,
        "n_rounds": n_rounds,
        "model": model,
        "verdict": verdict,
    }


def _build_dimensions_text(dimensions: list[dict]) -> str:
    """构建评审维度的文本描述"""
    lines = []
    for d in dimensions:
        desc = d.get("description", d["name"])
        lines.append(f"- {d['name']}（权重{d['weight']}）: {desc}")
    return "\n".join(lines)


def _parse_judgment(response: str, solution_id: str, dimensions: list[dict]) -> Judgment | None:
    """解析Judge的JSON回复"""
    data = extract_json(response)
    if not data:
        return None

    scores_data = data.get("scores", [])
    if not scores_data:
        return None

    # 构建维度权重查找表
    weight_map = {d["name"]: d["weight"] for d in dimensions}

    scores = []
    for s in scores_data:
        name = s.get("name", "")
        score = int(s.get("score", 5))
        score = max(1, min(10, score))  # 钳制到1-10
        reason = s.get("reason", "无理由")
        weight = weight_map.get(name, 1.0 / len(dimensions))
        scores.append(DimensionScore(
            name=name, score=score, reason=reason, weight=weight
        ))

    # 计算raw_fitness
    total_weight = sum(sc.weight for sc in scores)
    if total_weight == 0:
        total_weight = 1.0
    raw_fitness = sum(sc.score * sc.weight for sc in scores) / total_weight / 10.0

    return Judgment(
        solution_id=solution_id,
        scores=scores,
        raw_fitness=raw_fitness,
    )


async def evaluate_solution(
    solution: Solution,
    dimensions: list[dict],
    model: str,
    rounds: int = 1,
) -> StableJudgment:
    """对单个Solution进行（可去噪的）评审"""
    dims_text = _build_dimensions_text(dimensions)
    user_prompt = JUDGE_USER.format(
        dimensions_text=dims_text,
        solution_content=solution.content,
    )

    # 多轮独立评审
    calls = [
        {
            "model": model,
            "system_prompt": JUDGE_SYSTEM,
            "user_prompt": user_prompt,
            "temperature": 0.3,
            "max_tokens": 1000,
            "json_mode": True,
        }
        for _ in range(rounds)
    ]

    if rounds == 1:
        responses = [await call_llm(**calls[0])]
    else:
        responses = await call_llm_batch(calls, max_concurrent=5)

    # 解析所有评审结果
    judgments = []
    for resp in responses:
        if isinstance(resp, str) and not resp.startswith("ERROR:"):
            j = _parse_judgment(resp, solution.id, dimensions)
            if j:
                judgments.append(j)

    if not judgments:
        # 所有评审都失败了，给一个默认分
        default_scores = [
            DimensionScore(name=d["name"], score=5, reason="评审失败，使用默认分", weight=d["weight"])
            for d in dimensions
        ]
        return StableJudgment(
            solution_id=solution.id,
            median_scores=default_scores,
            raw_fitness=0.5,
            final_fitness=0.5,
        )

    # 取中位数（如果只有1轮就直接用）
    if len(judgments) == 1:
        j = judgments[0]
        return StableJudgment(
            solution_id=solution.id,
            median_scores=j.scores,
            raw_fitness=j.raw_fitness,
            final_fitness=j.raw_fitness,
        )

    # 多轮去噪: 每个维度取中位数
    dim_names = [d["name"] for d in dimensions]
    median_scores = []
    for name in dim_names:
        all_scores_for_dim = []
        all_reasons_for_dim = []
        weight = next((d["weight"] for d in dimensions if d["name"] == name), 0.25)

        for j in judgments:
            for sc in j.scores:
                if sc.name == name:
                    all_scores_for_dim.append(sc.score)
                    all_reasons_for_dim.append(sc.reason)
                    break

        if all_scores_for_dim:
            med_score = int(median(all_scores_for_dim))
            reason = all_reasons_for_dim[len(all_reasons_for_dim) // 2]
        else:
            med_score = 5
            reason = "无评分数据"

        median_scores.append(DimensionScore(
            name=name, score=med_score, reason=reason, weight=weight
        ))

    # 计算raw_fitness
    total_weight = sum(sc.weight for sc in median_scores)
    if total_weight == 0:
        total_weight = 1.0
    raw_fitness = sum(sc.score * sc.weight for sc in median_scores) / total_weight / 10.0

    return StableJudgment(
        solution_id=solution.id,
        median_scores=median_scores,
        raw_fitness=raw_fitness,
        final_fitness=raw_fitness,
    )


async def evaluate_population(
    solutions: list[Solution],
    dimensions: list[dict],
    model: str,
    rounds: int = 1,
    diversity_weight: float = 0.15,
) -> list[StableJudgment]:
    """评估整个种群"""
    import asyncio

    # 并发评审所有Solution
    sem = asyncio.Semaphore(25)

    async def eval_one(sol):
        async with sem:
            return await evaluate_solution(sol, dimensions, model, rounds)

    judgments = await asyncio.gather(*[eval_one(s) for s in solutions])

    # 计算多样性加分 (embedding-based)
    if diversity_weight > 0:
        try:
            from evohive.engine.embedding import compute_diversity_scores
            diversity_scores = await compute_diversity_scores(solutions)
            for j_result, div_score in zip(judgments, diversity_scores):
                j_result.diversity_bonus = div_score
        except Exception:
            # Fallback to Jaccard-based diversity
            keyword_sets = []
            for sol in solutions:
                words = set(sol.content.split())
                keyword_sets.append(words)

            for i, (j, sol) in enumerate(zip(judgments, solutions)):
                if not keyword_sets[i]:
                    j.diversity_bonus = 0.0
                    continue

                distances = []
                for k, other_set in enumerate(keyword_sets):
                    if k == i or not other_set:
                        continue
                    intersection = len(keyword_sets[i] & other_set)
                    union = len(keyword_sets[i] | other_set)
                    jaccard = intersection / union if union > 0 else 0
                    distances.append(1.0 - jaccard)

                j.diversity_bonus = sum(distances) / len(distances) if distances else 0.0

            # 归一化diversity_bonus到0-1
            max_div = max(j.diversity_bonus for j in judgments) if judgments else 1.0
            if max_div > 0:
                for j in judgments:
                    j.diversity_bonus = j.diversity_bonus / max_div

        # 计算final_fitness
        for j in judgments:
            j.final_fitness = (
                j.raw_fitness * (1 - diversity_weight)
                + j.diversity_bonus * diversity_weight
            )
    else:
        for j in judgments:
            j.final_fitness = j.raw_fitness

    # 更新Solution的fitness
    judgment_map = {j.solution_id: j for j in judgments}
    for sol in solutions:
        if sol.id in judgment_map:
            j = judgment_map[sol.id]
            sol.fitness = j.final_fitness
            sol.raw_fitness = j.raw_fitness
            sol.diversity_bonus = j.diversity_bonus

    return list(judgments)
