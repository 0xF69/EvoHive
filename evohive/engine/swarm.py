"""Swarm 层 — 千级 Agent 轻量策略探索

第一层: 生成 500-1000 个极轻量的 idea seed (50-100 token each)
第二层: Embedding 聚类, 发现 30-50 个策略方向
第三层: 选出代表性 seed, 扩展为完整方案送入进化引擎

成本估算: 1000 seeds ≈ $0.05-0.10 (使用 Flash 级模型)
"""

import asyncio
import random
import math
from typing import Optional, Callable

from evohive.models import Solution, Thinker, ThinkerDNA
from evohive.llm import call_llm, call_llm_batch, extract_json
from evohive.prompts.swarm_prompts import (
    SWARM_PERSONA_SYSTEM, SWARM_PERSONA_USER,
    SWARM_SEED_SYSTEM, SWARM_SEED_USER,
    EXPAND_SEED_SYSTEM, EXPAND_SEED_USER,
)
from evohive.engine.logger import get_logger, log_event

_logger = get_logger("evohive.engine.swarm")


async def generate_swarm_personas(
    problem: str,
    total_count: int,
    models: list[str],
    batch_size: int = 50,
) -> list[str]:
    """批量生成大量极简 persona.

    将总数分成多个 batch, 每个 batch 由一个模型生成 batch_size 个 persona.

    Args:
        problem: 问题描述
        total_count: 需要的 persona 总数
        models: 可用模型列表 (会轮换使用)
        batch_size: 每次 LLM 调用生成的 persona 数量

    Returns:
        persona 字符串列表
    """
    n_batches = math.ceil(total_count / batch_size)

    calls = []
    for i in range(n_batches):
        model = models[i % len(models)]
        count = min(batch_size, total_count - i * batch_size)
        calls.append({
            "model": model,
            "system_prompt": SWARM_PERSONA_SYSTEM,
            "user_prompt": SWARM_PERSONA_USER.format(count=count, problem=problem),
            "temperature": 0.95,
            "max_tokens": max(500, count * 30),
            "json_mode": True,
        })

    responses = await call_llm_batch(calls, max_concurrent=20)

    all_personas = []
    for resp in responses:
        if isinstance(resp, str) and not resp.startswith("ERROR:"):
            data = extract_json(resp)
            if data and isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "persona" in item:
                        all_personas.append(item["persona"])
                    elif isinstance(item, str):
                        all_personas.append(item)

    # Deduplicate and trim
    seen = set()
    unique = []
    for p in all_personas:
        key = p[:20]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique[:total_count]


async def generate_swarm_seeds(
    problem: str,
    personas: list[str],
    models: list[str],
    context: str = "",
    on_progress: Optional[Callable] = None,
) -> list[dict]:
    """为每个 persona 生成一个轻量 idea seed.

    Args:
        problem: 问题描述
        personas: persona 列表
        models: 模型列表 (轮换使用)
        context: 搜索上下文 (可选)
        on_progress: 进度回调

    Returns:
        [{"persona": str, "seed": str, "model": str}, ...]
    """
    calls = []
    meta = []

    for i, persona in enumerate(personas):
        model = models[i % len(models)]
        ctx_text = f"\n参考信息:\n{context}" if context else ""

        calls.append({
            "model": model,
            "system_prompt": SWARM_SEED_SYSTEM.format(persona=persona),
            "user_prompt": SWARM_SEED_USER.format(problem=problem, context=ctx_text),
            "temperature": 0.9,
            "max_tokens": 200,  # Seeds are short
        })
        meta.append({"persona": persona, "model": model})

    # High concurrency for lightweight calls
    responses = await call_llm_batch(calls, max_concurrent=50)

    seeds = []
    for i, resp in enumerate(responses):
        if isinstance(resp, str) and not resp.startswith("ERROR:") and len(resp.strip()) > 10:
            seeds.append({
                "persona": meta[i]["persona"],
                "seed": resp.strip(),
                "model": meta[i]["model"],
            })

    if on_progress:
        on_progress(f"生成了 {len(seeds)}/{len(personas)} 个策略种子")

    return seeds


async def cluster_seeds(
    seeds: list[dict],
    n_clusters: int = 0,
    embedding_model: str = "text-embedding-3-small",
) -> list[list[dict]]:
    """将 seeds 按语义相似度聚类.

    Args:
        seeds: seed 列表
        n_clusters: 聚类数 (0 = auto, sqrt(n))
        embedding_model: Embedding 模型

    Returns:
        聚类结果: [[seed1, seed2, ...], [seed3, seed4, ...], ...]
    """
    if len(seeds) < 3:
        return [seeds]

    # Auto cluster count
    if n_clusters <= 0:
        n_clusters = max(5, min(50, int(math.sqrt(len(seeds)))))

    # Get embeddings
    from evohive.engine.embedding import get_embeddings_batch, cosine_similarity

    texts = [s["seed"] for s in seeds]
    embeddings = await get_embeddings_batch(texts, embedding_model)

    # Filter out seeds with failed embeddings
    valid_seeds = []
    valid_embeddings = []
    for seed, emb in zip(seeds, embeddings):
        if emb is not None:
            valid_seeds.append(seed)
            valid_embeddings.append(emb)

    if not valid_embeddings:
        # Fallback: random clustering
        return _random_cluster(seeds, n_clusters)

    # Simple K-means clustering (no sklearn dependency)
    clusters = _kmeans_cluster(valid_seeds, valid_embeddings, n_clusters)

    # Add failed-embedding seeds to random clusters
    for seed, emb in zip(seeds, embeddings):
        if emb is None:
            idx = random.randint(0, len(clusters) - 1)
            clusters[idx].append(seed)

    # Remove empty clusters
    clusters = [c for c in clusters if c]

    return clusters


def _kmeans_cluster(
    seeds: list[dict],
    embeddings: list[list[float]],
    k: int,
    max_iter: int = 20,
) -> list[list[dict]]:
    """Simple K-means clustering implementation."""
    n = len(embeddings)
    dim = len(embeddings[0])

    if k >= n:
        return [[s] for s in seeds]

    # Initialize centroids randomly
    indices = random.sample(range(n), k)
    centroids = [list(embeddings[i]) for i in indices]

    assignments = [0] * n

    for _ in range(max_iter):
        # Assign each point to nearest centroid
        new_assignments = []
        for emb in embeddings:
            best_k = 0
            best_dist = float('inf')
            for ki in range(k):
                dist = sum((a - b) ** 2 for a, b in zip(emb, centroids[ki]))
                if dist < best_dist:
                    best_dist = dist
                    best_k = ki
            new_assignments.append(best_k)

        if new_assignments == assignments:
            break
        assignments = new_assignments

        # Update centroids
        for ki in range(k):
            members = [embeddings[i] for i in range(n) if assignments[i] == ki]
            if members:
                centroids[ki] = [
                    sum(m[d] for m in members) / len(members)
                    for d in range(dim)
                ]

    # Build cluster lists
    clusters: list[list[dict]] = [[] for _ in range(k)]
    for i, ki in enumerate(assignments):
        clusters[ki].append(seeds[i])

    return [c for c in clusters if c]


def _random_cluster(seeds: list[dict], k: int) -> list[list[dict]]:
    """Fallback: random clustering when embedding fails."""
    random.shuffle(seeds)
    clusters: list[list[dict]] = [[] for _ in range(k)]
    for i, seed in enumerate(seeds):
        clusters[i % k].append(seed)
    return [c for c in clusters if c]


def select_representatives(
    clusters: list[list[dict]],
    max_representatives: int = 50,
) -> list[dict]:
    """从每个 cluster 选出代表性 seed.

    每个 cluster 选 1 个最居中的 seed (或最长的作为代理).

    Args:
        clusters: 聚类结果
        max_representatives: 最多选出的代表数

    Returns:
        代表性 seed 列表
    """
    representatives = []

    # Distribute quota proportionally
    total_seeds = sum(len(c) for c in clusters)

    for cluster in clusters:
        if not cluster:
            continue

        # Proportional allocation (at least 1 per cluster)
        quota = max(1, round(len(cluster) / total_seeds * max_representatives))

        # Select: prefer longer seeds (more detailed = more representative)
        sorted_seeds = sorted(cluster, key=lambda s: len(s["seed"]), reverse=True)
        representatives.extend(sorted_seeds[:quota])

    return representatives[:max_representatives]


async def expand_seeds_to_solutions(
    seeds: list[dict],
    problem: str,
    models: list[str],
    generation: int = 0,
    search_context: str = "",
) -> list[Solution]:
    """将选中的 seeds 扩展为完整 Solution.

    Args:
        seeds: 代表性 seed 列表
        problem: 问题描述
        models: 可用模型列表
        generation: 所属代数
        search_context: 搜索上下文

    Returns:
        Solution 列表
    """
    calls = []
    for i, seed in enumerate(seeds):
        model = models[i % len(models)]
        ctx = f"\n{search_context}" if search_context else ""
        calls.append({
            "model": model,
            "system_prompt": EXPAND_SEED_SYSTEM,
            "user_prompt": EXPAND_SEED_USER.format(
                problem=problem,
                seed_content=seed["seed"],
                search_context=ctx,
            ),
            "temperature": 0.8,
            "max_tokens": 4000,
        })

    responses = await call_llm_batch(calls, max_concurrent=20)

    solutions = []
    for i, (seed, resp) in enumerate(zip(seeds, responses)):
        if isinstance(resp, str) and not resp.startswith("ERROR:"):
            content = resp
        else:
            content = seed["seed"]  # Fallback to raw seed

        sol = Solution(
            content=content,
            generation=generation,
            thinker_id=f"swarm_{i:04d}",
        )
        solutions.append(sol)

    return solutions


async def run_swarm_phase(
    problem: str,
    models: list[str],
    total_seeds: int = 500,
    max_representatives: int = 50,
    embedding_model: str = "text-embedding-3-small",
    search_context: str = "",
    on_progress: Optional[Callable] = None,
) -> tuple[list[Solution], dict]:
    """运行完整的 Swarm 阶段.

    Args:
        problem: 问题描述
        models: 可用模型列表 (Flash 级别推荐)
        total_seeds: 生成的种子总数
        max_representatives: 最终选出的代表数
        embedding_model: Embedding 模型
        search_context: 问题搜索上下文
        on_progress: 进度回调

    Returns:
        (solutions, stats) - 扩展后的 Solution 列表和统计信息
    """
    def _log(msg):
        if on_progress:
            on_progress(msg)

    # Step 1: Generate personas
    _log(f"Swarm: 生成 {total_seeds} 个多样化角色...")
    log_event(_logger, "swarm_phase_start",
              total_seeds=total_seeds,
              max_representatives=max_representatives,
              models=models,
              embedding_model=embedding_model)
    personas = await generate_swarm_personas(problem, total_seeds, models)
    _log(f"Swarm: 获得 {len(personas)} 个角色")

    # Step 2: Generate seeds
    _log(f"Swarm: {len(personas)} 个角色并发生成策略种子...")
    seeds = await generate_swarm_seeds(problem, personas, models, search_context, on_progress)
    _log(f"Swarm: 获得 {len(seeds)} 个策略种子")

    # Step 3: Cluster
    _log(f"Swarm: 语义聚类中...")
    clusters = await cluster_seeds(seeds, embedding_model=embedding_model)
    _log(f"Swarm: 发现 {len(clusters)} 个策略方向")

    # Step 4: Select representatives
    representatives = select_representatives(clusters, max_representatives)
    _log(f"Swarm: 选出 {len(representatives)} 个代表性种子")
    log_event(_logger, "swarm_clustering",
              n_clusters=len(clusters),
              cluster_sizes=[len(c) for c in clusters],
              n_representatives=len(representatives))

    # Step 5: Expand to full solutions
    _log(f"Swarm: 扩展为完整方案...")
    solutions = await expand_seeds_to_solutions(
        representatives, problem, models,
        generation=0, search_context=search_context,
    )
    _log(f"Swarm: 生成 {len(solutions)} 个完整方案")

    log_event(_logger, "swarm_phase_end",
              total_personas=len(personas),
              total_seeds=len(seeds),
              n_clusters=len(clusters),
              n_representatives=len(representatives),
              n_solutions=len(solutions))

    stats = {
        "total_personas": len(personas),
        "total_seeds": len(seeds),
        "n_clusters": len(clusters),
        "cluster_sizes": [len(c) for c in clusters],
        "n_representatives": len(representatives),
        "n_solutions": len(solutions),
    }

    return solutions, stats
