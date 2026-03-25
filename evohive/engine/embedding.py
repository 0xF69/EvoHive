"""语义相似度模块 — 基于Embedding的语义相似度计算

替代原始的Jaccard词袋相似度，提供更准确的语义距离度量。
支持多种Embedding Provider，失败时自动回退到Jaccard。
"""

import asyncio
import math
import os
from typing import Optional

# Embedding缓存 (content_hash -> vector)
_embedding_cache: dict[str, list[float]] = {}

# 默认Embedding模型优先级
DEFAULT_EMBEDDING_MODELS = [
    "text-embedding-3-small",  # OpenAI
]


def _content_hash(text: str) -> str:
    """Generate a simple hash for caching"""
    import hashlib
    return hashlib.md5(text[:2000].encode()).hexdigest()


async def get_embedding(text: str, model: str = "text-embedding-3-small") -> Optional[list[float]]:
    """Get embedding vector for a single text.

    Tries litellm.aembedding(), returns None on failure.
    """
    cache_key = _content_hash(text)
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    try:
        import litellm
        # Truncate to avoid token limits
        truncated = text[:8000]
        response = await litellm.aembedding(
            model=model,
            input=[truncated],
        )
        vector = response.data[0]["embedding"]
        _embedding_cache[cache_key] = vector
        return vector
    except Exception:
        return None


async def get_embeddings_batch(
    texts: list[str],
    model: str = "text-embedding-3-small",
    max_concurrent: int = 10,
) -> list[Optional[list[float]]]:
    """Get embeddings for multiple texts concurrently."""
    sem = asyncio.Semaphore(max_concurrent)

    async def get_one(text):
        async with sem:
            return await get_embedding(text, model)

    return await asyncio.gather(*[get_one(t) for t in texts])


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Fallback: keyword-level Jaccard similarity."""
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


async def compute_pairwise_similarity(
    solutions: list,  # list[Solution]
    embedding_model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """Compute pairwise similarity matrix for solutions.

    Returns NxN matrix of similarity scores (0-1).
    Uses embedding cosine similarity, falls back to Jaccard on failure.
    """
    n = len(solutions)
    if n == 0:
        return []

    texts = [s.content[:2000] for s in solutions]

    # Try embedding-based similarity
    embeddings = await get_embeddings_batch(texts, embedding_model)

    # Check if we got valid embeddings
    valid_count = sum(1 for e in embeddings if e is not None)
    use_embedding = valid_count >= n * 0.8  # At least 80% success

    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            if use_embedding and embeddings[i] is not None and embeddings[j] is not None:
                sim = cosine_similarity(embeddings[i], embeddings[j])
            else:
                sim = jaccard_similarity(texts[i], texts[j])
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


async def compute_population_similarity_embedding(
    solutions: list,
    embedding_model: str = "text-embedding-3-small",
) -> float:
    """Compute average pairwise similarity of a population.

    Drop-in replacement for judge.compute_population_similarity,
    but uses semantic embeddings.
    """
    if len(solutions) < 2:
        return 0.0

    matrix = await compute_pairwise_similarity(solutions, embedding_model)
    n = len(solutions)

    total_sim = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_sim += matrix[i][j]
            pair_count += 1

    return total_sim / pair_count if pair_count > 0 else 0.0


async def compute_diversity_scores(
    solutions: list,
    embedding_model: str = "text-embedding-3-small",
) -> list[float]:
    """Compute diversity score for each solution (how different it is from others).

    Returns list of scores in [0, 1] where 1 = maximally diverse.
    """
    n = len(solutions)
    if n < 2:
        return [1.0] * n

    matrix = await compute_pairwise_similarity(solutions, embedding_model)

    diversity_scores = []
    for i in range(n):
        # Average distance to all other solutions
        distances = [1.0 - matrix[i][j] for j in range(n) if j != i]
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        diversity_scores.append(avg_distance)

    # Normalize to [0, 1]
    max_div = max(diversity_scores) if diversity_scores else 1.0
    if max_div > 0:
        diversity_scores = [d / max_div for d in diversity_scores]

    return diversity_scores


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache = {}
