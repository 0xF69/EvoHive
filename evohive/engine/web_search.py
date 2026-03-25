"""Web Search 工具 — 为进化方案注入真实数据

支持多种搜索 API，按优先级尝试:
1. Tavily (TAVILY_API_KEY)
2. Serper (SERPER_API_KEY)

如果都没配置，返回空结果（不中断流程）。
"""

import os
import json
import asyncio
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from evohive.engine.logger import get_logger

_logger = get_logger("evohive.engine.web_search")


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for real-time information.

    Returns:
        List of {"title": str, "snippet": str, "url": str}
    """
    # Try Tavily first
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    if tavily_key and HAS_HTTPX:
        results = await _search_tavily(query, tavily_key, max_results)
        if results:
            return results

    # Try Serper
    serper_key = os.environ.get("SERPER_API_KEY", "")
    if serper_key and HAS_HTTPX:
        results = await _search_serper(query, serper_key, max_results)
        if results:
            return results

    # No search API configured
    return []


async def search_context_for_problem(problem: str, max_queries: int = 3) -> str:
    """Generate search context for a problem.

    Searches for background info and returns formatted context string.
    Returns empty string if no search API is configured.
    """
    results = await web_search(problem, max_results=5)
    if not results:
        return ""

    lines = ["以下是相关的真实市场信息和数据（请在生成方案时参考）:"]
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"- {title}: {snippet}")
        if url:
            lines.append(f"  来源: {url}")

    return "\n".join(lines)


async def search_for_chapter(topic: str, problem: str) -> str:
    """Search for data relevant to a specific chapter topic.

    Returns formatted search results for chapter expansion.
    """
    query = f"{problem} {topic}"
    results = await web_search(query, max_results=3)
    if not results:
        return ""

    lines = [f"关于「{topic}」的真实数据参考:"]
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        lines.append(f"- {title}: {snippet}")

    return "\n".join(lines)


async def _search_tavily(query: str, api_key: str, max_results: int) -> list[dict]:
    """Search via Tavily API"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            if response.status_code == 200:
                data = response.json()
                results = []
                for r in data.get("results", []):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("content", "")[:300],
                        "url": r.get("url", ""),
                    })
                return results
    except Exception as e:
        _logger.error("Tavily search failed for query %r: %s", query, e)
    return []
async def _search_serper(query: str, api_key: str, max_results: int) -> list[dict]:
    """Search via Serper API"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": max_results},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            )
            if response.status_code == 200:
                data = response.json()
                results = []
                for r in data.get("organic", []):
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", "")[:300],
                        "url": r.get("link", ""),
                    })
                return results
    except Exception as e:
        _logger.error("Serper search failed for query %r: %s", query, e)
    return []
