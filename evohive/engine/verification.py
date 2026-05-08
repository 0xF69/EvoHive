"""Build an auditable verification report for an evolved answer.

This module is intentionally deterministic. It gives EvoHive a stable
"claim graph" layer before we plug in heavier LLM or web-based verifiers.
"""

from __future__ import annotations

from datetime import UTC, datetime
import re


_CLAIM_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)]|[一二三四五六七八九十]+[、.])\s*")


def _clean_claim(text: str) -> str:
    text = _LIST_PREFIX_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def _split_claims(answer: str, max_claims: int) -> list[str]:
    chunks = []
    for raw in _CLAIM_SPLIT_RE.split(answer or ""):
        claim = _clean_claim(raw)
        if len(claim) < 12:
            continue
        chunks.append(claim[:500])
        if len(chunks) >= max_claims:
            break
    return chunks


def _classify_claim(claim: str) -> str:
    lower = claim.lower()
    if re.search(r"\d|%|倍|天|周|月|年|美元|成本|tokens?", claim, re.I):
        return "quantitative"
    if any(token in lower for token in ["because", "therefore", "leads to", "drives", "causes"]):
        return "causal"
    if any(token in claim for token in ["因为", "所以", "导致", "带来", "驱动"]):
        return "causal"
    if any(token in lower for token in ["should", "recommend", "need to", "use ", "build "]):
        return "recommendation"
    if any(token in claim for token in ["建议", "应该", "需要", "采用", "构建", "优化"]):
        return "recommendation"
    if any(token in lower for token in ["api", "code", "test", "endpoint", "database", "model"]):
        return "implementation"
    if any(token in claim for token in ["接口", "代码", "测试", "数据库", "模型", "后端"]):
        return "implementation"
    return "factual"


def _risk_flags(claim: str) -> list[str]:
    lower = claim.lower()
    flags: list[str] = []
    if any(token in lower for token in ["always", "never", "guarantee", "must", "100%"]):
        flags.append("absolute_language")
    if any(token in claim for token in ["一定", "永远", "绝对", "保证", "必须", "无法"]):
        flags.append("absolute_language")
    if re.search(r"\d+(?:\.\d+)?\s*(?:%|倍|x|美元|天|周|月|年)", claim, re.I):
        flags.append("numeric_claim")
    if len(claim) > 280:
        flags.append("compound_claim")
    return flags


def _claim_confidence(claim: str, has_search_context: bool, lineage_summary: dict) -> float:
    confidence = 0.52
    if has_search_context:
        confidence += 0.12
    if lineage_summary.get("finalist_count", 0) > 0:
        confidence += 0.08
    if lineage_summary.get("node_count", 0) >= 3:
        confidence += 0.05
    flags = _risk_flags(claim)
    if "absolute_language" in flags:
        confidence -= 0.15
    if "numeric_claim" in flags and not has_search_context:
        confidence -= 0.10
    if "compound_claim" in flags:
        confidence -= 0.05
    return max(0.05, min(0.95, round(confidence, 2)))


def build_verification_report(
    *,
    problem: str,
    final_answer: str,
    lineage_graph: dict | None = None,
    search_context: str = "",
    max_claims: int = 12,
) -> dict:
    """Create a claim-level verification scaffold for the final answer."""
    lineage_graph = lineage_graph or {}
    lineage_summary = lineage_graph.get("summary", {})
    has_search_context = bool((search_context or "").strip())
    claims = []

    for index, claim in enumerate(_split_claims(final_answer, max_claims), start=1):
        flags = _risk_flags(claim)
        confidence = _claim_confidence(claim, has_search_context, lineage_summary)
        claims.append({
            "id": f"claim-{index:02d}",
            "text": claim,
            "kind": _classify_claim(claim),
            "status": "needs_external_verification" if flags else "internally_supported",
            "confidence": confidence,
            "risk_flags": flags,
            "evidence": {
                "lineage": {
                    "node_count": lineage_summary.get("node_count", 0),
                    "finalist_count": lineage_summary.get("finalist_count", 0),
                },
                "search_context": has_search_context,
            },
        })

    high_risk = [claim for claim in claims if claim["risk_flags"]]
    avg_confidence = (
        round(sum(claim["confidence"] for claim in claims) / len(claims), 2)
        if claims else 0.0
    )

    return {
        "version": "verification-report.v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "problem": problem,
        "answer_chars": len(final_answer or ""),
        "summary": {
            "claim_count": len(claims),
            "high_risk_claim_count": len(high_risk),
            "average_confidence": avg_confidence,
            "has_search_context": has_search_context,
            "lineage_node_count": lineage_summary.get("node_count", 0),
        },
        "claims": claims,
        "next_verifiers": [
            "external_source_check",
            "numeric_sanity_check",
            "counterargument_judge",
            "execution_check_if_code",
        ],
    }
