"""Deterministic claim verification loop scaffolding.

The loop is intentionally cheap: it does not call another model yet. It turns
claims into verification tasks, attaches available evidence, and decides
whether a stronger verifier should spend tokens on the claim later.
"""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{2,}")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "") if len(token) >= 2}


def _context_snippets(search_context: str, limit: int = 12) -> list[str]:
    chunks = []
    for raw in re.split(r"\n{2,}|\n[-*]\s+|\n\d+[.)]\s+", search_context or ""):
        text = " ".join(raw.split())
        if len(text) < 24:
            continue
        chunks.append(text[:500])
        if len(chunks) >= limit:
            break
    return chunks


def _best_evidence(claim_text: str, snippets: list[str]) -> tuple[list[dict], float]:
    claim_tokens = _tokens(claim_text)
    if not claim_tokens:
        return [], 0.0
    scored = []
    for snippet in snippets:
        snippet_tokens = _tokens(snippet)
        overlap = claim_tokens & snippet_tokens
        if not overlap:
            continue
        score = len(overlap) / max(len(claim_tokens), 1)
        scored.append((score, snippet, sorted(overlap)[:12]))
    scored.sort(key=lambda item: item[0], reverse=True)
    evidence = [
        {
            "type": "search_context",
            "support_score": round(score, 3),
            "matched_terms": terms,
            "snippet": snippet,
        }
        for score, snippet, terms in scored[:3]
    ]
    best_score = scored[0][0] if scored else 0.0
    return evidence, round(best_score, 3)


def _verification_status(claim: dict, support_score: float) -> str:
    flags = set(claim.get("risk_flags") or [])
    if support_score >= 0.45 and not flags:
        return "supported"
    if support_score >= 0.35:
        return "partially_supported"
    if flags:
        return "needs_verifier"
    return "weakly_supported"


def _next_action(claim: dict, status: str) -> str:
    flags = set(claim.get("risk_flags") or [])
    if "numeric_claim" in flags:
        return "run_numeric_sanity_check"
    if "absolute_language" in flags:
        return "ask_counterargument_judge"
    if claim.get("kind") == "implementation":
        return "run_execution_or_unit_test"
    if status in {"needs_verifier", "weakly_supported"}:
        return "run_external_source_check"
    return "no_extra_verifier_needed"


def _query_for_claim(claim: dict) -> str:
    text = " ".join((claim.get("text") or "").split())
    return text[:180]


def build_claim_verification_report(
    *,
    verification_report: dict,
    search_context: str = "",
    max_claims: int = 12,
) -> dict:
    """Verify extracted claims against already available evidence."""
    claims = (verification_report or {}).get("claims", [])[:max_claims]
    snippets = _context_snippets(search_context)
    verified = []
    for claim in claims:
        evidence, support_score = _best_evidence(claim.get("text", ""), snippets)
        status = _verification_status(claim, support_score)
        verified.append({
            "id": claim.get("id"),
            "text": claim.get("text", ""),
            "kind": claim.get("kind", "factual"),
            "status": status,
            "support_score": support_score,
            "confidence": claim.get("confidence", 0.0),
            "risk_flags": claim.get("risk_flags", []),
            "evidence": evidence,
            "next_action": _next_action(claim, status),
        })

    counts: dict[str, int] = {}
    for claim in verified:
        counts[claim["status"]] = counts.get(claim["status"], 0) + 1

    return {
        "version": "claim-verification-loop.v1",
        "claim_count": len(verified),
        "evidence_source": "search_context" if snippets else "none",
        "summary": {
            "status_counts": counts,
            "needs_external_verifier": sum(
                1 for claim in verified
                if claim["next_action"] != "no_extra_verifier_needed"
            ),
            "average_support_score": round(
                sum(claim["support_score"] for claim in verified) / len(verified),
                3,
            ) if verified else 0.0,
        },
        "claims": verified,
    }


async def build_claim_search_verification_report(
    *,
    verification_report: dict,
    base_report: dict,
    max_claims: int = 5,
    max_results: int = 3,
) -> dict:
    """Actively search for evidence for claims that need stronger verification."""
    from evohive.engine.web_search import web_search

    claims_by_id = {
        claim.get("id"): claim
        for claim in (verification_report or {}).get("claims", [])
    }
    candidates = [
        claim for claim in (base_report or {}).get("claims", [])
        if claim.get("next_action") != "no_extra_verifier_needed"
    ][:max_claims]

    searched = []
    for claim in candidates:
        original_claim = claims_by_id.get(claim.get("id"), claim)
        query = _query_for_claim(original_claim)
        results = await web_search(query, max_results=max_results)
        snippets = [
            " ".join(filter(None, [result.get("title", ""), result.get("snippet", "")]))
            for result in results
        ]
        evidence, support_score = _best_evidence(original_claim.get("text", ""), snippets)
        for evidence_item, result in zip(evidence, results):
            evidence_item["type"] = "active_search"
            evidence_item["url"] = result.get("url", "")
            evidence_item["title"] = result.get("title", "")
        merged_evidence = list(claim.get("evidence") or []) + evidence
        merged_support = max(float(claim.get("support_score") or 0.0), support_score)
        updated = dict(claim)
        updated.update({
            "active_search_query": query,
            "active_search_results": len(results),
            "evidence": merged_evidence,
            "support_score": round(merged_support, 3),
            "status": _verification_status(original_claim, merged_support),
        })
        updated["next_action"] = _next_action(original_claim, updated["status"])
        searched.append(updated)

    return {
        "version": "claim-search-verification.v1",
        "searched_claim_count": len(searched),
        "claims": searched,
        "summary": {
            "supported_after_search": sum(
                1 for claim in searched
                if claim.get("status") in {"supported", "partially_supported"}
            ),
            "still_needs_verifier": sum(
                1 for claim in searched
                if claim.get("next_action") != "no_extra_verifier_needed"
            ),
        },
    }
