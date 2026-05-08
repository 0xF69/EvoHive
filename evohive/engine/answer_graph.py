"""Build a render-ready answer graph from evolution and verification data."""

from __future__ import annotations

import math


def _excerpt(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _node(
    node_id: str,
    node_type: str,
    label: str,
    *,
    text: str = "",
    group: str = "",
    x: float = 0.0,
    y: float = 0.0,
    data: dict | None = None,
) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "label": label,
        "text": text,
        "group": group or node_type,
        "position": {"x": round(x, 2), "y": round(y, 2)},
        "data": data or {},
    }


def _edge(
    source: str,
    target: str,
    edge_type: str,
    *,
    label: str = "",
    weight: float = 1.0,
    data: dict | None = None,
) -> dict:
    return {
        "id": f"{edge_type}:{source}->{target}",
        "source": source,
        "target": target,
        "type": edge_type,
        "label": label,
        "weight": round(weight, 4),
        "data": data or {},
    }


def _solution_content_map(top_solutions: list[dict] | None) -> dict[str, str]:
    return {
        str(solution.get("id")): solution.get("content", "")
        for solution in (top_solutions or [])
        if solution.get("id")
    }


def _solution_position(index: int, total: int, generation: int) -> tuple[float, float]:
    if total <= 1:
        return (-300.0, 0.0)
    angle = (index / total) * math.tau
    radius = 260 + min(max(generation, 0), 8) * 26
    return (-220 + math.cos(angle) * radius, math.sin(angle) * radius)


def build_answer_graph(
    *,
    problem: str,
    final_answer: str,
    lineage_graph: dict | None = None,
    verification_report: dict | None = None,
    top_solutions: list[dict] | None = None,
    max_solution_nodes: int = 80,
) -> dict:
    """Create one graph that the UI can render as a mind-map style canvas."""
    lineage_graph = lineage_graph or {}
    verification_report = verification_report or {}
    solution_text = _solution_content_map(top_solutions)
    source_nodes = (lineage_graph.get("nodes") or [])[:max_solution_nodes]
    source_edges = lineage_graph.get("edges") or []
    claims = verification_report.get("claims") or []
    next_verifiers = verification_report.get("next_verifiers") or []

    nodes = [
        _node(
            "problem",
            "problem",
            "User Question",
            text=_excerpt(problem, 260),
            group="root",
            x=-640,
            y=0,
            data={"full_text": problem},
        ),
        _node(
            "answer:final",
            "final_answer",
            "Collapsed Answer",
            text=_excerpt(final_answer, 320),
            group="answer",
            x=320,
            y=0,
            data={
                "full_text": final_answer,
                "claim_count": verification_report.get("summary", {}).get("claim_count", 0),
                "average_confidence": verification_report.get("summary", {}).get("average_confidence", 0),
            },
        ),
    ]
    edges = [_edge("problem", "answer:final", "asks_for", label="evolves into", weight=0.6)]

    source_node_ids = set()
    for index, solution in enumerate(source_nodes):
        solution_id = str(solution.get("id", ""))
        if not solution_id:
            continue
        source_node_ids.add(solution_id)
        graph_id = f"solution:{solution_id}"
        x, y = _solution_position(index, max(len(source_nodes), 1), int(solution.get("generation") or 0))
        nodes.append(_node(
            graph_id,
            "answer_quantum",
            solution.get("label") or solution_id,
            text=_excerpt(solution_text.get(solution_id, ""), 180),
            group=solution.get("state", "solution"),
            x=x,
            y=y,
            data={
                "solution_id": solution_id,
                "generation": solution.get("generation", 0),
                "fitness": solution.get("fitness", 0),
                "state": solution.get("state", "active"),
                "mutation": solution.get("mutation"),
                "thinker_id": solution.get("thinker_id"),
                "cluster_id": solution.get("cluster_id"),
            },
        ))

        if solution.get("state") == "finalist":
            edges.append(_edge(graph_id, "answer:final", "collapses_to", label="finalist", weight=1.2))
        elif int(solution.get("generation") or 0) == 0:
            edges.append(_edge("problem", graph_id, "seeds", label="seed", weight=0.35))

    for edge in source_edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source not in source_node_ids or target not in source_node_ids:
            continue
        relation = edge.get("type", "evolves")
        edges.append(_edge(
            f"solution:{source}",
            f"solution:{target}",
            f"lineage_{relation}",
            label=relation,
            weight=0.8,
            data={"generation": edge.get("generation"), "detail": edge.get("detail")},
        ))

    claim_count = max(len(claims), 1)
    for index, claim in enumerate(claims):
        claim_id = str(claim.get("id", f"claim-{index + 1:02d}"))
        y = (index - (claim_count - 1) / 2) * 112
        nodes.append(_node(
            f"claim:{claim_id}",
            "claim",
            claim_id,
            text=_excerpt(claim.get("text", ""), 260),
            group=claim.get("kind", "claim"),
            x=760,
            y=y,
            data={
                "kind": claim.get("kind"),
                "status": claim.get("status"),
                "confidence": claim.get("confidence", 0),
                "risk_flags": claim.get("risk_flags", []),
            },
        ))
        edges.append(_edge(
            "answer:final",
            f"claim:{claim_id}",
            "contains_claim",
            label=claim.get("kind", "claim"),
            weight=max(float(claim.get("confidence", 0.2)), 0.2),
        ))

    for index, verifier in enumerate(next_verifiers[:6]):
        verifier_id = f"verifier:{verifier}"
        nodes.append(_node(
            verifier_id,
            "verifier",
            verifier,
            group="verifier",
            x=1120,
            y=(index - (min(len(next_verifiers), 6) - 1) / 2) * 96,
            data={"verifier": verifier},
        ))
        for claim in claims:
            if claim.get("risk_flags") or verifier in {"counterargument_judge", "external_source_check"}:
                edges.append(_edge(
                    f"claim:{claim.get('id')}",
                    verifier_id,
                    "needs_verifier",
                    label="verify",
                    weight=0.45,
                ))

    return {
        "version": "answer-graph.v1",
        "layout": {
            "engine": "client",
            "coordinate_space": "cartesian",
            "recommended_renderer": "svg-or-canvas",
        },
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "solution_node_count": len(source_node_ids),
            "claim_node_count": len(claims),
            "verifier_node_count": min(len(next_verifiers), 6),
            "truncated_solution_nodes": max(0, len(lineage_graph.get("nodes") or []) - max_solution_nodes),
        },
    }
