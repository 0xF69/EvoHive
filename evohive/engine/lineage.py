from evohive.models import Solution


def _solution_label(solution: Solution) -> str:
    if solution.thinker_id:
        return solution.thinker_id
    if solution.parent_a_id or solution.parent_b_id:
        return "derived"
    if solution.seed_content:
        return "swarm-seed"
    return "solution"


def build_lineage_graph(solutions: list[Solution], final_ids: set[str] | None = None) -> dict:
    """Build a compact graph of solution ancestry and evolutionary state."""
    final_ids = final_ids or set()
    nodes = []
    edges = []
    seen_edges = set()

    for solution in solutions:
        state = "finalist" if solution.id in final_ids else "active" if solution.is_alive else "eliminated"
        if solution.elimination_reason:
            state = "eliminated"
        if solution.mutation_applied:
            state = "mutated"
        if solution.parent_a_id or solution.parent_b_id:
            state = "derived" if solution.id not in final_ids else "finalist"

        nodes.append({
            "id": solution.id,
            "label": _solution_label(solution),
            "generation": solution.generation,
            "fitness": round(solution.fitness or 0.0, 6),
            "raw_fitness": round(solution.raw_fitness or 0.0, 6),
            "diversity_bonus": round(solution.diversity_bonus or 0.0, 6),
            "state": state,
            "thinker_id": solution.thinker_id,
            "cluster_id": solution.cluster_id,
            "mutation": solution.mutation_applied,
            "elimination_reason": solution.elimination_reason,
            "red_team_vulnerability": solution.red_team_vulnerability,
            "pressure_resilience": solution.pressure_resilience,
            "debate_elo": solution.debate_elo,
            "execution_score": solution.execution_score,
        })

        for parent_id, relation in (
            (solution.parent_a_id, "parent_a"),
            (solution.parent_b_id, "parent_b"),
        ):
            if not parent_id:
                continue
            edge_key = (parent_id, solution.id, relation)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edges.append({
                "source": parent_id,
                "target": solution.id,
                "type": relation,
                "generation": solution.generation,
            })

        if solution.mutation_applied:
            edges.append({
                "source": solution.id,
                "target": solution.id,
                "type": "mutation",
                "generation": solution.generation,
                "detail": solution.mutation_applied,
            })

    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "finalist_count": len(final_ids),
            "mutated_count": sum(1 for node in nodes if node["mutation"]),
            "eliminated_count": sum(1 for node in nodes if node["state"] == "eliminated"),
        },
    }
