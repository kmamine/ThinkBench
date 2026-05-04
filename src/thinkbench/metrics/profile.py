"""Profile computation — aggregates all 22 metrics into CognitiveProfile (v2)."""

from collections import defaultdict

from ..extract.schemas import ThoughtGraph, CognitiveProfile
from . import breadth, depth, structure, metacognitive, efficiency


def compute_profile(
    graph: ThoughtGraph, model: str, domain: str | None = None
) -> CognitiveProfile:
    """Compute all 22 metrics and return a CognitiveProfile."""
    return CognitiveProfile(
        model=model,
        domain=domain,
        trace_id=graph.trace_id,
        # Breadth
        branching_factor=breadth.branching_factor(graph),
        unique_perspective_count=breadth.unique_perspective_count(graph),
        domain_spread=breadth.domain_spread(graph),
        first_idea_diversity=breadth.first_idea_diversity(graph),
        # Depth
        max_elaboration_chain=depth.max_elaboration_chain(graph),
        mean_branch_depth=depth.mean_branch_depth(graph),
        specificity_gradient=depth.specificity_gradient(graph),
        reasoning_density=depth.reasoning_density(graph),
        # Structure
        exploration_exploitation_ratio=structure.exploration_exploitation_ratio(graph),
        backtracking_rate=structure.backtracking_rate(graph),
        cross_branch_connectivity=structure.cross_branch_connectivity(graph),
        convergence_index=structure.convergence_index(graph),
        graph_density=structure.graph_density(graph),
        revision_depth=structure.revision_depth(graph),
        # Metacognitive
        self_reflection_rate=metacognitive.self_reflection_rate(graph),
        critique_to_hypothesis_ratio=metacognitive.critique_to_hypothesis_ratio(graph),
        hedging_density=metacognitive.hedging_density(graph),
        perspective_taking=metacognitive.perspective_taking(graph),
        # Efficiency
        token_per_idea=efficiency.token_per_idea(graph),
        redundancy_ratio=efficiency.redundancy_ratio(graph),
        # Summary
        avg_tokens=float(graph.token_count),
        avg_tus=float(len(graph.nodes)),
    )


def aggregate_profiles(
    profiles: list[CognitiveProfile],
    trace_model_map: dict[str, str] | None = None,
    trace_prompt_map: dict[str, str] | None = None,
) -> list[dict]:
    """Group profiles by (model, prompt_variant) and average all numeric metrics."""
    groups: dict[tuple[str, str], list[CognitiveProfile]] = defaultdict(list)
    for p in profiles:
        model_name = (trace_model_map or {}).get(p.trace_id or "", p.model)
        prompt_variant = (trace_prompt_map or {}).get(p.trace_id or "", "normal")
        groups[(model_name, prompt_variant)].append(p)

    result = []
    for (model_name, prompt_variant), profs in sorted(groups.items()):
        metrics: dict[str, list[float]] = defaultdict(list)
        for p in profs:
            for field, value in p.model_dump().items():
                if field not in ("model", "domain", "trace_id") and isinstance(value, (int, float)):
                    metrics[field].append(float(value))
        row = {"model": model_name, "prompt_variant": prompt_variant, "num_traces": len(profs)}
        row.update({k: sum(v) / len(v) for k, v in sorted(metrics.items()) if v})
        result.append(row)
    return result
