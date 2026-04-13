"""Profile computation - aggregates all metrics into CognitiveProfile."""

from ..extract.schemas import ThoughtGraph, CognitiveProfile
from . import breadth, depth, structure, metacognitive, efficiency


def compute_profile(
    graph: ThoughtGraph, model: str, domain: str | None = None
) -> CognitiveProfile:
    """Compute all 22 metrics and aggregate into CognitiveProfile."""

    return CognitiveProfile(
        model=model,
        domain=domain,
        branching_factor=breadth.branching_factor(graph),
        unique_perspective_count=breadth.unique_perspective_count(graph),
        domain_spread=breadth.domain_spread(graph),
        first_idea_diversity=breadth.first_idea_diversity(graph),
        max_elaboration_chain=depth.max_elaboration_chain(graph),
        mean_branch_depth=depth.mean_branch_depth(graph),
        specificity_gradient=depth.specificity_gradient(graph),
        reasoning_density=depth.reasoning_density(graph),
        exploration_exploitation_ratio=structure.exploration_exploitation_ratio(graph),
        backtracking_rate=structure.backtracking_rate(graph),
        cross_branch_connectivity=structure.cross_branch_connectivity(graph),
        convergence_index=structure.convergence_index(graph),
        orphan_ratio=structure.orphan_ratio(graph),
        graph_density=structure.graph_density(graph),
        cycle_count=structure.cycle_count(graph),
        mean_cycle_length=structure.mean_cycle_length(graph),
        self_reflection_rate=metacognitive.self_reflection_rate(graph),
        critique_to_hypothesis_ratio=metacognitive.critique_to_hypothesis_ratio(graph),
        hedging_density=metacognitive.hedging_density(graph),
        perspective_taking=metacognitive.perspective_taking(graph),
        token_per_idea=efficiency.token_per_idea(graph),
        redundancy_ratio=efficiency.redundancy_ratio(graph),
    )
