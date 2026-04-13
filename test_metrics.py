"""Test script for metrics computation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkbench.extract.schemas import ThoughtGraph
from thinkbench.metrics import compute_profile


def test_metrics():
    """Test metrics on extracted graph."""
    graph_path = Path("data/graphs/graph_test.json")

    with open(graph_path) as f:
        graph_data = json.load(f)

    graph = ThoughtGraph(**graph_data)

    print(f"Testing metrics on graph: {graph.trace_id}")
    print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
    print()

    profile = compute_profile(graph, model=graph.model, domain=graph.domain)

    print("=== Cognitive Profile (22 metrics) ===")
    print(f"Model: {profile.model}")
    print(f"Domain: {profile.domain}")
    print()

    print("Breadth:")
    print(f"  branching_factor: {profile.branching_factor:.3f}")
    print(f"  unique_perspective_count: {profile.unique_perspective_count:.3f}")
    print(f"  domain_spread: {profile.domain_spread:.3f}")
    print(f"  first_idea_diversity: {profile.first_idea_diversity:.3f}")
    print()

    print("Depth:")
    print(f"  max_elaboration_chain: {profile.max_elaboration_chain:.3f}")
    print(f"  mean_branch_depth: {profile.mean_branch_depth:.3f}")
    print(f"  specificity_gradient: {profile.specificity_gradient:.3f}")
    print(f"  reasoning_density: {profile.reasoning_density:.3f}")
    print()

    print("Structure:")
    print(
        f"  exploration_exploitation_ratio: {profile.exploration_exploitation_ratio:.3f}"
    )
    print(f"  backtracking_rate: {profile.backtracking_rate:.3f}")
    print(f"  cross_branch_connectivity: {profile.cross_branch_connectivity:.3f}")
    print(f"  convergence_index: {profile.convergence_index:.3f}")
    print(f"  orphan_ratio: {profile.orphan_ratio:.3f}")
    print(f"  graph_density: {profile.graph_density:.3f}")
    print(f"  cycle_count: {profile.cycle_count:.3f}")
    print(f"  mean_cycle_length: {profile.mean_cycle_length:.3f}")
    print()

    print("Metacognitive:")
    print(f"  self_reflection_rate: {profile.self_reflection_rate:.3f}")
    print(f"  critique_to_hypothesis_ratio: {profile.critique_to_hypothesis_ratio:.3f}")
    print(f"  hedging_density: {profile.hedging_density:.3f}")
    print(f"  perspective_taking: {profile.perspective_taking:.3f}")
    print()

    print("Efficiency:")
    print(f"  token_per_idea: {profile.token_per_idea:.3f}")
    print(f"  redundancy_ratio: {profile.redundancy_ratio:.3f}")
    print()

    print("=== Profile Vector ===")
    vec = profile.to_vector()
    print(f"22-dim vector: [{', '.join(f'{v:.2f}' for v in vec)}]")

    output_path = Path("data/profiles/profile_test.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile.model_dump(), f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    test_metrics()
