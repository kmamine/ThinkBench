"""Structure metrics for cognitive profiles."""

import logging

import networkx as nx
from ..extract.schemas import ThoughtGraph, NodeType, NodeFamily

logger = logging.getLogger(__name__)


def exploration_exploitation_ratio(graph: ThoughtGraph) -> float:
    """(EXPLORATION nodes) / (ELABORATION nodes)."""
    if not graph.nodes:
        return 0.0

    exp_count = sum(1 for n in graph.nodes if n.node_family == NodeFamily.EXPLORATION)
    ela_count = sum(1 for n in graph.nodes if n.node_family == NodeFamily.ELABORATION)

    if ela_count == 0:
        return float(exp_count) if exp_count > 0 else 0.0

    return exp_count / ela_count


def backtracking_rate(graph: ThoughtGraph) -> float:
    """Proportion of nodes with outgoing BACK or CRIT edges."""
    if not graph.nodes:
        return 0.0

    back_nodes = {
        e.source for e in graph.edges if e.edge_type.value in ("BACK", "CRIT")
    }

    return len(back_nodes) / len(graph.nodes)


def cross_branch_connectivity(graph: ThoughtGraph) -> float:
    """Number of SYNT edges / number of branches."""
    if not graph.nodes:
        return 0.0

    synt_edges = [e for e in graph.edges if e.edge_type.value == "SYNT"]
    if not synt_edges:
        return 0.0

    exp_nodes = [
        n.tu_id for n in graph.nodes if n.node_family == NodeFamily.EXPLORATION
    ]
    if not exp_nodes:
        return 0.0

    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.tu_id)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)

    branches = 0
    for exp in exp_nodes:
        if exp in G:
            descendants = nx.descendants(G, exp)
            if descendants:
                branches += 1

    if branches == 0:
        return float(len(synt_edges))

    return len(synt_edges) / branches


def convergence_index(graph: ThoughtGraph) -> float:
    """(SYN nodes in last quartile) / (total SYN nodes)."""
    if not graph.nodes:
        return 0.0

    syn_nodes = [n for n in graph.nodes if n.node_type == NodeType.SYN]
    if not syn_nodes:
        return 0.0

    n = len(graph.nodes)
    quartile_start = (3 * n) // 4

    late_synth = sum(1 for n in syn_nodes if n.tu_id >= quartile_start)

    return late_synth / len(syn_nodes)


def orphan_ratio(graph: ThoughtGraph) -> float:
    """Proportion of EXPLORATION nodes with no elaboration children."""
    if not graph.nodes:
        return 0.0

    exp_nodes = [n for n in graph.nodes if n.node_family == NodeFamily.EXPLORATION]
    if not exp_nodes:
        return 0.0

    orphans = 0
    for exp in exp_nodes:
        has_elab_child = any(
            e.source == exp.tu_id
            and e.edge_type.value in ("ELAB", "SPC", "IMP", "JUS", "BRCH", "SUPP")
            for e in graph.edges
        )
        if not has_elab_child:
            orphans += 1

    return orphans / len(exp_nodes)


def graph_density(graph: ThoughtGraph) -> float:
    """|E| / (|V| * (|V|-1) / 2)"""
    if not graph.nodes:
        return 0.0

    n = len(graph.nodes)
    if n < 2:
        return 0.0

    max_edges = n * (n - 1) / 2
    return len(graph.edges) / max_edges


def cycle_count(graph: ThoughtGraph) -> float:
    """Number of distinct simple cycles."""
    if not graph.nodes:
        return 0.0

    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.tu_id)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)

    try:
        cycles = list(nx.simple_cycles(G))
        return float(len(cycles))
    except:
        return 0.0


def mean_cycle_length(graph: ThoughtGraph) -> float:
    """Mean length of all simple cycles (capped at 10k cycles)."""
    if not graph.nodes:
        return 0.0

    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.tu_id)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)

    try:
        cycles = list(nx.simple_cycles(G))
        if not cycles:
            return 0.0

        cycles = cycles[:10000]
        lengths = [len(c) for c in cycles]
        return sum(lengths) / len(lengths)
    except:
        return 0.0
