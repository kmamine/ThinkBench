"""Structure metrics for cognitive profiles (v2)."""

import networkx as nx
from ..extract.schemas import ThoughtGraph, NodeType, NodeFamily, EdgeType


def exploration_exploitation_ratio(graph: ThoughtGraph) -> float:
    """|N_EXPLORATION| / max(|N_ELABORATION|, 1)"""
    if not graph.nodes:
        return 0.0
    exp = sum(1 for n in graph.nodes if n.node_family == NodeFamily.EXPLORATION)
    ela = sum(1 for n in graph.nodes if n.node_family == NodeFamily.ELABORATION)
    return exp / max(ela, 1)


def backtracking_rate(graph: ThoughtGraph) -> float:
    """|E_BACK| / max(|E_sem|, 1)"""
    if not graph.nodes:
        return 0.0
    e_back = sum(1 for e in graph.edges if e.edge_type == EdgeType.BACK)
    e_sem = sum(1 for e in graph.edges if not e.is_sequential)
    return e_back / max(e_sem, 1)


def cross_branch_connectivity(graph: ThoughtGraph) -> float:
    """Fraction of branch pairs connected by at least one SYNT or SUPP edge."""
    if not graph.nodes:
        return 0.0
    G_brch = nx.Graph()
    for n in graph.nodes:
        G_brch.add_node(n.tu_id)
    for e in graph.edges:
        if e.edge_type == EdgeType.BRCH:
            G_brch.add_edge(e.source, e.target)
    components = list(nx.connected_components(G_brch))
    if len(components) < 2:
        return 0.0
    node_branch = {nid: b for b, comp in enumerate(components) for nid in comp}
    cross = [
        e for e in graph.edges
        if e.edge_type in (EdgeType.SYNT, EdgeType.SUPP)
        and node_branch.get(e.source, -1) != node_branch.get(e.target, -2)
    ]
    total_pairs = len(components) * (len(components) - 1) / 2
    connected = {(min(node_branch[e.source], node_branch[e.target]),
                  max(node_branch[e.source], node_branch[e.target]))
                 for e in cross
                 if e.source in node_branch and e.target in node_branch}
    return len(connected) / total_pairs


def convergence_index(graph: ThoughtGraph) -> float:
    """sum(d_in(SYN nodes)) / (|V| * mean_d_in) in the semantic subgraph."""
    if not graph.nodes:
        return 0.0
    syn_ids = {n.tu_id for n in graph.nodes if n.node_type == NodeType.SYN}
    if not syn_ids:
        return 0.0
    in_deg: dict[int, int] = {n.tu_id: 0 for n in graph.nodes}
    for e in graph.edges:
        if not e.is_sequential:
            in_deg[e.target] = in_deg.get(e.target, 0) + 1
    mean_d_in = sum(in_deg.values()) / max(len(in_deg), 1)
    if mean_d_in == 0:
        return 0.0
    return sum(in_deg.get(sid, 0) for sid in syn_ids) / (len(graph.nodes) * mean_d_in)


def graph_density(graph: ThoughtGraph) -> float:
    """|E_sem| / (|V| * (|V| - 1)) — semantic edges only."""
    n = len(graph.nodes)
    if n < 2:
        return 0.0
    e_sem = sum(1 for e in graph.edges if not e.is_sequential)
    return e_sem / (n * (n - 1))


def revision_depth(graph: ThoughtGraph) -> float:
    """Mean sequential distance between endpoints of BACK edges."""
    distances = [abs(e.source - e.target) for e in graph.edges if e.edge_type == EdgeType.BACK]
    return sum(distances) / len(distances) if distances else 0.0
