"""Depth metrics for cognitive profiles (v2)."""

import networkx as nx
from ..extract.schemas import ThoughtGraph, EdgeType


def max_elaboration_chain(graph: ThoughtGraph) -> float:
    """Longest directed path composed exclusively of ELAB edges."""
    if not graph.nodes:
        return 0.0
    G = nx.DiGraph()
    for n in graph.nodes:
        G.add_node(n.tu_id)
    for e in graph.edges:
        if e.edge_type == EdgeType.ELAB:
            G.add_edge(e.source, e.target)
    if not G.edges():
        return 0.0
    try:
        return float(max(nx.dag_longest_path_length(G), 0))
    except nx.NetworkXError:
        return 0.0


def mean_branch_depth(graph: ThoughtGraph) -> float:
    """Average depth from root nodes (in-degree=0) in the semantic subgraph."""
    if not graph.nodes:
        return 0.0
    G = nx.DiGraph()
    for n in graph.nodes:
        G.add_node(n.tu_id)
    for e in graph.edges:
        if not e.is_sequential:
            G.add_edge(e.source, e.target)
    roots = [v for v in G.nodes() if G.in_degree(v) == 0]
    if not roots:
        return 0.0
    all_depths = [
        d for root in roots
        for n, d in nx.single_source_shortest_path_length(G, root).items()
        if n != root
    ]
    return float(sum(all_depths) / len(all_depths)) if all_depths else 0.0


def specificity_gradient(graph: ThoughtGraph) -> float:
    """Slope of lexical specificity vs sequential position.

    Specificity proxy (spacy-free): ratio of concrete tokens —
    numbers, capitalised non-sentence-start words, and symbol-like
    tokens (%, $, units) — to total word-tokens per TU.
    A positive slope indicates the reasoning becomes more concrete
    and grounded over time.
    """
    if not graph.nodes:
        return 0.0
    import re
    import numpy as np

    _NUM    = re.compile(r'^-?\d+(?:[.,]\d+)*%?$')
    _SYMBOL = re.compile(r'[%$£€°]')
    _WORD   = re.compile(r'\b\w+\b')

    positions, densities = [], []
    for idx, node in enumerate(graph.nodes):
        words = _WORD.findall(node.text[:500])
        if not words:
            continue
        concrete = 0
        for wi, w in enumerate(words):
            if _NUM.match(w):
                concrete += 1
            elif _SYMBOL.search(w):
                concrete += 1
            elif w[0].isupper() and wi > 0:
                # Capitalised mid-sentence → likely proper noun / named entity
                concrete += 1
        positions.append(float(idx))
        densities.append(concrete / len(words))

    if len(positions) < 3:
        return 0.0
    try:
        return float(np.polyfit(positions, densities, 1)[0])
    except Exception:
        return 0.0


def reasoning_density(graph: ThoughtGraph) -> float:
    """Fraction of nodes participating in at least one semantic (non-SEQ) edge."""
    if not graph.nodes:
        return 0.0
    sem_nodes: set[int] = set()
    for e in graph.edges:
        if not e.is_sequential:
            sem_nodes.add(e.source)
            sem_nodes.add(e.target)
    return len(sem_nodes) / len(graph.nodes)
