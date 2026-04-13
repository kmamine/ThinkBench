"""Breadth metrics for cognitive profiles."""

import networkx as nx
from ..extract.schemas import ThoughtGraph, NodeType, NodeFamily


def branching_factor(graph: ThoughtGraph) -> float:
    """Mean out-degree of EXPLORATION-family nodes via BRCH edges."""
    if not graph.nodes:
        return 0.0

    G = _build_nx_graph(graph)
    exp_nodes = [n for n in graph.nodes if _is_exploration(n)]

    if not exp_nodes:
        return 0.0

    brch_edges = [e for e in graph.edges if e.edge_type.value == "BRCH"]

    out_degrees = {}
    for e in brch_edges:
        if e.source in exp_nodes:
            out_degrees[e.source] = out_degrees.get(e.source, 0) + 1

    if not out_degrees:
        return 0.0

    return sum(out_degrees.values()) / len(out_degrees)


def unique_perspective_count(graph: ThoughtGraph) -> float:
    """Number of connected components in EXPLORATION subgraph (no SEQ edges)."""
    if not graph.nodes:
        return 0.0

    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.tu_id)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)

    exp_nodes = [n.tu_id for n in graph.nodes if _is_exploration(n)]
    if not exp_nodes:
        return 0.0

    exp_subgraph = G.subgraph(exp_nodes).to_undirected()

    seq_edges = [
        (e.source, e.target) for e in graph.edges if e.edge_type.value == "SEQ"
    ]
    exp_subgraph.remove_edges_from(seq_edges)

    components = list(nx.connected_components(exp_subgraph))
    return float(len(components))


def domain_spread(graph: ThoughtGraph) -> float:
    """For policy/ethics: count distinct stakeholder perspectives."""
    if not graph.nodes:
        return 0.0

    stakeholders = set()
    stakeholder_markers = [
        r"\b(passenger|driver|user|customer)\b",
        r"\bpedestrian\b",
        r"\bgovernment\b",
        r"\bmanufacturer\b",
        r"\binsurance\b",
        r"\blaw\b",
        r"\bsociety\b",
        r"\bpublic\b",
    ]

    import re

    for node in graph.nodes:
        text_lower = node.text.lower()
        for pattern in stakeholder_markers:
            if re.search(pattern, text_lower):
                stakeholders.add(pattern)

    return float(len(stakeholders))


def first_idea_diversity(graph: ThoughtGraph) -> float:
    """Cosine distance between embeddings of first 3 EXPLORATION nodes."""
    if not graph.nodes:
        return 0.0

    exp_nodes = [n for n in graph.nodes if _is_exploration(n)]
    if len(exp_nodes) < 2:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        texts = [n.text[:200] for n in exp_nodes[:3]]
        embeddings = model.encode(texts)

        if len(embeddings) < 2:
            return 0.0

        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(embeddings)

        n = len(embeddings)
        total_sim = sum(sim_matrix[i][j] for i in range(n) for j in range(i + 1, n))
        pairs = n * (n - 1) / 2
        avg_sim = total_sim / pairs if pairs > 0 else 0

        return 1.0 - avg_sim
    except ImportError:
        return 0.0


def _is_exploration(node) -> bool:
    return node.node_family == NodeFamily.EXPLORATION if node.node_family else False


def _build_nx_graph(graph: ThoughtGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.tu_id)
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)
    return G
