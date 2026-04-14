"""Depth metrics for cognitive profiles."""

import logging

import networkx as nx
from ..extract.schemas import ThoughtGraph, NodeType, NodeFamily

logger = logging.getLogger(__name__)


def max_elaboration_chain(graph: ThoughtGraph) -> float:
    """Longest path following only ELAB edges (ELAB-only subgraph is acyclic)."""
    if not graph.nodes:
        return 0.0

    G = _build_elab_graph(graph)

    if not G.nodes():
        return 0.0

    try:
        longest = max(nx.dag_longest_path_length(G), 0)
        return float(longest)
    except nx.NetworkXError:
        return 0.0


def mean_branch_depth(graph: ThoughtGraph) -> float:
    """Average depth across all branches rooted at EXPLORATION nodes."""
    if not graph.nodes:
        return 0.0

    G = _build_elab_graph(graph)
    exp_nodes = [
        n.tu_id for n in graph.nodes if n.node_family == NodeFamily.EXPLORATION
    ]

    if not exp_nodes:
        return 0.0

    depths = []
    for root in exp_nodes:
        if root not in G:
            continue
        try:
            lengths = nx.single_source_shortest_path_length(G, root)
            branch_lengths = [l for n, l in lengths.items() if n != root]
            depths.extend(branch_lengths)
        except nx.NetworkXError:
            continue

    return float(sum(depths) / len(depths)) if depths else 0.0


def specificity_gradient(graph: ThoughtGraph) -> float:
    """Correlation between node depth and entity density."""
    import logging

    logger = logging.getLogger(__name__)

    if not graph.nodes:
        return 0.0

    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except ImportError:
        logger.debug("spaCy not available, returning 0.0 for specificity_gradient")
        return 0.0
    except Exception as e:
        logger.debug(
            f"Failed to load spaCy model: {e}, returning 0.0 for specificity_gradient"
        )
        return 0.0

    G = _build_elab_graph(graph)

    depth_entity_counts = []
    for node in graph.nodes:
        if node.tu_id not in G:
            continue
        try:
            depth = nx.shortest_path_length(G, list(G.nodes())[0], node.tu_id)
        except:
            depth = 0

        doc = nlp(node.text[:500])
        entity_count = len(doc.ents)
        token_count = len(doc)

        if token_count > 0:
            density = entity_count / token_count
            depth_entity_counts.append((depth, density))

    if len(depth_entity_counts) < 3:
        return 0.0

    depths = [d for d, _ in depth_entity_counts]
    densities = [e for _, e in depth_entity_counts]

    if len(set(depths)) < 2:
        return 0.0

    try:
        from scipy import stats

        corr, _ = stats.pearsonr(depths, densities)
        return float(corr) if abs(corr) > 0 else 0.0
    except:
        return 0.0


def reasoning_density(graph: ThoughtGraph) -> float:
    """Ratio of (JUS + IMP + CON nodes) to total nodes."""
    if not graph.nodes:
        return 0.0

    reasoning_types = {NodeType.JUS, NodeType.IMP, NodeType.CON}
    reasoning_count = sum(1 for n in graph.nodes if n.node_type in reasoning_types)

    return reasoning_count / len(graph.nodes)


def _build_elab_graph(graph: ThoughtGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node.tu_id)

    for edge in graph.edges:
        if edge.edge_type.value in (
            "ELAB",
            "SPC",
            "IMP",
            "JUS",
            "CON",
            "SEQ",
            "BRCH",
            "SUPP",
        ):
            G.add_edge(edge.source, edge.target)

    return G
