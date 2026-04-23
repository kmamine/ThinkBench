"""Breadth metrics for cognitive profiles (v2)."""

from ..extract.schemas import NodeType, ThoughtGraph, EdgeType
from ..utils.models import get_embed_model


def branching_factor(graph: ThoughtGraph) -> float:
    """|E_BRCH| / |V|"""
    if not graph.nodes:
        return 0.0
    brch = sum(1 for e in graph.edges if e.edge_type == EdgeType.BRCH)
    return brch / len(graph.nodes)


def unique_perspective_count(graph: ThoughtGraph) -> float:
    """|N_RFR|"""
    return float(sum(1 for n in graph.nodes if n.node_type == NodeType.RFR))


def domain_spread(graph: ThoughtGraph) -> float:
    """Distinct semantic clusters among BRS+HYP nodes (agglomerative, cosine threshold=0.45)."""
    target = [n for n in graph.nodes if n.node_type in (NodeType.BRS, NodeType.HYP)]
    if len(target) < 2:
        return float(len(target))
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        model = get_embed_model()
        embs = model.encode([n.text[:300] for n in target], normalize_embeddings=True)
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.45, metric="cosine", linkage="average"
        ).fit_predict(embs)
        return float(len(set(labels)))
    except ImportError:
        return float(len(target))


def first_idea_diversity(graph: ThoughtGraph) -> float:
    """Mean pairwise cosine distance among the first HYP node embeddings within this trace."""
    hyp = [n for n in graph.nodes if n.node_type == NodeType.HYP]
    if len(hyp) < 2:
        return 0.0
    try:
        import numpy as np
        model = get_embed_model()
        embs = model.encode([n.text[:200] for n in hyp[:3]], normalize_embeddings=True)
        n = len(embs)
        dists = [1.0 - float(np.dot(embs[i], embs[j])) for i in range(n) for j in range(i + 1, n)]
        return sum(dists) / len(dists) if dists else 0.0
    except ImportError:
        return 0.0
