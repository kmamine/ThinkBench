"""Efficiency metrics for cognitive profiles."""

import numpy as np
from ..extract.schemas import ThoughtGraph, NodeFamily


def token_per_idea(graph: ThoughtGraph) -> float:
    """Total tokens / UPC (Unique Perspective Count)."""
    if not graph.nodes:
        return 0.0

    exp_nodes = [n for n in graph.nodes if n.node_family == NodeFamily.EXPLORATION]
    if not exp_nodes:
        return float(graph.token_count)

    from ..metrics.breadth import unique_perspective_count

    upc = unique_perspective_count(graph)

    if upc == 0:
        return float(graph.token_count)

    return graph.token_count / upc


def redundancy_ratio(graph: ThoughtGraph) -> float:
    """Proportion of TU pairs with embedding similarity > 0.90."""
    if len(graph.nodes) < 2:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [n.text[:300] for n in graph.nodes]
        embeddings = model.encode(texts)

        sim_matrix = cosine_similarity(embeddings)

        n = len(embeddings)
        high_sim_pairs = 0
        total_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > 0.90:
                    high_sim_pairs += 1
                total_pairs += 1

        return high_sim_pairs / total_pairs if total_pairs > 0 else 0.0

    except ImportError:
        return 0.0
