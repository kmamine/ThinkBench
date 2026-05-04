"""Efficiency metrics for cognitive profiles (v2)."""

import numpy as np
from ..extract.schemas import ThoughtGraph
from ..utils.models import get_embed_model


def token_per_idea(graph: ThoughtGraph) -> float:
    """Total tokens / unique_perspective_count (|N_RFR|)."""
    from .breadth import unique_perspective_count
    upc = unique_perspective_count(graph)
    return graph.token_count / upc if upc > 0 else graph.token_count / max(len(graph.nodes), 1)


def redundancy_ratio(graph: ThoughtGraph) -> float:
    """Fraction of TU pairs with cosine similarity > 0.90."""
    if len(graph.nodes) < 2:
        return 0.0
    try:
        model = get_embed_model()
        embs = model.encode([n.text[:300] for n in graph.nodes], normalize_embeddings=True)
        n = len(embs)
        high = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if float(np.dot(embs[i], embs[j])) > 0.75
        )
        total = n * (n - 1) // 2
        return high / total if total > 0 else 0.0
    except ImportError:
        return 0.0
