"""Metacognitive metrics for cognitive profiles."""

import re
from ..extract.schemas import ThoughtGraph, NodeType


def self_reflection_rate(graph: ThoughtGraph) -> float:
    """Proportion of MET nodes."""
    if not graph.nodes:
        return 0.0

    met_count = sum(1 for n in graph.nodes if n.node_type == NodeType.MET)

    return met_count / len(graph.nodes)


def critique_to_hypothesis_ratio(graph: ThoughtGraph) -> float:
    """CRT nodes / HYP nodes."""
    if not graph.nodes:
        return 0.0

    crt_count = sum(1 for n in graph.nodes if n.node_type == NodeType.CRT)
    hyp_count = sum(1 for n in graph.nodes if n.node_type == NodeType.HYP)

    if hyp_count == 0:
        return float(crt_count) if crt_count > 0 else 0.0

    return crt_count / hyp_count


def hedging_density(graph: ThoughtGraph) -> float:
    """Proportion of TUs containing uncertainty markers."""
    if not graph.nodes:
        return 0.0

    hedge_patterns = [
        r"\bmight\b",
        r"\bmaybe\b",
        r"\bcould\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\bprobably\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bit seems\b",
        r"\bit appears\b",
        r"\bmay be\b",
        r"\bmay not\b",
        r"\buncertain\b",
        r"\bunclear\b",
        r"\bnot sure\b",
        r"\blikely\b",
    ]

    hedged = 0
    for node in graph.nodes:
        text_lower = node.text.lower()
        if any(re.search(p, text_lower) for p in hedge_patterns):
            hedged += 1

    return hedged / len(graph.nodes)


def perspective_taking(graph: ThoughtGraph) -> float:
    """Proportion of RFR nodes."""
    if not graph.nodes:
        return 0.0

    rfr_count = sum(1 for n in graph.nodes if n.node_type == NodeType.RFR)

    return rfr_count / len(graph.nodes)
