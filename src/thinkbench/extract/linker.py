"""Graph assembly (v2). Segmentation and edge typing now happen in segmenter.py."""

from .schemas import Edge, ThoughtGraph, ThoughtUnit


def build_graph(
    trace: dict,
    thought_units: list[ThoughtUnit],
    edges: list[Edge],
) -> ThoughtGraph:
    """Assemble a ThoughtGraph from pre-computed TUs and edges."""
    return ThoughtGraph(
        trace_id=trace["trace_id"],
        model=trace["model"],
        question_id=trace["question_id"],
        domain=trace["domain"],
        run=trace["run"],
        nodes=thought_units,
        edges=edges,
        token_count=trace.get("token_count", len(trace.get("raw_cot", "").split())),
    )
