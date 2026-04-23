"""
Node type classifier (v2).

Primary: deterministic rule based on BoundaryClass from Pass 1.
Placeholder slot for a fine-tuned DeBERTa classifier once silver-label
training data is available.
"""

from .schemas import (
    BoundaryClass,
    BOUNDARY_NODE_MAP,
    NODE_FAMILY_MAP,
    NodeType,
    ThoughtUnit,
)


def classify_nodes(tus: list[ThoughtUnit]) -> list[ThoughtUnit]:
    """
    Assign node_type, node_family, and classification_confidence to each TU.

    Uses the deterministic boundary_class → node_type mapping.
    The first TU in the trace is always HYP (it opens the reasoning).
    Hard-boundary TUs (boundary_class != NONE) get confidence 1.0.
    Soft-boundary TUs (NONE) get confidence 0.5.
    """
    for idx, tu in enumerate(tus):
        if tu.boundary_class == BoundaryClass.NONE and idx == 0:
            node_type = NodeType.HYP
        else:
            node_type = BOUNDARY_NODE_MAP[tu.boundary_class]

        tu.node_type = node_type
        tu.node_family = NODE_FAMILY_MAP[node_type]
        tu.classification_confidence = (
            1.0 if tu.boundary_class != BoundaryClass.NONE else 0.5
        )

    return tus


def load_deberta_classifier(model_path: str):
    """Placeholder for the fine-tuned DeBERTa-base node classifier.
    Raises NotImplementedError until silver-label training is complete."""
    raise NotImplementedError(
        "Fine-tuned DeBERTa classifier not yet available. "
        "Run the silver-label generation pipeline first."
    )
