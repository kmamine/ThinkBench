"""
Node type classifier (v2 — edge-structure-based).

Classification priority (highest wins):
  1. Outgoing SYNT edges to ≥2 distinct targets              → SYN
  2. Outgoing BACK edge                                       → CRT
  3. boundary_class == META                                   → MET
  4. Incoming BRCH edge + semantically novel (RFR condition)  → RFR
  5. Incoming BRCH edge (no embeddings or sim outside range)  → HYP
  6. Outgoing CONT edge                                       → CMP
  7. Outgoing SUPP edge                                       → JUS
  8. boundary_class == CONVERGENCE                            → SYN
  Default                                                     → SPC

RFR vs HYP split (priority 4 vs 5):
  A branch-starting TU is classified RFR (reframing) when it re-engages
  with the original problem from a different angle — captured as
  0.25 < cos_sim(TU, TU_0) < 0.72.  Outside this range it is either
  too dissimilar (genuinely new hypothesis → HYP) or too similar
  (minor restatement → SPC via default).

Confidence:
  - Hard-boundary TUs (boundary_class != NONE): 1.0
  - Soft-boundary / default:                    0.5
"""

from __future__ import annotations

import numpy as np

from .schemas import (
    BoundaryClass,
    EdgeType,
    NODE_FAMILY_MAP,
    NodeType,
    ThoughtUnit,
    Edge,
)

# Similarity bounds for RFR detection (distance from first TU)
_RFR_SIM_LO: float = 0.25
_RFR_SIM_HI: float = 0.72

# Minimum SYNT out-degree to classify a node as SYN via edge structure
_SYN_MIN_TARGETS: int = 2

# MET detection: TU in the second half of the trace with high similarity
# to TU₀ (returning to original framing = metacognitive self-check)
# Only fires when no structural role was already assigned.
_MET_SIM_THRESH: float = 0.60   # minimum cos_sim to TU₀ to be considered meta
_MET_POS_FRAC: float   = 0.45   # must be past this fraction of the trace


def classify_nodes(
    tus: list[ThoughtUnit],
    edges: list[Edge],
    embeddings: np.ndarray | None = None,
) -> list[ThoughtUnit]:
    """Assign node_type, node_family, and classification_confidence to every TU.

    Parameters
    ----------
    tus:        Ordered list of ThoughtUnits (tu_id == position index).
    edges:      Edge list produced by segmenter (may include SEQ edges).
    embeddings: (N, D) float32 array, one row per TU, L2-normalised.
                Required for RFR detection; if None, all branch-receiving
                nodes default to HYP.
    """
    # Build per-node edge index
    out_edges: dict[int, list[Edge]] = {tu.tu_id: [] for tu in tus}
    in_edges: dict[int, list[Edge]] = {tu.tu_id: [] for tu in tus}
    for e in edges:
        out_edges.setdefault(e.source, []).append(e)
        in_edges.setdefault(e.target, []).append(e)

    first_emb: np.ndarray | None = embeddings[0] if embeddings is not None and len(embeddings) > 0 else None
    n_tus = len(tus)

    for idx, tu in enumerate(tus):
        outs = out_edges.get(tu.tu_id, [])
        ins = in_edges.get(tu.tu_id, [])

        out_types = {e.edge_type for e in outs}
        in_types = {e.edge_type for e in ins}

        node_type = _resolve(tu, idx, n_tus, outs, out_types, in_types, embeddings, first_emb)

        tu.node_type = node_type
        tu.node_family = NODE_FAMILY_MAP[node_type]
        tu.classification_confidence = (
            1.0 if tu.boundary_class != BoundaryClass.NONE else 0.5
        )

    return tus


def _resolve(
    tu: ThoughtUnit,
    idx: int,
    n_tus: int,
    outs: list[Edge],
    out_types: set[EdgeType],
    in_types: set[EdgeType],
    embeddings: np.ndarray | None,
    first_emb: np.ndarray | None,
) -> NodeType:
    # Priority 1: multi-target synthesis hub
    synt_targets = {e.target for e in outs if e.edge_type == EdgeType.SYNT}
    if len(synt_targets) >= _SYN_MIN_TARGETS:
        return NodeType.SYN

    # Priority 2: originates a backward loop → evaluative critique
    if EdgeType.BACK in out_types:
        return NodeType.CRT

    # Priority 3: explicit meta-reflection marker
    if tu.boundary_class == BoundaryClass.META:
        return NodeType.MET

    # Priority 4 / 5: starts a new branch
    if EdgeType.BRCH in in_types:
        if embeddings is not None and first_emb is not None and idx > 0:
            sim = float(np.dot(embeddings[idx], first_emb))
            if _RFR_SIM_LO < sim < _RFR_SIM_HI:
                return NodeType.RFR
        return NodeType.HYP

    # Priority 6: anchors a contrast
    if EdgeType.CONT in out_types:
        return NodeType.CMP

    # Priority 7: provides supporting evidence
    if EdgeType.SUPP in out_types:
        return NodeType.JUS

    # Priority 8: boundary-class convergence without multi-target SYNT
    if tu.boundary_class == BoundaryClass.CONVERGENCE:
        return NodeType.SYN

    # First TU of a trace always opens a hypothesis
    if idx == 0:
        return NodeType.HYP

    # Embedding-based META detection: TU in the second half of the trace
    # with high similarity to TU₀ — returning to original framing without
    # a structural edge role signals metacognitive self-monitoring.
    if (
        embeddings is not None
        and first_emb is not None
        and n_tus >= 4
        and idx >= int(n_tus * _MET_POS_FRAC)
    ):
        sim_to_first = float(np.dot(embeddings[idx], first_emb))
        if sim_to_first >= _MET_SIM_THRESH:
            return NodeType.MET

    return NodeType.SPC


def load_deberta_classifier(model_path: str):
    """Placeholder for the fine-tuned DeBERTa-base node classifier.
    Raises NotImplementedError until silver-label training is complete."""
    raise NotImplementedError(
        "Fine-tuned DeBERTa classifier not yet available. "
        "Run the silver-label generation pipeline first."
    )
