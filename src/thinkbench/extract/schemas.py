from pydantic import BaseModel
from enum import Enum


class NodeType(str, Enum):
    HYP = "HYP"
    RFR = "RFR"
    ANA = "ANA"
    BRS = "BRS"
    JUS = "JUS"
    SPC = "SPC"
    IMP = "IMP"
    CON = "CON"
    CRT = "CRT"
    CMP = "CMP"
    MET = "MET"
    SYN = "SYN"


class NodeFamily(str, Enum):
    EXPLORATION = "EXPLORATION"
    ELABORATION = "ELABORATION"
    EVALUATION = "EVALUATION"
    CONVERGENCE = "CONVERGENCE"


NODE_FAMILY_MAP = {
    NodeType.HYP: NodeFamily.EXPLORATION,
    NodeType.RFR: NodeFamily.EXPLORATION,
    NodeType.ANA: NodeFamily.EXPLORATION,
    NodeType.BRS: NodeFamily.EXPLORATION,
    NodeType.JUS: NodeFamily.ELABORATION,
    NodeType.SPC: NodeFamily.ELABORATION,
    NodeType.IMP: NodeFamily.ELABORATION,
    NodeType.CON: NodeFamily.ELABORATION,
    NodeType.CRT: NodeFamily.EVALUATION,
    NodeType.CMP: NodeFamily.EVALUATION,
    NodeType.MET: NodeFamily.EVALUATION,
    NodeType.SYN: NodeFamily.CONVERGENCE,
}


class EdgeType(str, Enum):
    ELAB = "ELAB"
    BRCH = "BRCH"
    BACK = "BACK"
    SYNT = "SYNT"
    CRIT = "CRIT"
    SUPP = "SUPP"
    CONT = "CONT"
    SEQ = "SEQ"


class BoundaryClass(str, Enum):
    BACKTRACK = "BACKTRACK"
    BRANCH = "BRANCH"
    META = "META"
    CONVERGENCE = "CONVERGENCE"
    ELABORATION = "ELABORATION"
    CONTRAST = "CONTRAST"
    SUPPORT = "SUPPORT"
    NONE = "NONE"  # soft boundary (TextTiling) or paragraph break


# Default edge type emitted at each boundary class
BOUNDARY_EDGE_MAP = {
    BoundaryClass.BACKTRACK: EdgeType.BACK,
    BoundaryClass.BRANCH: EdgeType.BRCH,
    BoundaryClass.META: EdgeType.SEQ,
    BoundaryClass.CONVERGENCE: EdgeType.SYNT,
    BoundaryClass.ELABORATION: EdgeType.ELAB,
    BoundaryClass.CONTRAST: EdgeType.CONT,
    BoundaryClass.SUPPORT: EdgeType.SUPP,
    BoundaryClass.NONE: EdgeType.SEQ,
}

# Deterministic node type assigned from boundary class (rule-based fallback)
BOUNDARY_NODE_MAP = {
    BoundaryClass.BACKTRACK: NodeType.CRT,
    BoundaryClass.BRANCH: NodeType.HYP,
    BoundaryClass.META: NodeType.MET,
    BoundaryClass.CONVERGENCE: NodeType.SYN,
    BoundaryClass.ELABORATION: NodeType.SPC,
    BoundaryClass.CONTRAST: NodeType.CMP,
    BoundaryClass.SUPPORT: NodeType.JUS,
    BoundaryClass.NONE: NodeType.SPC,  # overridden to HYP for first TU
}


class ThoughtUnit(BaseModel):
    tu_id: int
    text: str
    start_char: int
    end_char: int
    token_count: int = 0
    boundary_class: BoundaryClass = BoundaryClass.NONE
    node_type: NodeType | None = None
    node_family: NodeFamily | None = None
    classification_confidence: float | None = None


class Edge(BaseModel):
    source: int
    target: int
    edge_type: EdgeType
    confidence: float
    is_sequential: bool = False


class ThoughtGraph(BaseModel):
    trace_id: str
    model: str
    question_id: str
    domain: str
    run: int
    nodes: list[ThoughtUnit]
    edges: list[Edge]
    token_count: int


class CognitiveProfile(BaseModel):
    model: str
    domain: str | None = None
    trace_id: str | None = None
    # Breadth
    branching_factor: float = 0.0
    unique_perspective_count: float = 0.0
    domain_spread: float = 0.0
    first_idea_diversity: float = 0.0
    # Depth
    max_elaboration_chain: float = 0.0
    mean_branch_depth: float = 0.0
    specificity_gradient: float = 0.0
    reasoning_density: float = 0.0
    # Structure
    exploration_exploitation_ratio: float = 0.0
    backtracking_rate: float = 0.0
    cross_branch_connectivity: float = 0.0
    convergence_index: float = 0.0
    graph_density: float = 0.0
    revision_depth: float = 0.0
    # Metacognitive
    self_reflection_rate: float = 0.0
    critique_to_hypothesis_ratio: float = 0.0
    hedging_density: float = 0.0
    perspective_taking: float = 0.0
    # Efficiency
    token_per_idea: float = 0.0
    redundancy_ratio: float = 0.0
    # Summary
    avg_tokens: float = 0.0
    avg_tus: float = 0.0

    def to_vector(self) -> list[float]:
        return [
            self.branching_factor,
            self.unique_perspective_count,
            self.domain_spread,
            self.first_idea_diversity,
            self.max_elaboration_chain,
            self.mean_branch_depth,
            self.specificity_gradient,
            self.reasoning_density,
            self.exploration_exploitation_ratio,
            self.backtracking_rate,
            self.cross_branch_connectivity,
            self.convergence_index,
            self.graph_density,
            self.revision_depth,
            self.self_reflection_rate,
            self.critique_to_hypothesis_ratio,
            self.hedging_density,
            self.perspective_taking,
            self.token_per_idea,
            self.redundancy_ratio,
            self.avg_tokens,
            self.avg_tus,
        ]
