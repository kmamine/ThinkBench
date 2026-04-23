# CLAUDE.md — ThinkBench Implementation Spec

## Project Overview

**ThinkBench** is a framework for profiling LLM thinking behavior on open-ended problems. It extracts **Thought Graphs** (directed graphs with cycles) from reasoning traces, computes a 22-dimensional cognitive profile vector per model, and characterizes thinking styles across domains.

**Target venue**: EMNLP 2026  
**Repo name**: `thinkbench`  
**License**: Apache 2.0

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Package manager | uv (preferred) or pip |
| LLM calls (collection only) | openai-compatible client (vLLM endpoint) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| NLI (edge promotion) | cross-encoder/nli-deberta-v3-large |
| Graph storage | networkx |
| Data storage | JSON (graphs) + Parquet (metrics) |
| Visualization | matplotlib + seaborn (radar charts, heatmaps) + pyvis (graph rendering) |
| Experiment tracking | wandb (optional) |
| Testing | pytest |
| CLI | typer |

---

## Directory Structure

```
thinkbench/
├── CLAUDE.md                    # This file
├── pyproject.toml
├── README.md
├── data/
│   ├── questions/
│   │   ├── ethical_dilemmas.jsonl
│   │   ├── policy_design.jsonl
│   │   ├── strategic_planning.jsonl
│   │   ├── scientific_speculation.jsonl
│   │   ├── creative_problem_solving.jsonl
│   │   └── interpersonal_reasoning.jsonl
│   ├── traces/                  # Raw CoT traces (collected)
│   ├── graphs/                  # Extracted thought graphs (JSON)
│   ├── profiles/                # Computed profiles (JSON)
│   └── validation/              # Human annotation gold sets
├── src/
│   └── thinkbench/
│       ├── __init__.py
│       ├── collect/             # CoT collection
│       │   ├── __init__.py
│       │   ├── collector.py     # Multi-model CoT collector
│       │   └── models.py        # LLM client (vLLM/OpenAI-compatible)
│       ├── extract/             # Thought graph extraction pipeline
│       │   ├── __init__.py
│       │   ├── segmenter.py     # 3-pass non-generative segmentation
│       │   ├── classifier.py    # Rule-based boundary_class → node_type
│       │   ├── linker.py        # Graph assembly from TUs + edges
│       │   └── schemas.py       # Pydantic models for TU, Edge, ThoughtGraph
│       ├── metrics/             # Metric computation
│       │   ├── __init__.py
│       │   ├── breadth.py       # BF, UPC, DS, FID
│       │   ├── depth.py         # MEC, MBD, SG, RD
│       │   ├── structure.py     # EER, BR, CBC, CI, GD, RvD
│       │   ├── metacognitive.py # SRR, CHR, HD, PT
│       │   ├── efficiency.py    # TPI, RR
│       │   └── profile.py       # Aggregate into profile vector
│       ├── utils/               # Shared utilities
│       │   ├── __init__.py
│       │   └── models.py        # Lazy singletons: get_embed_model(), get_nli_model()
│       ├── analysis/            # Profiling and visualization
│       │   ├── __init__.py
│       │   ├── clustering.py    # PCA + k-means archetype discovery
│       │   ├── sensitivity.py   # Domain sensitivity analysis
│       │   ├── stability.py     # Test-retest reliability
│       │   ├── radar.py         # Radar chart generation
│       │   └── thinkcard.py     # Model card generation
│       ├── validate/            # Parser validation
│       │   ├── __init__.py
│       │   ├── annotation.py    # Annotation loading + IAA
│       │   └── evaluate.py      # Boundary F1, node/edge accuracy
│       └── cli.py               # CLI entrypoint
├── scripts/
│   ├── collect_traces.py        # Batch collection runner
│   ├── extract_graphs.py        # Batch graph extraction (non-generative)
│   ├── compute_profiles.py      # Batch metric computation
│   ├── run_experiment.py        # Modular experiment runner
│   ├── thinkbench_full.py       # Standalone full pipeline
│   └── generate_report.py       # Visualization + report
├── notebooks/
│   ├── 01_eda_traces.ipynb      # Trace statistics
│   ├── 02_graph_visualization.ipynb
│   ├── 03_profile_analysis.ipynb
│   └── 04_paper_figures.ipynb
└── tests/
    ├── test_segmenter.py
    ├── test_classifier.py
    ├── test_linker.py
    ├── test_metrics.py
    └── fixtures/                # Sample traces + expected outputs
```

---

## Data Models (src/thinkbench/extract/schemas.py)

```python
from pydantic import BaseModel
from enum import Enum

class BoundaryClass(str, Enum):
    BACKTRACK    = "BACKTRACK"    # Explicit revision/correction cue
    BRANCH       = "BRANCH"       # New direction / alternative
    META         = "META"         # Reflection on own reasoning
    CONVERGENCE  = "CONVERGENCE"  # Synthesis / conclusion cue
    ELABORATION  = "ELABORATION"  # Deepening existing idea
    CONTRAST     = "CONTRAST"     # Opposing perspective
    SUPPORT      = "SUPPORT"      # Evidence / justification cue
    NONE         = "NONE"         # Soft boundary (TextTiling)

class NodeType(str, Enum):
    # Exploration family
    HYP = "HYP"   # Hypothesis
    RFR = "RFR"    # Reframing
    ANA = "ANA"    # Analogy
    BRS = "BRS"    # Brainstorm
    # Elaboration family
    JUS = "JUS"    # Justification
    SPC = "SPC"    # Specification
    IMP = "IMP"    # Implication
    CON = "CON"    # Constraint
    # Evaluation family
    CRT = "CRT"    # Critique
    CMP = "CMP"    # Comparison
    MET = "MET"    # Meta-reflection
    # Convergence family
    SYN = "SYN"    # Synthesis

class NodeFamily(str, Enum):
    EXPLORATION = "EXPLORATION"
    ELABORATION = "ELABORATION"
    EVALUATION  = "EVALUATION"
    CONVERGENCE = "CONVERGENCE"

BOUNDARY_NODE_MAP = {
    BoundaryClass.BACKTRACK:   NodeType.CRT,
    BoundaryClass.BRANCH:      NodeType.HYP,
    BoundaryClass.META:        NodeType.MET,
    BoundaryClass.CONVERGENCE: NodeType.SYN,
    BoundaryClass.ELABORATION: NodeType.SPC,
    BoundaryClass.CONTRAST:    NodeType.CMP,
    BoundaryClass.SUPPORT:     NodeType.JUS,
    BoundaryClass.NONE:        NodeType.SPC,  # overridden for idx==0 → HYP
}

class EdgeType(str, Enum):
    ELAB = "ELAB"  # Elaboration
    BRCH = "BRCH"  # Branching
    BACK = "BACK"  # Backtracking
    SYNT = "SYNT"  # Synthesis
    CRIT = "CRIT"  # Critique
    SUPP = "SUPP"  # Support
    CONT = "CONT"  # Contrast
    SEQ  = "SEQ"   # Sequential (default)

BOUNDARY_EDGE_MAP = {
    BoundaryClass.BACKTRACK:   EdgeType.BACK,
    BoundaryClass.BRANCH:      EdgeType.BRCH,
    BoundaryClass.META:        EdgeType.SEQ,
    BoundaryClass.CONVERGENCE: EdgeType.SYNT,
    BoundaryClass.ELABORATION: EdgeType.ELAB,
    BoundaryClass.CONTRAST:    EdgeType.CONT,
    BoundaryClass.SUPPORT:     EdgeType.SUPP,
    BoundaryClass.NONE:        EdgeType.SEQ,
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
    source: int       # tu_id
    target: int       # tu_id
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
    branching_factor: float
    unique_perspective_count: float
    domain_spread: float
    first_idea_diversity: float
    # Depth
    max_elaboration_chain: float
    mean_branch_depth: float
    specificity_gradient: float
    reasoning_density: float
    # Structure
    exploration_exploitation_ratio: float
    backtracking_rate: float
    cross_branch_connectivity: float
    convergence_index: float
    graph_density: float
    revision_depth: float
    # Metacognitive
    self_reflection_rate: float
    critique_to_hypothesis_ratio: float
    hedging_density: float
    perspective_taking: float
    # Efficiency
    token_per_idea: float
    redundancy_ratio: float
    # Summary
    avg_tokens: float
    avg_tus: float

    def to_vector(self) -> list[float]:
        """Return the 22-dim profile vector in canonical order."""
        return [
            self.branching_factor, self.unique_perspective_count,
            self.domain_spread, self.first_idea_diversity,
            self.max_elaboration_chain, self.mean_branch_depth,
            self.specificity_gradient, self.reasoning_density,
            self.exploration_exploitation_ratio, self.backtracking_rate,
            self.cross_branch_connectivity, self.convergence_index,
            self.graph_density, self.revision_depth,
            self.self_reflection_rate, self.critique_to_hypothesis_ratio,
            self.hedging_density, self.perspective_taking,
            self.token_per_idea, self.redundancy_ratio,
            self.avg_tokens, self.avg_tus,
        ]
```

---

## Implementation Order

Build in this exact order. Each step must work end-to-end before moving to the next.

### Phase 1: Foundation (Week 1)
1. **`schemas.py`** — All Pydantic models above. Write tests.
2. **`questions/`** — Create the 180-question dataset (30 per domain). Store as JSONL:
   ```json
   {"id": "D1_001", "domain": "ethical_dilemmas", "text": "A self-driving car's braking system...", "expected_angles": ["utilitarian", "deontological", "legal liability"], "difficulty": "medium"}
   ```
3. **`collector.py`** — Collect traces from any OpenAI-compatible endpoint. Must handle:
   - Rate limiting with exponential backoff
   - Model-specific CoT extraction (`<think>` tags vs. plain output)
   - Resumability (skip already-collected traces by trace_id)

### Phase 2: Graph Extraction (Week 2-3) — Non-generative
4. **`segmenter.py`** — 3-pass pipeline (no LLM calls):

   **Pass 1 — Hard boundaries (regex)**: Match cue-phrase patterns from a compiled lexicon to assign `BoundaryClass`. Paragraph breaks → SEQ. Merge spans < 3 sentences or 50 tokens with the following span.

   **Pass 2 — Soft boundaries (TextTiling)**: Encode sentences with `all-MiniLM-L6-v2`. Slide window of size 3, compute cosine similarity. Detect local minima via `scipy.signal.find_peaks` (prominence ≥ 0.15). Insert soft boundary if similarity < τ (30th percentile of within-trace similarities).

   **Pass 3 — NLI edge promotion (DeBERTa)**: For adjacent TU pairs within window=4 with SEQ edge, run `cross-encoder/nli-deberta-v3-large` NLI with SUPP/CONT/ELAB hypotheses (threshold 0.75). For BACKTRACK spans > 4 apart, run BACK hypothesis (threshold 0.70). All pairs batched in a single `predict()` call.

5. **`classifier.py`** — Rule-based `boundary_class → node_type` map (deterministic). Confidence = 1.0 for hard-boundary TUs, 0.5 for soft (NONE class). DeBERTa fine-tuned node classifier is a future TODO — `load_deberta_classifier()` raises `NotImplementedError`.

6. **`linker.py`** — Graph assembly only. Takes `(trace_dict, tus, edges)` produced by the segmenter and assembles the `ThoughtGraph` Pydantic object.

### Phase 3: Metrics (Week 3-4)
7. **`breadth.py`** through **`efficiency.py`** — All metrics operate on a `ThoughtGraph` object:
   - `branching_factor`: `|E_BRCH| / max(|V|, 1)`
   - `unique_perspective_count`: count of RFR nodes
   - `domain_spread`: agglomerative clustering (cosine threshold=0.45) of BRS+HYP embeddings
   - `reasoning_density`: `|nodes with ≥1 semantic edge| / |V|`
   - `backtracking_rate`: `|E_BACK| / max(|E_sem|, 1)`
   - `graph_density`: semantic edges only: `|E_sem| / (|V|×(|V|-1))`
   - `revision_depth`: `mean(|pos(u) - pos(v)|) for BACK edges`
   - `convergence_index`: `sum(d_in(v) for SYN nodes) / (|V| × mean_d_in)`
   - `cross_branch_connectivity`: fraction of cross-branch node pairs with SYNT/SUPP edge
   - `redundancy_ratio`: pairwise cosine similarity > 0.90
   - `hedging_density`: regex hedge-word detection

8. **`profile.py`** — `compute_profile(graph) → CognitiveProfile` and `aggregate_profiles(profiles, trace_model_map) → list[dict]`.

### Phase 4: Analysis (Week 4-5)
9. **`clustering.py`** — PCA (z-score normalized) + k-means (k=2..6, silhouette selection) → archetype labels
10. **`sensitivity.py`** — std across 6 domains per metric → sensitivity score
11. **`stability.py`** — ICC(3,1) per metric across K=3 runs; flag ICC < 0.6 as unreliable
12. **`radar.py`** + **`thinkcard.py`** — min-max normalized radar charts + ThinkCard

### Phase 5: Validation (Week 5-6)
13. **`validate/`** — Boundary F1, node macro-F1, edge accuracy; Cohen's κ IAA
14. **Paper figures** — `scripts/generate_figures.py`

---

## Key Design Decisions

### Why non-generative extraction (v2)?
v1 used a generative LLM (Claude Sonnet / GPT-4.1) to segment, classify, and link reasoning traces. This introduced circular validity — analyzing a model's output with another model introduces bias. v2 replaces all extraction steps with deterministic rules (regex cue phrases), discriminative models (DeBERTa NLI cross-encoder), and embedding similarity (MiniLM). No generative LLM is involved between trace collection and profile output.

### Why directed graphs, not trees or DAGs?
Trees (LCoT2Tree) cannot represent cross-branch synthesis. DAGs forbid cycles — but cycles are real patterns: iterative refinement (propose → critique → revise → re-evaluate), oscillation between perspectives, circular justification. The full directed graph preserves all of this. Depth metrics are computed on the ELAB-only subgraph, which is naturally acyclic.

### Why open-ended questions?
On closed problems, structural analysis is secondary to correctness. Open-ended questions remove this confound entirely — the structure IS the evaluation.

### Why profiles not rankings?
A single-number ranking collapses the distinction between "Divergent Explorer" (good for brainstorming) and "Deep Deliberator" (good for analysis). The 22-dim profile captures this; a rank does not.

### Why 12 node types?
- Fewer (e.g., LCoT2Tree's ~5) collapses important distinctions (hypothesis vs. analogy vs. reframing)
- More causes annotation noise (classifier accuracy drops below useful thresholds)
- 12 types in 4 families gives useful granularity for both fine-grained and coarse analysis

---

## Testing Strategy

- **Unit tests**: Each metric function tested against hand-constructed graphs with known expected values
- **Integration tests**: End-to-end pipeline on 5 sample traces — check graph structure is valid (all nodes reachable, no self-loops, ELAB-only subgraph is acyclic)
- **Regression tests**: Golden outputs for 3 traces — compare against expected graphs when pipeline changes
- **Fixtures**: `tests/fixtures/` contains sample CoT traces with expected segmentation and graph output

---

## CLI Commands

```bash
# Collect traces
thinkbench collect --models deepseek-r1,qwq-32b --domains all --runs 3

# Extract thought graphs (non-generative — no parser LLM needed)
thinkbench extract --input data/traces/ --output data/graphs/

# Compute profiles
thinkbench profile --input data/graphs/ --output data/profiles/

# Generate analysis
thinkbench analyze --input data/profiles/ --output results/

# Validate parser
thinkbench validate --gold data/validation/ --predictions data/graphs/

# Generate paper figures
thinkbench figures --input results/ --output paper/figures/
```

---

## Critical Implementation Notes

1. **Idempotency**: Every script must be resumable. Use trace_id as the primary key. Skip already-processed traces.

2. **Parallel processing**: Use asyncio + semaphore for LLM collection calls (concurrency=5). Extraction is synchronous (CPU/GPU-bound NLI inference).

3. **Reproducibility**: Fix random seeds. Log all collection parameters (model, temperature). Pin all dependency versions in pyproject.toml.

4. **Shared model singletons**: Embedding and NLI models are loaded lazily via `utils/models.py`. Both `get_embed_model()` and `get_nli_model()` return the same instance across all callers — do not instantiate separately in individual metric functions.

5. **Batched NLI**: Pass all (premise, hypothesis) pairs for a trace as a single list to `nli_model.predict()`. Never call predict in a loop one pair at a time.

6. **Graph validation**: After building a thought graph, validate:
   - No self-loops
   - All edge sources/targets reference valid node IDs
   - The ELAB-only subgraph is acyclic
   - Every node is reachable from node 0 via some path
   - At least one EXPLORATION node exists

7. **Embedding model**: `all-MiniLM-L6-v2` for speed during development; switch to `gte-large-en-v1.5` or `nomic-embed-text-v1.5` for final experiments.
