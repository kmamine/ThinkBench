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
| LLM calls | litellm (unified API for all providers) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2 for speed, or gte-large-en-v1.5 for quality) |
| Graph storage | networkx |
| Data storage | SQLite (traces) + JSON (graphs) + Parquet (metrics) |
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
│   │   ├── ethical_dilemmas.json
│   │   ├── policy_design.json
│   │   ├── strategic_planning.json
│   │   ├── scientific_speculation.json
│   │   ├── creative_problem_solving.json
│   │   └── interpersonal_reasoning.json
│   ├── traces/                  # Raw CoT traces (collected)
│   ├── graphs/                  # Extracted thought graphs (JSON)
│   ├── profiles/                # Computed profiles (Parquet)
│   └── validation/              # Human annotation gold sets
├── src/
│   └── thinkbench/
│       ├── __init__.py
│       ├── collect/             # CoT collection
│       │   ├── __init__.py
│       │   ├── collector.py     # Multi-model CoT collector
│       │   └── models.py        # Model configs and API wrappers
│       ├── extract/             # Thought graph extraction pipeline
│       │   ├── __init__.py
│       │   ├── segmenter.py     # Stage A: CoT → thought units
│       │   ├── classifier.py    # Stage B: TU → node types
│       │   ├── linker.py        # Stage C: edge detection + graph assembly
│       │   └── schemas.py       # Pydantic models for TU, Edge, ThoughtGraph
│       ├── metrics/             # Metric computation
│       │   ├── __init__.py
│       │   ├── breadth.py       # BF, UPC, DS, FID
│       │   ├── depth.py         # MEC, MBD, SG, RD
│       │   ├── structure.py     # EER, BR, CBC, CI, OR, GD, CC, MCL
│       │   ├── metacognitive.py # SRR, CHR, HD, PT
│       │   ├── efficiency.py    # TPI, RR
│       │   └── profile.py       # Aggregate into profile vector
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
│   ├── extract_graphs.py        # Batch graph extraction
│   ├── compute_profiles.py      # Batch metric computation
│   └── generate_figures.py      # Paper figure generation
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
    ELAB = "ELAB"  # Elaboration
    BRCH = "BRCH"  # Branching
    BACK = "BACK"  # Backtracking
    SYNT = "SYNT"  # Synthesis
    CRIT = "CRIT"  # Critique
    SUPP = "SUPP"  # Support
    CONT = "CONT"  # Contrast
    SEQ  = "SEQ"   # Sequential (default)

class ThoughtUnit(BaseModel):
    tu_id: int
    text: str
    start_char: int
    end_char: int
    node_type: NodeType | None = None
    node_family: NodeFamily | None = None
    classification_confidence: float | None = None

class Edge(BaseModel):
    source: int  # tu_id
    target: int  # tu_id
    edge_type: EdgeType
    confidence: float

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
    domain: str | None = None  # None = global profile
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
    orphan_ratio: float
    graph_density: float
    # Cycles
    cycle_count: float
    mean_cycle_length: float
    # Metacognitive
    self_reflection_rate: float
    critique_to_hypothesis_ratio: float
    hedging_density: float
    perspective_taking: float
    # Efficiency
    token_per_idea: float
    redundancy_ratio: float

    def to_vector(self) -> list[float]:
        """Return the 22-dim profile vector in canonical order."""
        return [
            self.branching_factor, self.unique_perspective_count,
            self.domain_spread, self.first_idea_diversity,
            self.max_elaboration_chain, self.mean_branch_depth,
            self.specificity_gradient, self.reasoning_density,
            self.exploration_exploitation_ratio, self.backtracking_rate,
            self.cross_branch_connectivity, self.convergence_index,
            self.orphan_ratio, self.graph_density,
            self.cycle_count, self.mean_cycle_length,
            self.self_reflection_rate, self.critique_to_hypothesis_ratio,
            self.hedging_density, self.perspective_taking,
            self.token_per_idea, self.redundancy_ratio,
        ]
```

---

## Implementation Order

Build in this exact order. Each step must work end-to-end before moving to the next.

### Phase 1: Foundation (Week 1)
1. **`schemas.py`** — All Pydantic models above. Write tests.
2. **`questions/`** — Create the 180-question dataset (30 per domain). Store as JSON:
   ```json
   [
     {
       "id": "D1_001",
       "domain": "ethical_dilemmas",
       "text": "A self-driving car's braking system...",
       "expected_angles": ["utilitarian", "deontological", "legal liability"],
       "difficulty": "medium"
     }
   ]
   ```
3. **`collector.py`** — Use litellm to collect traces. Must handle:
   - Rate limiting with exponential backoff
   - Model-specific CoT extraction (some models have `<think>` tags, others don't)
   - Progress tracking and resumability (skip already-collected traces)
   - Cost tracking per model

### Phase 2: Graph Extraction (Week 2-3)
4. **`segmenter.py`** — Implement the segmentation prompt. Key:
   - Use structured output (JSON mode) from the parser LLM
   - Validate that TUs are non-overlapping and cover the full trace
   - Cache results (segmentation is the most expensive step)
   - Batch processing with async calls

5. **`classifier.py`** — Implement node type classification. Key:
   - Classify with context (include previous TU for disambiguation)
   - Return confidence scores
   - Batch classify all TUs in a trace in a single call when possible
   - Validate against the 12-type taxonomy (reject invalid types)

6. **`linker.py`** — Implement the 4-pass linking algorithm:
   ```python
   def build_graph(nodes: list[ThoughtUnit]) -> ThoughtGraph:
       edges = []
       # Pass 1: Sequential backbone
       for i in range(len(nodes) - 1):
           edges.append(Edge(source=i, target=i+1, edge_type=EdgeType.SEQ, confidence=1.0))

       # Pass 2: Local semantic linking (window=5)
       for i in range(1, len(nodes)):
           for j in range(max(0, i-5), i):
               rel = classify_edge(nodes[j], nodes[i])
               if rel and rel.confidence >= 0.7:
                   edges.append(rel)
                   # Remove SEQ between j and i if j == i-1
                   if j == i - 1:
                       edges = [e for e in edges if not (e.source == j and e.target == i and e.edge_type == EdgeType.SEQ)]

       # Pass 3: Long-range synthesis
       syn_nodes = [n for n in nodes if n.node_type == NodeType.SYN]
       for syn in syn_nodes:
           candidates = find_synthesis_sources(syn, nodes)
           for src in candidates:
               edges.append(Edge(source=src.tu_id, target=syn.tu_id, edge_type=EdgeType.SYNT, confidence=candidates[src]))

       # Pass 4: Backtracking detection
       back_nodes = [n for n in nodes if n.node_type in (NodeType.CRT, NodeType.MET) or has_backtrack_markers(n.text)]
       for bn in back_nodes:
           target = find_backtrack_target(bn, nodes)
           if target:
               edges.append(Edge(source=bn.tu_id, target=target.tu_id, edge_type=EdgeType.BACK, confidence=target.confidence))

       return ThoughtGraph(nodes=nodes, edges=edges, ...)
   ```

### Phase 3: Metrics (Week 3-4)
7. **`breadth.py`** through **`efficiency.py`** — Implement each metric family.
   All metrics operate on a `ThoughtGraph` object and return floats.
   
   Key implementation notes:
   - `unique_perspective_count`: Build subgraph of EXPLORATION nodes, remove SEQ edges, count connected components
   - `max_elaboration_chain`: BFS/DFS on ELAB-only subgraph (naturally acyclic), find longest path
   - `cross_branch_connectivity`: Requires identifying "branches" first (subtrees rooted at EXPLORATION nodes), then counting SYNT edges between different branches
   - `specificity_gradient`: Use spaCy NER to count entities per token per node, correlate with depth in ELAB subgraph
   - `redundancy_ratio`: Embed all TUs, compute pairwise cosine similarity matrix, count pairs > 0.90
   - `hedging_density`: Regex pattern matching for hedge words + uncertainty markers
   - `cycle_count`: Use `networkx.simple_cycles()` on the full graph
   - `mean_cycle_length`: Mean of `len(c) for c in nx.simple_cycles(G)` — cap search at 10k cycles to avoid combinatorial explosion on dense graphs

8. **`profile.py`** — Aggregate metrics into `CognitiveProfile` objects. Compute per-trace profiles, then aggregate to per-model-per-domain and global profiles.

### Phase 4: Analysis (Week 4-5)
9. **`clustering.py`** — PCA + k-means on profile vectors.
   - Standardize profiles (z-score) before PCA
   - Select k via silhouette score (test k=2..6)
   - Assign archetype labels based on cluster centroids

10. **`sensitivity.py`** — Compute domain sensitivity per model.
    - For each metric, compute std across 6 domains
    - Aggregate into a single sensitivity score

11. **`stability.py`** — Test-retest reliability using K=3 runs.
    - ICC(3,1) for each metric
    - Flag metrics with ICC < 0.6 as unreliable

12. **`radar.py`** + **`thinkcard.py`** — Visualization.
    - Radar charts: normalize all metrics to [0,1] using min-max across models
    - ThinkCard: radar chart + archetype label + top-3 distinctive metrics + example trace excerpt

### Phase 5: Validation (Week 5-6)
13. **`validate/`** — Build annotation toolkit and compute validation metrics.
    - Export annotation tasks as CSV files
    - Compute boundary F1, node macro-F1, edge accuracy
    - Cohen's κ for inter-annotator agreement

14. **Paper figures** — Generate all figures for the paper via `scripts/generate_figures.py`.

---

## LLM Usage Budget

| Step | LLM Used | Est. Tokens/Trace | Total Est. |
|---|---|---|---|
| CoT Collection | 8 target models | ~2000 output | 8.6M output tokens |
| Segmentation | Claude Sonnet / GPT-4.1 | ~4000 in + ~2000 out | 26M total |
| Classification | Same | ~1500 in + ~200 out | 7.3M total |
| Local linking | Same | ~3000 in + ~500 out (per trace) | 15M total |
| Long-range linking | Same | ~2000 in + ~200 out (per SYN node) | ~5M total |
| **Total parser cost** | | | **~55M tokens** |

**Estimated cost**: ~$50-80 with Claude Sonnet / GPT-4.1-mini pricing. Well within $150 budget.

---

## Key Design Decisions

### Why directed graphs, not trees or DAGs?
Trees (LCoT2Tree) cannot represent cross-branch synthesis — the single most interesting feature of sophisticated reasoning. When a model says "Combining the economic argument from earlier with the fairness concern I raised...", that's a node with two parents from different branches. Trees force you to attach it to one. Beyond that, DAGs still forbid cycles — but cycles are real reasoning patterns. Iterative refinement (propose → critique → revise → re-evaluate) forms natural loops. Oscillation between perspectives ("on one hand... but then again...") is cyclical. Even circular justification is a pattern worth detecting as a reasoning flaw. The full directed graph preserves all of this. Depth metrics are computed on the ELAB-only subgraph, which is naturally acyclic.

### Why open-ended questions?
On closed problems, structural analysis is always secondary to correctness. The most interesting structural pattern in the world doesn't matter if the answer is wrong. Open-ended questions remove this confound entirely — the structure IS the evaluation.

### Why profiles not rankings?
A single-number ranking would be misleading. The whole point is that different models think differently, and those differences are *useful* depending on the task. A model that's a "Divergent Explorer" is better for brainstorming; a "Deep Deliberator" is better for analysis. The profile communicates this; a rank does not.

### Why 12 node types and not fewer/more?
- Fewer (e.g., LCoT2Tree's ~5) collapses important distinctions (hypothesis vs. analogy vs. reframing all become "exploration")
- More causes annotation noise (classifier accuracy drops below useful thresholds)
- 12 types grouped into 4 families gives a useful granularity for both fine-grained and coarse analysis

---

## Prompts Registry

All LLM prompts used in the pipeline live in `src/thinkbench/extract/prompts/`:
- `segment.txt` — Segmentation prompt
- `classify_node.txt` — Node type classification
- `classify_edge.txt` — Edge relationship detection
- `synthesis_source.txt` — Long-range synthesis source verification
- `backtrack_target.txt` — Backtracking target identification

Store as text files, not hardcoded strings. Version control them. Log prompt versions with results.

---

## Testing Strategy

- **Unit tests**: Each metric function tested against hand-constructed graphs with known expected values
- **Integration tests**: End-to-end pipeline on 5 sample traces, check graph structure is valid (all nodes reachable, edge types match node types, ELAB-only subgraph is acyclic)
- **Regression tests**: Golden outputs for 3 traces — if pipeline changes, compare against expected graphs
- **Fixtures**: `tests/fixtures/` contains 5 real CoT traces (one per model) with expected segmentation, classification, and graph output

---

## CLI Commands

```bash
# Collect traces
thinkbench collect --models deepseek-r1,qwq-32b --domains all --runs 3

# Extract thought graphs
thinkbench extract --input data/traces/ --output data/graphs/ --parser claude-sonnet

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

1. **Idempotency**: Every script must be resumable. Use trace_id as the primary key. Skip already-processed traces. Write outputs atomically (write to .tmp, then rename).

2. **Cost control**: Add a `--dry-run` flag to every script that shows how many API calls will be made and estimated cost before executing.

3. **Caching**: Cache all LLM responses with a content hash of the input. Use diskcache or sqlite. The segmentation step is the most expensive — never re-segment a trace you've already processed.

4. **Parallel processing**: Use asyncio + semaphore for LLM calls. Default concurrency: 5 for collection, 10 for parsing (cheaper calls).

5. **Reproducibility**: Fix random seeds. Log all LLM call parameters (model, temperature, prompt version). Pin all dependency versions in pyproject.toml.

6. **Graph validation**: After building a thought graph, validate:
   - No self-loops
   - All edge sources/targets reference valid node IDs
   - The ELAB-only subgraph is acyclic (elaboration does not loop — if it does, the edge was misclassified)
   - Cycles in the full graph are permitted and expected (iterative refinement, oscillation)
   - Every node is reachable from node 0 via some path
   - At least one EXPLORATION node exists
   - Node types and edge types are consistent (e.g., SYNT edges target SYN nodes)

7. **Embedding model**: Use the same embedding model throughout (for consistency). Recommend `all-MiniLM-L6-v2` for speed during development, switch to `gte-large-en-v1.5` or `nomic-embed-text-v1.5` for final experiments.

8. **Parser LLM choice**: Use a strong non-reasoning model for parsing (Claude Sonnet, GPT-4.1). Do NOT use a reasoning model as the parser — you don't want the parser's own thinking patterns to influence the segmentation. This is also cheaper.
