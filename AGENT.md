# AGENT.md — ThinkBench Agent Spec

## Project Overview

**ThinkBench** is a framework for profiling LLM thinking behavior on open-ended problems. It extracts **Thought Graphs** (directed graphs with cycles) from reasoning traces, computes a 22-dimensional cognitive profile vector per model, and characterizes thinking styles across domains.

**Target venue**: EMNLP 2026  
**Repo name**: `thinkbench`  
**License**: Apache 2.0

---

## Pipeline Summary

The pipeline has three phases. Only Phase 1 (collection) uses an LLM. Phases 2 and 3 are fully non-generative.

```
Questions (.jsonl)
    │
    ▼
Phase 1: Collection  ──── LLM endpoint (vLLM/OpenAI-compatible)
    │  traces_*.jsonl
    ▼
Phase 2: Extraction  ──── Non-generative: regex + MiniLM + DeBERTa NLI
    │  graphs/*.json
    ▼
Phase 3: Profiling   ──── 22 cognitive metrics → CognitiveProfile
    │  profiles/results.json
    ▼
Report / Radar Chart
```

---

## Extraction Pipeline (v2 — Non-generative)

Segmentation uses a 3-pass approach with no generative LLM calls:

**Pass 1 — Hard boundaries (regex)**: Cue phrases mapped to `BoundaryClass` (BACKTRACK, BRANCH, META, CONVERGENCE, ELABORATION, CONTRAST, SUPPORT). Paragraph breaks → SEQ. Short spans merged with following span.

**Pass 2 — Soft boundaries (TextTiling)**: Sentences encoded with `all-MiniLM-L6-v2`. Local minima in cosine similarity (prominence ≥ 0.15, threshold = 30th percentile) become soft boundaries (`BoundaryClass.NONE`).

**Pass 3 — NLI edge promotion**: `cross-encoder/nli-deberta-v3-large` promotes SEQ edges to typed semantic edges (SUPP/CONT/ELAB at threshold 0.75; BACK at 0.70). All pairs batched in one `predict()` call.

Node type assignment is deterministic via `BOUNDARY_NODE_MAP`:

| BoundaryClass | NodeType | EdgeType |
|---|---|---|
| BACKTRACK | CRT | BACK |
| BRANCH | HYP | BRCH |
| META | MET | SEQ |
| CONVERGENCE | SYN | SYNT |
| ELABORATION | SPC | ELAB |
| CONTRAST | CMP | CONT |
| SUPPORT | JUS | SUPP |
| NONE (soft, idx=0) | HYP | SEQ |
| NONE (soft, idx>0) | SPC | SEQ |

---

## Key Schemas

### ThoughtUnit
```python
ThoughtUnit(
    tu_id=0,
    text="...",
    start_char=0, end_char=147,
    token_count=32,
    boundary_class=BoundaryClass.BRANCH,
    node_type=NodeType.HYP,
    node_family=NodeFamily.EXPLORATION,
    classification_confidence=1.0,
)
```

### Edge
```python
Edge(source=0, target=1, edge_type=EdgeType.ELAB, confidence=0.92, is_sequential=False)
```

### CognitiveProfile (22 metrics + 2 summary)
```
Breadth:      branching_factor, unique_perspective_count, domain_spread, first_idea_diversity
Depth:        max_elaboration_chain, mean_branch_depth, specificity_gradient, reasoning_density
Structure:    exploration_exploitation_ratio, backtracking_rate, cross_branch_connectivity,
              convergence_index, graph_density, revision_depth
Metacognitive: self_reflection_rate, critique_to_hypothesis_ratio, hedging_density, perspective_taking
Efficiency:   token_per_idea, redundancy_ratio
Summary:      avg_tokens, avg_tus
```

---

## Metric Formulas (v2)

| Metric | Formula |
|---|---|
| `branching_factor` | `\|E_BRCH\| / max(\|V\|, 1)` |
| `unique_perspective_count` | count of RFR nodes |
| `domain_spread` | agglomerative clusters (cosine threshold=0.45) on BRS+HYP embeddings |
| `reasoning_density` | `\|nodes with ≥1 semantic edge\| / \|V\|` |
| `backtracking_rate` | `\|E_BACK\| / max(\|E_sem\|, 1)` |
| `graph_density` | `\|E_sem\| / (\|V\|×(\|V\|-1))` (semantic edges only) |
| `revision_depth` | `mean(\|pos(u) - pos(v)\|) for BACK edges` |
| `convergence_index` | `sum(d_in(v) for SYN nodes) / (\|V\| × mean_d_in)` |
| `cross_branch_connectivity` | fraction of cross-branch pairs with SYNT/SUPP edge |
| `token_per_idea` | `avg_tokens / max(unique_perspective_count, 1)` |

---

## Shared Utilities

`src/thinkbench/utils/models.py` provides lazy singletons:
- `get_embed_model()` → `SentenceTransformer("all-MiniLM-L6-v2")`
- `get_nli_model()` → `CrossEncoder("cross-encoder/nli-deberta-v3-large", num_labels=3)`

Never instantiate these directly in metric functions — always use the getters.

---

## Running the Pipeline

```bash
# Full pipeline
python scripts/thinkbench_full.py \
    --questions data/questions/ethical_dilemmas.jsonl \
    --runs 3 --output data

# Modular (collection only)
python scripts/run_experiment.py \
    --questions data/questions/ethical_dilemmas.jsonl \
    --runs 3 --collect

# Extraction only (non-generative, no LLM needed)
python scripts/extract_graphs.py \
    --input data/traces/ --output data/graphs/

# Profile computation
python scripts/run_experiment.py --compute
```

Environment variables for collection:
```
VLLM_API_KEY    # auth token
VLLM_ENDPOINT   # e.g. http://10.17.1.57:8978
VLLM_MODEL      # e.g. Qwen/Qwen3.5-35B-A3B
```
