# ThinkBench v2 — Full Pipeline Specification

**Version:** 2.0  
**Target venue:** EMNLP 2025  
**Scope:** Characterizing reasoning patterns of native reasoning LLMs on open-ended problems via cognitive graph analysis

---

## 1. Overview

ThinkBench is a framework for profiling the *structural reasoning behavior* of large language models that produce explicit chain-of-thought traces via `<think>` blocks. Rather than measuring whether a model answers correctly — which is undefined for genuinely open-ended questions — ThinkBench characterizes **how** a model reasons: its breadth of exploration, depth of elaboration, metacognitive activity, structural coherence, and efficiency of thought.

The pipeline takes raw `<think>` traces as input and produces a **22-dimensional cognitive profile vector** per model, derived from a **Thought Graph** — a directed graph where nodes are atomic reasoning units and edges are typed semantic relations between them.

### 1.1 Design Constraints

The pipeline is built around four hard constraints motivated by reproducibility and methodological rigor:

1. **No generative extraction.** Graph construction (segmentation, edge typing) uses only deterministic rule-based components and discriminative models. No LLM is called during extraction. This eliminates the circular validity problem of using a generative model to analyze its own or peer outputs.

2. **`<think>`-native.** The pipeline is designed specifically for models that produce internal monologue traces (DeepSeek-R1, QwQ-32B, Qwen3 series, GLM-4, Arcee Trinity, MiniMax-01). These models produce a qualitatively distinct discourse register — self-directed, exploratory, with explicit revision markers — that enables robust rule-based segmentation.

3. **Open-ended only.** All questions in the benchmark are designed to have no fixed correct answer. This forces evaluation of reasoning structure rather than outcome, and prevents the confound of optimizing token generation to reach a known target.

4. **Model-agnostic metrics.** All 22 metrics are computed from graph topology and surface linguistic features. They do not depend on semantic similarity to a reference answer.

---

## 2. Dataset

### 2.1 Question Design

The benchmark consists of **200 questions** spanning 5 domains, 40 questions each:

| Domain | Description | Example |
|--------|-------------|---------|
| **Ethical Dilemmas** | Trolley-problem variants, resource allocation, conflicting obligations | "A hospital has one ICU bed and two patients of equal medical need..." |
| **Scientific Hypotheticals** | Causal reasoning under counterfactual conditions | "If the speed of light were 10% slower, what would change in physics and cosmology?" |
| **Strategic Planning** | Multi-stakeholder optimization under uncertainty | "A small nation with limited resources faces simultaneous climate and economic crises..." |
| **Philosophical Thought Experiments** | Identity, free will, consciousness, epistemology | "If a ship has every plank replaced over time, is it the same ship?" |
| **Creative Problem Solving** | Design and invention under constraints | "Design a public transit system for a city where 60% of streets are pedestrian-only" |

Questions are designed to satisfy three criteria: (a) no objectively correct answer exists, (b) multiple coherent solution frameworks are available, (c) the question is rich enough to support at least 500 tokens of genuine reasoning.

### 2.2 Model Roster

All models are native reasoning models with `<think>` block output:

- DeepSeek-R1 (671B)
- QwQ-32B
- Qwen3-235B-A22B
- Qwen3-32B
- Qwen3.5-35B-A3B
- GLM-4-Plus
- Arcee-Trinity
- MiniMax-01

Each model is queried with **K=3 independent runs** per question at temperature 0.7, yielding a maximum of **4,800 traces** (200 × 3 × 8 models). Traces with fewer than 100 tokens in the `<think>` block are discarded.

### 2.3 Collection Protocol

The `<think>` block is extracted exactly as produced by the model, with no post-processing except stripping the XML tags themselves. The visible answer (`<answer>` or post-`</think>` text) is stored separately but **not used in any metric computation**. Temperature is fixed at 0.7 for all runs; top-p at 0.95.

System prompt:
```
Think through this problem carefully and thoroughly. Explore multiple 
perspectives, consider edge cases, and reason step by step. Do not 
rush to a conclusion.
```

No additional instructions about format, length, or structure are given, to avoid priming specific graph topologies.

---

## 3. Segmentation: Three-Pass Non-Generative Pipeline

This is the most critical and most redesigned component relative to v1. The goal is to decompose a raw `<think>` trace into **Thought Units (TUs)** — the atomic nodes of the Thought Graph — without using a generative model.

A Thought Unit is formally defined as: *the minimal contiguous span of text that expresses a single reasoning move, where a reasoning move is an action from the set {hypothesize, justify, elaborate, constrain, critique, compare, reframe, synthesize, reflect}.*

### 3.1 Pass 1: Hard Boundary Detection (Rule-Based, Deterministic)

Pass 1 identifies **hard boundaries** — positions where a reasoning move unambiguously ends and a new one begins, signaled by explicit discourse markers. These markers are derived from empirical analysis of `<think>` output across QwQ and DeepSeek-R1 traces, grouped by the reasoning move transition they signal.

The cue-phrase lexicon is organized into **7 boundary classes**, each of which carries a default edge type that will be assigned to the transition without further analysis:

#### Boundary Class → Default Edge Type

| Class | Default Edge | Example Cues |
|-------|-------------|--------------|
| `BACKTRACK` | `BACK` | *"Wait,"*, *"Actually,"*, *"No wait,"*, *"Hmm,"*, *"Let me reconsider"*, *"I was wrong"*, *"Going back to"*, *"On second thought"*, *"That's not right"* |
| `BRANCH` | `BRCH` | *"Alternatively,"*, *"Another approach"*, *"What if instead"*, *"Or perhaps"*, *"A different way to think"*, *"Let me try a different angle"* |
| `META` | `SEQ` (promoted later) | *"I'm overcomplicating this"*, *"Let me step back"*, *"This is getting circular"*, *"I need to be more careful"*, *"Let me re-read the question"* |
| `CONVERGENCE` | `SYNT` | *"So the key insight is"*, *"Therefore,"*, *"This means that"*, *"To summarize"*, *"The answer is"*, *"Putting this together"* |
| `ELABORATION` | `ELAB` | *"Specifically,"*, *"More precisely,"*, *"To be concrete,"*, *"In other words,"*, *"That is,"*, *"For example,"* |
| `CONTRAST` | `CONT` | *"However,"*, *"But,"*, *"On the other hand,"*, *"In contrast,"*, *"Despite this,"* |
| `SUPPORT` | `SUPP` | *"Because,"*, *"Since,"*, *"This is because,"*, *"The reason is,"*, *"Evidence for this:"* |

**Implementation details:**

- Matching is done at the **sentence start** using compiled regex with case-insensitive matching and word boundary anchors
- Multi-word cues take priority over single-word cues when both match at the same position
- Paragraph breaks (double newline) are always treated as hard boundaries with default `SEQ` edge
- A minimum span length of **3 sentences or 50 tokens** is enforced — if a hard boundary would produce a TU shorter than this, it is merged with the following span

The output of Pass 1 is a sequence of *(span, boundary_class)* pairs where the boundary class is known, not inferred.

### 3.2 Pass 2: Soft Boundary Detection (Semantic Shift, TextTiling)

Pass 2 operates on the remaining spans from Pass 1 — those that contain no hard boundary cues. It identifies **soft boundaries** caused by implicit topic shifts using a sentence-embedding-based TextTiling approach.

**Algorithm:**

1. Encode all sentences within each Pass-1 span using `all-MiniLM-L6-v2` (384-dim, fast, sufficient for relative similarity)
2. For each pair of adjacent sentence windows of size `w=3`, compute cosine similarity between the left window mean embedding and the right window mean embedding
3. Construct a similarity curve over sentence positions
4. Detect local minima (valleys) in the similarity curve using `scipy.signal.find_peaks` on the inverted signal, with a minimum prominence of `δ=0.15`
5. Insert a soft boundary at each detected valley if and only if the similarity at that position is below a threshold `τ`

**Threshold calibration:** `τ` is set to the 30th percentile of all within-trace pairwise similarities computed on a held-out calibration set of 20 traces per model. This makes `τ` model-adaptive — models with more uniform semantic profiles (e.g., highly focused reasoning) will have a higher `τ`, avoiding over-segmentation.

**Soft boundary edge type:** All soft boundaries are assigned `SEQ` by default. Pass 3 may promote these edges, but they are never left disconnected.

**Output:** A refined sequence of TUs where each TU is either hard-bounded (with a typed edge) or soft-bounded (with a `SEQ` edge, subject to promotion).

### 3.3 Pass 3: NLI-Based Edge Promotion

Pass 3 upgrades `SEQ` edges to semantically typed edges where there is a detectable semantic relation, using a zero-shot NLI approach. This is the **only model inference step** in the extraction pipeline, and it uses a discriminative classifier (not a generative model).

**Model:** `cross-encoder/nli-deberta-v3-large` (fine-tuned on MNLI+SNLI, ~400M parameters)

**Protocol:** For each pair of adjacent TUs `(TU_i, TU_j)` where the current edge type is `SEQ`, and where `j - i ≤ 4` (local window), construct three NLI hypotheses:

```
H_SUPP: "The second passage provides evidence or justification for the first."
H_CONT: "The second passage contradicts or opposes the first."
H_ELAB: "The second passage is a more specific or detailed version of the first."
```

The premise is the concatenation of `TU_i` text. For each hypothesis, run the NLI cross-encoder and record the entailment score `p(entailment | premise, hypothesis)`.

**Promotion rule:**
- If `max(p_SUPP, p_CONT, p_ELAB) > 0.75`, assign the corresponding edge type
- If `p_SUPP > 0.75` and `p_CONT > 0.75` simultaneously (rare but possible), assign `CONT` (the stronger/more specific relation)
- Otherwise, keep `SEQ`

**Non-local back-references:** For `BACK` edges beyond the local window, a secondary scan checks all TU pairs `(TU_i, TU_j)` where `j - i > 4` and `TU_j` was assigned a `BACKTRACK` boundary class in Pass 1. For these, the NLI model is run with:

```
H_BACK: "The second passage revises or contradicts an earlier position."
```

If `p(entailment) > 0.70`, a `BACK` edge is added as a non-sequential directed edge (in addition to the sequential backbone).

**Output:** A fully typed edge list for the Thought Graph construction.

### 3.4 Segmentation Validation

To validate the pipeline, a subset of **30 traces** (3 per model, 3 domains) are manually annotated by two independent annotators who mark TU boundaries and assign boundary types. Inter-annotator agreement is measured using:

- **Boundary F1** (WindowDiff metric, standard for discourse segmentation)
- **Cohen's κ** for boundary type agreement among agreed boundaries

The pipeline is evaluated against the gold annotations using the same F1 metric.

---

## 4. Node Classification

Once TUs are segmented, each is assigned a **node type** from the 12-type taxonomy (4 families). Unlike v1, node classification in v2 is done using a **fine-tuned DeBERTa-base classifier** rather than prompting a generative model.

### 4.1 Node Type Taxonomy

| Family | Type | Code | Definition |
|--------|------|------|------------|
| EXPLORATION | Hypothesis | HYP | Proposes a possible answer or approach |
| EXPLORATION | Reframing | RFR | Restates the problem from a different angle |
| EXPLORATION | Analogy | ANA | Draws a parallel to another domain |
| EXPLORATION | Brainstorm | BRS | Lists options without evaluation |
| ELABORATION | Justification | JUS | Provides evidence or logical support |
| ELABORATION | Specification | SPC | Makes an idea more concrete or precise |
| ELABORATION | Implication | IMP | Derives a consequence from a prior idea |
| ELABORATION | Constraint | CON | Identifies a limit, caveat, or boundary condition |
| EVALUATION | Critique | CRT | Points out a flaw or weakness |
| EVALUATION | Comparison | CMP | Weighs two or more alternatives |
| EVALUATION | Meta-reflection | MET | Reflects on the reasoning process itself |
| CONVERGENCE | Synthesis | SYN | Combines insights from multiple reasoning threads |

### 4.2 Classifier Training

The classifier is trained on **silver-label data** generated as follows:

1. Collect 500 `<think>` traces from QwQ-32B (different from benchmark data)
2. Use GPT-4o to classify each TU with confidence scores — this is the **only** use of a generative model in the entire pipeline, and it is used only for data generation, not for inference on benchmark data
3. Keep only classifications with `confidence ≥ 0.85` (estimated ~70% retention)
4. Fine-tune `DeBERTa-base` on the resulting labeled dataset with 80/10/10 split

**Features:** TU text + preceding TU text (2-sentence context window) + boundary class from Pass 1 (as a special token prefix).

The boundary class prefix improves classification accuracy significantly because boundary class and node type are correlated: a `BACKTRACK` boundary almost always introduces a `CRT` or `HYP` node; a `CONVERGENCE` boundary almost always introduces a `SYN` node. Using this as a feature prevents the classifier from having to re-derive what the boundary detection already knows.

### 4.3 Rule-Based Fallback

If classifier confidence is below 0.60 for a given TU, a deterministic fallback is applied based on the TU's boundary class:

| Boundary Class | Fallback Node Type |
|---|---|
| BACKTRACK | CRT |
| BRANCH | HYP |
| META | MET |
| CONVERGENCE | SYN |
| ELABORATION | SPC |
| CONTRAST | CMP |
| SUPPORT | JUS |
| SEQ (soft) | HYP (first TU in trace), SPC (otherwise) |

---

## 5. Graph Construction

The **Thought Graph** `G = (V, E)` is a directed multigraph where:

- Each vertex `v ∈ V` is a Thought Unit with attributes: `{tu_id, text, node_type, node_family, char_start, char_end, token_count, classification_confidence}`
- Each edge `e ∈ E` is a typed directed relation with attributes: `{source_id, target_id, edge_type, confidence, is_sequential}`

### 5.1 Graph Assembly

The graph is assembled in three layers:

**Layer 1 — Sequential backbone:** Every TU `i` is connected to TU `i+1` with a `SEQ` edge. This guarantees connectivity and ensures the orphan ratio is structurally zero (every node has at least one edge).

**Layer 2 — Semantic edges from Pass 1:** For all hard-boundary transitions, replace the `SEQ` edge with the typed edge determined by the boundary class.

**Layer 3 — Promoted edges from Pass 3:** Add promoted edges (SUPP, CONT, ELAB) as additional edges alongside the sequential backbone. For non-local `BACK` edges, add as supplementary edges without removing the sequential one.

**Key property:** By construction, the graph is always connected. Orphan nodes cannot exist because every TU participates in the sequential backbone. The orphan ratio metric from v1 is therefore retired — it was a pipeline health diagnostic, not a cognitive feature.

### 5.2 Graph Schema

```python
@dataclass
class ThoughtUnit:
    tu_id: int
    text: str
    node_type: NodeType          # 12-class taxonomy
    node_family: NodeFamily      # 4-class taxonomy
    char_start: int
    char_end: int
    token_count: int
    classification_confidence: float
    boundary_class: BoundaryClass  # NEW: from Pass 1

@dataclass  
class ThoughtEdge:
    source_id: int
    target_id: int
    edge_type: EdgeType          # SEQ | ELAB | BRCH | BACK | SYNT | CRIT | SUPP | CONT
    confidence: float
    is_sequential: bool          # True if part of sequential backbone

@dataclass
class ThoughtGraph:
    trace_id: str
    model: str
    domain: str
    question_id: str
    run: int
    nodes: List[ThoughtUnit]
    edges: List[ThoughtEdge]
    raw_cot: str
    token_count: int
```

---

## 6. Cognitive Metrics

All 22 metrics are computed directly from the Thought Graph topology and surface text features. They are grouped into 5 categories.

### 6.1 Notation

- `V` = set of all nodes, `|V|` = number of nodes (TUs)
- `E` = set of all edges, `E_sem` = semantic (non-SEQ) edges only
- `E_type` = edges of a specific type (e.g., `E_BACK`, `E_ELAB`)
- `N_type` = nodes of a specific type (e.g., `N_HYP`, `N_MET`)
- `d_out(v)` = out-degree of node v (semantic edges only)
- `T` = total token count of the trace

### 6.2 Breadth Metrics (4)

These measure the **horizontal span** of the reasoning — how many different angles, domains, and initial directions the model explores.

**`branching_factor`** — Average number of semantic out-edges per node, measuring the tendency to spawn new reasoning directions from each thought:
```
branching_factor = |E_BRCH| / |V|
```

**`unique_perspective_count`** — Number of distinct RFR (Reframing) nodes, indicating how many times the model restated the problem from a fresh angle:
```
unique_perspective_count = |N_RFR|
```

**`domain_spread`** — Number of distinct semantic clusters among BRS and HYP nodes, computed via agglomerative clustering (cosine distance, threshold=0.45) on sentence embeddings of those nodes:
```
domain_spread = n_clusters(N_BRS ∪ N_HYP, threshold=0.45)
```

**`first_idea_diversity`** — Pairwise cosine distance among the embeddings of the first HYP node across K=3 runs for the same question. Measures how differently the model initializes its reasoning under identical prompts:
```
first_idea_diversity = mean(cosine_dist(embed(HYP_0^i), embed(HYP_0^j))) for all i≠j in {1..K}
```
*This metric is defined at the question level, not the trace level, and requires K≥2.*

### 6.3 Depth Metrics (4)

These measure the **vertical reach** of reasoning — how far the model elaborates and specifies any given idea.

**`max_elaboration_chain`** — Length of the longest directed path composed exclusively of ELAB edges:
```
max_elaboration_chain = max_path_length(G[E_ELAB])
```

**`mean_branch_depth`** — Average depth of nodes in the graph, where depth is the length of the shortest path from any root node (in-degree = 0 in the semantic subgraph):
```
mean_branch_depth = mean(depth(v) for v in V)
```

**`specificity_gradient`** — Rate of change in named entity density (NER count / token count) along the sequential path from start to end. Positive values indicate progressive specification; negative indicates abstract drift:
```
specificity_gradient = slope(NER_density(v_i) ~ position(v_i)) via linear regression
```
*Requires spaCy `en_core_web_sm`. Computed on the sequential TU order.*

**`reasoning_density`** — Fraction of nodes that carry a semantic edge (ELAB, BRCH, BACK, SYNT, CRIT, SUPP, CONT), as opposed to being connected only by the sequential backbone:
```
reasoning_density = |{v : ∃e ∈ E_sem s.t. source(e)=v or target(e)=v}| / |V|
```

### 6.4 Structure Metrics (6)

These measure the **topological properties** of the Thought Graph — its branching patterns, self-correction behavior, integration of ideas, and overall coherence.

*Note: v2 retires `orphan_ratio` (structural artifact of the old linker) and `cycle_count` / `mean_cycle_length` is reframed as revision metrics since the graph is not a DAG.*

**`exploration_exploitation_ratio`** — Ratio of EXPLORATION-family nodes to ELABORATION-family nodes, measuring the balance between generating new ideas and deepening existing ones:
```
E/E_ratio = |N_EXPLORATION| / max(|N_ELABORATION|, 1)
```

**`backtracking_rate`** — Fraction of all semantic edges that are BACK edges, measuring how often the model explicitly revises prior positions:
```
backtracking_rate = |E_BACK| / max(|E_sem|, 1)
```

**`cross_branch_connectivity`** — Fraction of node pairs in different connected components of the BRCH-edge subgraph that are connected by at least one SYNT or SUPP edge. Measures integration of parallel reasoning threads:
```
cross_branch_connectivity = |{(b1,b2) : branch(b1)≠branch(b2), ∃ SYNT/SUPP edge between them}| / |branch_pairs|
```

**`convergence_index`** — Fraction of SYN nodes relative to total nodes, weighted by their in-degree in the semantic subgraph (SYN nodes with more predecessors are stronger convergence signals):
```
convergence_index = sum(d_in(v) for v in N_SYN) / (|V| * mean_d_in)
```

**`graph_density`** — Density of the semantic edge subgraph:
```
graph_density = |E_sem| / (|V| * (|V| - 1))
```

**`revision_depth`** — *Replaces `cycle_count`/`mean_cycle_length`.* For each BACK edge `(u, v)` where `v` precedes `u` in sequential order, the revision depth is the number of TUs between `v` and `u`. High revision depth means the model revisits ideas from far back:
```
revision_depth = mean(|position(u) - position(v)| for (u,v) in E_BACK)
```

### 6.5 Metacognitive Metrics (4)

These measure the model's awareness of and commentary on its own reasoning process.

**`self_reflection_rate`** — Fraction of nodes classified as MET (Meta-reflection):
```
self_reflection_rate = |N_MET| / |V|
```

**`critique_to_hypothesis_ratio`** — Ratio of CRT nodes to HYP nodes, measuring whether the model evaluates its hypotheses as much as it generates them:
```
C/H_ratio = |N_CRT| / max(|N_HYP|, 1)
```

**`hedging_density`** — Fraction of TUs containing epistemic uncertainty markers. The hedging lexicon includes: *"might", "could", "possibly", "perhaps", "I'm not sure", "it's unclear", "arguably", "one could argue", "I think", "it seems", "probably", "likely", "uncertain"*:
```
hedging_density = |{v : hedging_marker ∈ text(v)}| / |V|
```

**`perspective_taking`** — Fraction of nodes classified as RFR, measuring how often the model adopts a new viewpoint:
```
perspective_taking = |N_RFR| / |V|
```

### 6.6 Efficiency Metrics (2)

These measure the token cost of the reasoning relative to its informational content.

**`token_per_idea`** — Average tokens consumed per unique reasoning unit (Unique Perspective Count):
```
token_per_idea = T / max(unique_perspective_count, 1)
```

**`redundancy_ratio`** — Fraction of TU pairs with cosine similarity above 0.90 (near-duplicate ideas), normalized by total pairs:
```
redundancy_ratio = |{(i,j) : i<j, cosine(embed(TU_i), embed(TU_j)) > 0.90}| / C(|V|, 2)
```
*Requires sentence-transformers.*

### 6.7 Summary Table

| # | Metric | Category | Requires |
|---|--------|----------|---------|
| 1 | branching_factor | Breadth | graph |
| 2 | unique_perspective_count | Breadth | graph |
| 3 | domain_spread | Breadth | embeddings |
| 4 | first_idea_diversity | Breadth | embeddings, K≥2 |
| 5 | max_elaboration_chain | Depth | graph |
| 6 | mean_branch_depth | Depth | graph |
| 7 | specificity_gradient | Depth | spaCy |
| 8 | reasoning_density | Depth | graph |
| 9 | exploration_exploitation_ratio | Structure | graph |
| 10 | backtracking_rate | Structure | graph |
| 11 | cross_branch_connectivity | Structure | graph |
| 12 | convergence_index | Structure | graph |
| 13 | graph_density | Structure | graph |
| 14 | revision_depth | Structure | graph |
| 15 | self_reflection_rate | Metacognitive | graph |
| 16 | critique_to_hypothesis_ratio | Metacognitive | graph |
| 17 | hedging_density | Metacognitive | lexicon |
| 18 | perspective_taking | Metacognitive | graph |
| 19 | token_per_idea | Efficiency | tokens |
| 20 | redundancy_ratio | Efficiency | embeddings |
| 21 | avg_tokens | Summary | tokens |
| 22 | avg_tus | Summary | graph |

---

## 7. Profile Aggregation

For each model `m`, a cognitive profile is computed by aggregating metrics across all traces attributed to that model.

### 7.1 Per-Trace Profile

Each trace produces a 22-dimensional metric vector. All metrics that depend on K=3 runs (`first_idea_diversity`) are computed at the question level and then averaged across questions.

### 7.2 Model-Level Aggregation

For each metric `k` and model `m`, the model-level value is:
```
μ_k(m) = mean over all traces t of metric_k(t)
σ_k(m) = std over all traces t of metric_k(t)
CV_k(m) = σ_k(m) / μ_k(m)   # coefficient of variation = stability measure
```

The **stability report** flags any metric where `CV > 0.5` for any model as potentially unreliable.

### 7.3 Domain-Conditioned Profiles

Profiles are also computed separately per domain, producing a `(model × domain × metric)` tensor that allows analysis of: (a) domain sensitivity of each model, (b) cross-domain consistency of a model's profile, (c) interaction effects between model type and domain.

### 7.4 Normalization for Visualization

For radar chart visualization, each metric is normalized to [0,1] using **empirical min-max bounds** derived from the full benchmark results, not hand-coded bounds. This ensures the radar chart reflects the actual relative position of each model within the observed distribution, rather than an a priori assumption about possible ranges.

---

## 8. Validation Protocol

### 8.1 Segmentation Validation (Component-Level)

- **Annotators:** 2 independent annotators
- **Material:** 30 traces (3 per model, 3 domains, ~5,000 TUs total)
- **Metrics:** WindowDiff boundary F1, Cohen's κ for boundary class agreement
- **Target:** F1 ≥ 0.75, κ ≥ 0.65 (sufficient for a new segmentation task)

### 8.2 Perturbation Validation (Metric Sensitivity)

To verify that metrics respond in the expected direction to known structural changes, controlled perturbations are applied to 20 held-out traces:

| Perturbation | Expected effect |
|---|---|
| Truncate trace to first 30% | ↓ max_elaboration_chain, ↓ convergence_index |
| Duplicate every TU | ↑ redundancy_ratio |
| Remove all hedging markers | ↓ hedging_density |
| Remove all BACK-type boundaries | ↓ backtracking_rate |
| Remove all MET nodes | ↓ self_reflection_rate |

Each perturbation is expected to shift the target metric by at least 1 standard deviation while leaving unrelated metrics unchanged. This is a falsifiable sensitivity test that does not require external ground truth.

### 8.3 Discriminability Validation (Cross-Model)

The primary validity claim is that metrics meaningfully discriminate between models that differ in known ways. Two specific discrimination tests are reported:

**Test 1 — Reasoning vs. instruction-tuned:** Compare profiles of reasoning-native models (all 8 in the roster) against a non-reasoning baseline (e.g., Llama-3.1-8B-Instruct run without `<think>`-style prompting). Reasoning models should show higher `backtracking_rate`, `self_reflection_rate`, and `revision_depth`.

**Test 2 — Scale effect within family:** Compare Qwen3-7B vs. Qwen3-32B vs. Qwen3-235B profiles. Larger models should show higher `max_elaboration_chain`, `cross_branch_connectivity`, and `convergence_index`.

Statistical significance is assessed using Mann-Whitney U test (non-parametric, appropriate for metric distributions) with Bonferroni correction across the 22 metrics.

---

## 9. Pipeline Architecture Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                      INPUT: 200 questions                        │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: COLLECTION                                             │
│  • 8 reasoning models, K=3 runs each                             │
│  • Extract <think> block only                                    │
│  • Output: traces_*.jsonl (up to 4,800 traces)                  │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2: SEGMENTATION (Non-generative, 3-pass)                 │
│                                                                  │
│  Pass 1: Cue-phrase lexicon → hard boundaries + edge types       │
│          [deterministic, regex, 7 boundary classes]              │
│                              ↓                                   │
│  Pass 2: TextTiling (MiniLM embeddings) → soft boundaries        │
│          [discriminative embeddings, no generation]              │
│                              ↓                                   │
│  Pass 3: DeBERTa-NLI → SEQ edge promotion                        │
│          [discriminative classifier, window=4, threshold=0.75]   │
│                                                                  │
│  Output: TU sequence with typed boundaries                       │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3: NODE CLASSIFICATION                                    │
│  • DeBERTa-base fine-tuned on silver-label data                  │
│  • 12-type taxonomy, 4 families                                  │
│  • Rule-based fallback for low-confidence predictions            │
│  Output: TUs with node_type, node_family                         │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 4: GRAPH CONSTRUCTION                                     │
│  • Layer 1: Sequential backbone (always connected)               │
│  • Layer 2: Typed edges from Pass 1 boundaries                   │
│  • Layer 3: Promoted edges from Pass 3 NLI                       │
│  • Layer 4: Non-local BACK edges                                 │
│  Output: ThoughtGraph (V, E) per trace                           │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 5: METRIC COMPUTATION                                     │
│  • 22 metrics across 5 categories                                │
│  • Per-trace, per-question (K-aggregated), per-model             │
│  • Domain-conditioned profiles                                   │
│  Output: profile_tensor [model × domain × metric]               │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 6: VALIDATION & REPORTING                                 │
│  • Segmentation IAA (WindowDiff F1, κ)                           │
│  • Perturbation sensitivity tests                                │
│  • Cross-model discriminability (Mann-Whitney + Bonferroni)      │
│  • Radar chart (empirical normalized bounds)                     │
│  • Domain heatmap                                                │
│  Output: benchmark_report.md, figures/                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 10. Key Differences from v1

| Component | v1 | v2 |
|---|---|---|
| Segmentation | Generative LLM prompt | 3-pass non-generative (cue-phrase + TextTiling + NLI) |
| Edge typing | Generative LLM 4-pass | Deterministic from boundary class + NLI promotion |
| Node classification | Generative LLM per-TU | Fine-tuned DeBERTa-base discriminative classifier |
| Orphan problem | Present (54.5%) | Structurally eliminated (sequential backbone) |
| Circular validity | Present (same model extracts and generates) | Eliminated (no generative model in extraction) |
| Normalization bounds | Hardcoded a priori | Empirical from benchmark results |
| `cycle_count` metric | Ill-defined (non-DAG artifact) | Replaced by `revision_depth` |
| `orphan_ratio` metric | Pipeline health proxy | Retired |
| Dataset | 10 questions, 1 domain | 200 questions, 5 domains |
| Model coverage | 1 model (development) | 8 reasoning models |
| Validation | None | IAA + perturbation + discriminability |

---

*ThinkBench v2 — Pipeline Specification*
*For questions about implementation, see `AGENT.md`. For metric derivations, see `METRICS.md`.*