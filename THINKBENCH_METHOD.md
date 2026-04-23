# ThinkBench: Profiling How LLMs Think on Open-Ended Problems

## Positioning & Novelty

**Gap exploited**: All existing CoT structural analysis (LCoT2Tree, CoT Encyclopedia, Gandhi et al.) evaluates reasoning on problems with ground truth. Nobody profiles *how* LLMs think when there is no correct answer — the only setting where reasoning process is the *sole* object of evaluation.

**Key differentiators from closest prior work**:

| Dimension | LCoT2Tree (EMNLP'25) | CoT Encyclopedia | Gandhi et al. | **ThinkBench (ours)** |
|---|---|---|---|---|
| Structure | Tree | Flat strategy clusters | Binary behavior flags | **Directed graph** (cross-branch links + cycles) |
| Task type | Closed (MATH, code) | Closed (benchmarks) | Closed (Countdown) | **Open-ended, no ground truth** |
| Goal | Predict correctness | Control strategy | Explain RL improvement | **Profile thinking style** |
| Output | GNN classifier | Strategy distribution | Behavior presence/absence | **Cognitive profile vector** |
| Models | 5 reasoning LLMs | Reasoning LLMs | 2 (Qwen, Llama) | **8+ thinking LLMs** |
| Scope | Per-trace analysis | Per-trace strategy | Per-model behavior | **Per-model fingerprint across domains** |

**One-line pitch**: *ThinkBench is the first benchmark that profiles LLM thinking behavior on problems where the process IS the product.*

---

## 1. Question Set Design

### 1.1 Principles

Questions must satisfy ALL of:
- **No single correct answer**: Multiple legitimate perspectives exist
- **Reasoning-demanding**: Cannot be answered with a fact lookup or one-liner
- **Thinking-model activating**: Complex enough that reasoning LLMs activate their extended thinking
- **Domain-diverse**: Cover distinct cognitive demands
- **Culturally neutral**: No Western-centric framing bias

### 1.2 Domain Taxonomy (6 domains, ~30 questions each = 180 total)

**D1: Ethical Dilemmas** — Force tradeoff reasoning under moral uncertainty  
*Examples*:
- "A self-driving car's braking system can save either the passenger or two pedestrians, but the pedestrians were jaywalking. How should the system decide?"
- "Should a doctor inform a patient's employer about a diagnosis that poses no risk to others but may cause the patient to lose their job?"

**D2: Policy Design** — Require weighing competing stakeholder interests  
*Examples*:
- "Design a policy for regulating AI-generated content in journalism. What tradeoffs must be accepted?"
- "How should a city allocate limited budget between public transit expansion and road maintenance?"

**D3: Strategic Planning** — Demand multi-step forward reasoning with uncertainty  
*Examples*:
- "A mid-size bookstore chain faces competition from online retailers. Develop a 3-year strategy."
- "How should a country with declining birth rates restructure its pension system?"

**D4: Scientific Speculation** — Require reasoning beyond established knowledge  
*Examples*:
- "If we discovered microbial life on Europa, how should this change our approach to space exploration?"
- "What would be the most promising approach to achieving room-temperature superconductivity?"

**D5: Creative Problem Solving** — Demand lateral thinking and novel synthesis  
*Examples*:
- "Design an education system optimized for a world where AI can pass any exam."
- "How could cities be redesigned if most work becomes remote?"

**D6: Interpersonal Reasoning** — Require Theory of Mind and social cognition  
*Examples*:
- "A manager discovers their best performer is quietly undermining a colleague. How should they handle it?"
- "Two co-founders disagree on whether to take VC funding or bootstrap. How should they resolve this?"

### 1.3 Quality Criteria for Each Question
- At least 3 distinct reasonable approaches exist (verified by authors)
- Question is self-contained (no external knowledge needed beyond general education)
- Estimated thinking trace length: 500-5000 tokens
- No question can be answered with a memorized pattern

---

## 2. CoT Collection Protocol

### 2.1 Target Models
Only models that expose thinking/reasoning traces:
- DeepSeek-R1 (open-weight, full CoT visible)
- QwQ-32B (Qwen reasoning model)
- Kimi k1.5
- Grok-3-mini (thinking mode)
- Claude (extended thinking — if accessible via API)
- Gemini 2.5 Pro (thinking mode)
- Seed-1.5-Thinking
- GLM-4-Z1 (ZhiPu)

### 2.2 Collection Parameters
- **Runs per model per question**: K=3 (to measure intra-model variance)
- **Temperature**: default (model-native)
- **System prompt**: Minimal — "Think carefully about this problem." (no style guidance)
- **Output**: Raw thinking trace (not the final answer)
- **Total traces**: 8 models × 180 questions × 3 runs = **4,320 traces**

### 2.3 Storage Schema
```json
{
  "trace_id": "uuid",
  "model": "deepseek-r1",
  "question_id": "D1_003",
  "domain": "ethical_dilemmas",
  "run": 1,
  "raw_cot": "Let me think about this step by step...",
  "token_count": 2847,
  "collected_at": "2026-04-15T10:00:00Z"
}
```

---

## 3. Thought Graph Extraction (Core Method)

This is the technical heart of the paper. The pipeline has three stages: **Segment → Classify → Link**. The output is a **directed graph** (not a tree or DAG) — cycles are permitted and constitute a meaningful signal (iterative refinement, oscillation between perspectives, circular justification).

Critically, **no generative LLM is used in extraction** (v2). All segmentation and classification is done with deterministic rules and discriminative models. This eliminates circular validity — v1 used a generative LLM to analyze other models' outputs, introducing potential bias.

### 3.1 Stage A: Segmentation — CoT Text → Atomic Thought Units (3-Pass Non-Generative)

**Definition**: A *thought unit* (TU) is a contiguous span of the CoT that expresses a single coherent idea, claim, consideration, or reasoning step. It is the smallest unit that can be understood in isolation.

**Pass 1 — Hard Boundaries (Regex)**

A compiled lexicon of cue phrases is matched at sentence starts. Each match assigns a `BoundaryClass`:

| BoundaryClass | Example Cue Phrases |
|---|---|
| BACKTRACK | "Wait,", "Actually,", "Let me reconsider", "I was wrong" |
| BRANCH | "Alternatively,", "Another approach", "What if", "Or we could" |
| META | "I'm going in circles", "Let me step back", "I need to reconsider my approach" |
| CONVERGENCE | "In conclusion", "Bringing this together", "So the answer is" |
| ELABORATION | "Specifically,", "For example,", "In practice,", "More concretely" |
| CONTRAST | "On the other hand,", "However,", "But", "Yet" |
| SUPPORT | "Because", "This is supported by", "The reason is", "Evidence for this" |

Multi-word cues take priority over single-word cues. Paragraph breaks produce SEQ boundaries. Spans shorter than 3 sentences or 50 tokens are merged with the following span.

**Pass 2 — Soft Boundaries (TextTiling with MiniLM)**

On spans without a hard boundary, sentences are encoded with `all-MiniLM-L6-v2`. A sliding window of size 3 computes cosine similarity between adjacent windows. Local minima detected by `scipy.signal.find_peaks` (prominence ≥ 0.15) that fall below τ (30th percentile of within-trace adjacent-sentence similarities) become soft boundaries (`BoundaryClass.NONE`, edge `SEQ`).

**Pass 3 — NLI Edge Promotion (DeBERTa)**

`cross-encoder/nli-deberta-v3-large` promotes SEQ edges to typed semantic edges. For each adjacent TU pair within window=4, three hypotheses are tested (SUPP/CONT/ELAB) at threshold 0.75. For BACKTRACK TUs more than 4 positions apart, a BACK hypothesis is tested at threshold 0.70. All pairs for a trace are submitted as a **single batched** `predict()` call for efficiency.

**Validation protocol**:
- Human-annotate 50 traces (across models/domains) for segmentation boundaries
- Compute boundary F1 (with ±15 char tolerance window)
- Report inter-annotator agreement (2 annotators, Cohen's κ)
- Target: boundary F1 ≥ 0.80, κ ≥ 0.70

### 3.2 Stage B: Node Classification — Assigning Functional Types

**Each TU is classified into exactly one functional type** using a deterministic rule-based map from `BoundaryClass` to `NodeType`. No LLM call is needed.

#### BoundaryClass → NodeType Map

| BoundaryClass | NodeType | NodeFamily |
|---|---|---|
| BACKTRACK | CRT (Critique) | EVALUATION |
| BRANCH | HYP (Hypothesis) | EXPLORATION |
| META | MET (Meta-reflection) | EVALUATION |
| CONVERGENCE | SYN (Synthesis) | CONVERGENCE |
| ELABORATION | SPC (Specification) | ELABORATION |
| CONTRAST | CMP (Comparison) | EVALUATION |
| SUPPORT | JUS (Justification) | ELABORATION |
| NONE, idx=0 | HYP (Hypothesis) | EXPLORATION |
| NONE, idx>0 | SPC (Specification) | ELABORATION |

Classification confidence is 1.0 for hard-boundary TUs and 0.5 for soft-boundary (NONE class).

A fine-tuned DeBERTa node classifier is planned for a future iteration — the stub `load_deberta_classifier()` raises `NotImplementedError` as a placeholder.

#### Node Type Taxonomy (12 types, 4 families)

**Family 1: EXPLORATION** (generating new directions)
| Type | Code | Definition | Example marker phrases |
|---|---|---|---|
| Hypothesis | `HYP` | Proposes a possible answer, approach, or framing | "Perhaps...", "One approach could be...", "What if..." |
| Reframing | `RFR` | Re-states the problem from a different angle | "Looking at this differently...", "From X's perspective..." |
| Analogy | `ANA` | Draws a parallel to another domain/situation | "This is similar to...", "Like how..." |
| Brainstorm | `BRS` | Lists options/possibilities without deep evaluation | "Options include...", "We could also..." |

**Family 2: ELABORATION** (deepening an existing thread)
| Type | Code | Definition | Example marker phrases |
|---|---|---|---|
| Justification | `JUS` | Provides evidence or reasoning for a prior claim | "Because...", "This is supported by...", "The reason is..." |
| Specification | `SPC` | Makes a prior idea more concrete/detailed | "Specifically...", "For example...", "In practice..." |
| Implication | `IMP` | Derives a consequence from a prior claim | "This means...", "Therefore...", "As a result..." |
| Constraint | `CON` | Identifies limits, conditions, or caveats | "However, this only works if...", "The limitation is..." |

**Family 3: EVALUATION** (judging ideas)
| Type | Code | Definition | Example marker phrases |
|---|---|---|---|
| Critique | `CRT` | Points out weaknesses in a prior idea | "But this fails because...", "The problem with..." |
| Comparison | `CMP` | Explicitly weighs two or more alternatives | "Compared to X, Y is...", "The tradeoff is..." |
| Meta-reflection | `MET` | Reflects on the reasoning process itself | "I'm going in circles...", "Let me step back..." |

**Family 4: CONVERGENCE** (moving toward conclusions)
| Type | Code | Definition | Example marker phrases |
|---|---|---|---|
| Synthesis | `SYN` | Combines insights from multiple threads | "Bringing these together...", "Combining X and Y..." |

**Validation**:
- Human-classify 200 TUs (stratified by model and domain)
- Report macro-F1 per type and confusion matrix
- Target: macro-F1 ≥ 0.75

### 3.3 Stage C: Graph Assembly

The segmenter produces a complete list of `ThoughtUnit` objects (with `boundary_class` and typed edges already assigned by Passes 1–3). The linker's only role is to assemble these into a `ThoughtGraph` Pydantic object — no additional LLM calls or edge inference.

#### Edge Type Taxonomy (8 types)

| Edge Type | Code | Direction | Definition |
|---|---|---|---|
| Elaboration | `ELAB` | parent → child | Child deepens or specifies parent |
| Branching | `BRCH` | parent → child | Child explores a new direction from parent's setup |
| Backtracking | `BACK` | child → ancestor | Explicitly revises or abandons an earlier idea |
| Synthesis | `SYNT` | [multiple] → target | Target combines ideas from multiple source nodes |
| Critique | `CRIT` | source → target | Source evaluates/challenges target |
| Support | `SUPP` | source → target | Source provides evidence for target |
| Contrast | `CONT` | source ↔ target | Source and target represent opposing perspectives |
| Sequential | `SEQ` | prev → next | Default adjacency (no semantic relationship beyond order) |

Every edge carries an `is_sequential: bool` flag. The sequential backbone (Pass 1) sets `is_sequential=True`; NLI-promoted edges set it to `False`. Metrics that require "semantic edges only" filter on this flag.

#### Thought Graph Output Schema
```json
{
  "trace_id": "uuid",
  "model": "Qwen/Qwen3.5-35B-A3B",
  "question_id": "D1_001",
  "domain": "ethical_dilemmas",
  "run": 1,
  "token_count": 2847,
  "nodes": [
    {
      "tu_id": 0,
      "text": "Let me consider the ethical dimensions...",
      "boundary_class": "BRANCH",
      "node_type": "HYP",
      "node_family": "EXPLORATION",
      "classification_confidence": 1.0,
      "start_char": 0,
      "end_char": 147,
      "token_count": 28
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "edge_type": "ELAB",
      "confidence": 0.92,
      "is_sequential": false
    }
  ]
}
```

---

## 4. Metric Suite — The Cognitive Profile Vector

Each trace produces a numeric vector. Each model's profile is the mean vector across all its traces (with standard deviations for stability analysis).

### 4.1 Breadth Metrics

| Metric | Formula | Intuition |
|---|---|---|
| **Branching factor** (BF) | `\|E_BRCH\| / max(\|V\|, 1)` | How many distinct directions does the model explore? |
| **Unique perspective count** (UPC) | Count of RFR (Reframing) nodes | How many truly independent angles does the model try? |
| **Domain spread** (DS) | Number of agglomerative clusters (cosine threshold=0.45) of BRS+HYP node embeddings | Does the model consider multiple topically distinct framings? |
| **First-idea diversity** (FID) | Mean pairwise cosine distance among embeddings of first 3 HYP nodes | How different are the model's initial ideas from each other? |

### 4.2 Depth Metrics

*Note: Depth metrics are computed on the ELAB-only subgraph (which is naturally acyclic — elaboration doesn't loop back). This isolates the depth signal from iterative refinement cycles.*

| Metric | Formula | Intuition |
|---|---|---|
| **Max elaboration chain** (MEC) | Longest path following only ELAB edges from any single root | Deepest single-thread reasoning |
| **Mean branch depth** (MBD) | Average depth across all branches (semantic subgraph) | Typical elaboration depth |
| **Specificity gradient** (SG) | Linear regression slope of entity density vs. depth | Do ideas become more concrete as they're developed? |
| **Reasoning density** (RD) | `\|nodes with ≥1 semantic edge\| / \|V\|` | What fraction of thinking is logically connected? |

### 4.3 Structural Metrics

| Metric | Formula | Intuition |
|---|---|---|
| **Exploration-exploitation ratio** (EER) | EXPLORATION nodes / ELABORATION nodes | Breadth-first vs. depth-first style |
| **Backtracking rate** (BR) | `\|E_BACK\| / max(\|E_sem\|, 1)` | How often does the model revise itself? |
| **Cross-branch connectivity** (CBC) | Fraction of cross-branch node pairs with SYNT or SUPP edge | Does the model connect different threads? |
| **Convergence index** (CI) | `sum(d_in(v) for SYN nodes) / (\|V\| × mean_d_in)` | Degree to which synthesis nodes are well-connected |
| **Graph density** (GD) | `\|E_sem\| / (\|V\|×(\|V\|-1))` (semantic edges only) | Overall semantic connectivity |
| **Revision depth** (RvD) | `mean(\|pos(u) - pos(v)\|) for BACK edges` | How far back does the model revise? |

### 4.4 Metacognitive Metrics

| Metric | Formula | Intuition |
|---|---|---|
| **Self-reflection rate** (SRR) | MET nodes / total nodes | How often does the model monitor its own reasoning? |
| **Critique-to-hypothesis ratio** (CHR) | CRT nodes / HYP nodes | Does the model evaluate as much as it generates? |
| **Hedging density** (HD) | Proportion of TUs containing uncertainty markers ("might", "could", "it's possible") | How much epistemic humility? |
| **Perspective-taking** (PT) | RFR nodes / total nodes | How often does the model shift viewpoint? |

### 4.5 Efficiency Metrics

| Metric | Formula | Intuition |
|---|---|---|
| **Token-per-idea** (TPI) | `avg_tokens / max(UPC, 1)` | Verbosity relative to idea generation |
| **Redundancy ratio** (RR) | Proportion of TU pairs within same trace with embedding similarity > 0.90 | How repetitive is the reasoning? |

### 4.6 Profile Vector

The full profile vector for a model M on domain D:

```
V(M, D) = [BF, UPC, DS, FID, MEC, MBD, SG, RD, EER, BR, CBC, CI, GD, RvD, SRR, CHR, HD, PT, TPI, RR, avg_tokens, avg_tus]
```

**22 dimensions** (20 cognitive metrics + 2 summary stats), computed as means over all traces of model M in domain D.

**Global profile** V(M) = mean across all 6 domains.  
**Domain-sensitivity** = std(V(M, D)) across domains — measures how much the model adapts its style.

---

## 5. Cognitive Profiling & Analysis

### 5.1 Model Characterization

For each model, produce a **ThinkCard**:
- Radar chart of the 20-metric profile (normalized to [0,1] across all models)
- Dominant thinking archetype label (derived from clustering)
- Domain-sensitivity score
- Strengths/weaknesses summary

### 5.2 Archetype Discovery

Apply PCA to the 22-dim profile space to identify 2-3 principal components.  
Apply k-means (k=3 or 4, selected by silhouette score) to identify thinking archetypes.

**Expected archetypes** (hypothetical):
- **Divergent Explorer**: High BF, high UPC, low MEC — many ideas, shallow follow-through
- **Deep Deliberator**: Low BF, high MEC, high RD — few ideas, deeply developed
- **Dialectical Reasoner**: High BR, high CHR, high CBC — thesis-antithesis-synthesis pattern
- **Iterative Refiner**: High CC, short MCL, high BR — revisits and improves ideas in tight loops
- **Cautious Analyst**: High HD, high SRR, high CI — hedges frequently, converges carefully

### 5.3 Domain Sensitivity Analysis

Key research question: *Do models adapt their thinking style to the domain, or do they have a fixed cognitive signature?*

Compute: `sensitivity(M) = mean_over_metrics(std_across_domains(metric(M, D)))`

**High sensitivity** = model adapts (uses different strategies for ethics vs. strategy)  
**Low sensitivity** = fixed style (same approach regardless of domain)

### 5.4 Stability Analysis

Using the K=3 runs per question:
- Compute profile vector for each run
- Report intra-model, intra-question variance
- A model is "stable" if its profile is consistent across runs (low variance)
- A model is "volatile" if different runs produce different thinking patterns

### 5.5 Visualization Deliverables

1. **Radar charts** — one per model, overlaid for comparison
2. **PCA scatter** — all models projected to 2D, colored by archetype
3. **Domain heatmap** — models × domains × key metrics
4. **Pairwise distance matrix** — Jensen-Shannon divergence between model profiles
5. **Archetype exemplars** — one trace per archetype, visualized as a thought graph

---

## 6. Experimental Validation

### 6.1 Parser Validation
- 50 traces human-annotated for segmentation boundaries → boundary F1
- 200 TUs human-classified for node types → macro-F1, confusion matrix
- 100 TU pairs human-judged for edge types → edge classification accuracy
- All annotated by 2 independent annotators → Cohen's κ

### 6.2 Profile Stability
- Test-retest reliability across K=3 runs → intraclass correlation coefficient (ICC)
- Split-half reliability: split questions into two halves, check profile consistency

### 6.3 Discriminative Power
- Can profiles distinguish models? → ANOVA per metric across models
- Can profiles predict model identity? → leave-one-out classification accuracy using the profile vector as features

### 6.4 Baselines
- **Random segmentation** baseline (random boundaries, random types)
- **Length-only baseline** (profile based only on token count and sentence count)
- **LCoT2Tree adaptation** (run their tree method, compute metrics on tree) — shows directed graph adds information via cross-branch links and cycle detection

---

## 7. Related Work (Paper Structure)

Position against:
1. **CoT structural analysis**: LCoT2Tree, CoT Encyclopedia — we differ in task type (open-ended vs. closed) and structure (directed graph with cycles vs. tree/flat)
2. **Cognitive behavior detection**: Gandhi et al. — we go beyond binary detection to continuous profiling
3. **LLM reasoning benchmarks**: MATH, GPQA, LiveCodeBench — these evaluate answers, we evaluate process
4. **Thinking style in prompting**: StyleBench — evaluates prompting strategies, not native thinking traces
5. **Process supervision**: PRM work — trains verifiers, doesn't characterize thinking
6. **Cognitive science of reasoning**: Dual-process theory, heuristics & biases — our node taxonomy draws from this

---

## 8. Limitations & Future Work

- Node classification uses a rule-based boundary_class→node_type map (v2). A fine-tuned DeBERTa classifier per node type is planned for higher accuracy.
- Open-ended questions have no answer quality signal (by design — this is the point)
- Models without exposed CoT (e.g., GPT-4o) cannot be evaluated
- Profile stability may vary with prompt sensitivity
- NLI edge promotion uses a general-purpose cross-encoder — a domain-adapted model may improve edge precision
- Future: longitudinal profiling (does a model's thinking style change across versions?)
- Future: prescriptive application (route questions to the model with the best-matched thinking style)
