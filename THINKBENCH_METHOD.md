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

### 3.1 Stage A: Segmentation — CoT Text → Atomic Thought Units

**Definition**: A *thought unit* (TU) is a contiguous span of the CoT that expresses a single coherent idea, claim, consideration, or reasoning step. It is the smallest unit that can be understood in isolation.

**Segmentation Method**: LLM-based segmentation using a prompted parser.

**Parser prompt** (to a strong instruction model like Claude Sonnet or GPT-4.1):

```
You are a reasoning trace parser. Given a chain-of-thought reasoning trace, 
segment it into atomic thought units. Each thought unit should express exactly 
one idea, claim, hypothesis, consideration, or reasoning step.

Rules:
1. A thought unit is the smallest span that makes sense alone
2. Split at logical transitions: new ideas, direction changes, elaborations
3. Preserve the original text verbatim — only add boundary markers
4. Linguistic cues for boundaries include: "Wait,", "Actually,", "Alternatively,",
   "But", "Let me reconsider", "On the other hand", "Now,", "So,", "However,",
   "Hmm,", "What if", "Let me think about", paragraph breaks
5. Do NOT split within a single chain of deduction (A→B→C stays together if 
   it's one logical move)

Output format: Return a JSON array where each element is:
{
  "tu_id": integer (0-indexed),
  "text": "exact text span",
  "start_char": integer,
  "end_char": integer
}
```

**Validation protocol**:
- Human-annotate 50 traces (across models/domains) for segmentation boundaries
- Compute boundary F1 (with ±15 char tolerance window)
- Report inter-annotator agreement (2 annotators, Cohen's κ)
- Target: boundary F1 ≥ 0.80, κ ≥ 0.70

### 3.2 Stage B: Node Classification — Assigning Functional Types

**Each TU is classified into exactly one functional type**. This taxonomy is derived from cognitive science reasoning typologies and adapted for LLM traces.

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

**Classification Method**: LLM-based classifier with the taxonomy as a schema.

**Classifier prompt**:
```
Classify the following thought unit into exactly one type from the taxonomy.

Taxonomy:
[... full taxonomy above ...]

Thought unit: "{text}"
Context: This thought unit appears after: "{previous_tu_text}"

Output: {"type": "CODE", "confidence": 0.0-1.0, "rationale": "one sentence"}
```

**Validation**: 
- Human-classify 200 TUs (stratified by model and domain)
- Report macro-F1 per type and confusion matrix
- Target: macro-F1 ≥ 0.75

### 3.3 Stage C: Edge Linking — Building the Thought Graph

This is the hardest step and the key methodological contribution over LCoT2Tree (which only builds trees via sequential attachment).

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

#### Linking Algorithm

The linking proceeds in three passes:

**Pass 1: Sequential backbone**  
Connect every TU_i to TU_{i+1} with a `SEQ` edge. This creates the linear backbone.

**Pass 2: Local semantic linking (window = 5)**  
For each TU_i, examine TU_{i-5} through TU_{i-1}. Use an LLM classifier to determine if a non-sequential relationship exists:

```
Given two thought units from a reasoning trace, determine their relationship.

Thought unit A (earlier): "{tu_a_text}"
Thought unit B (later): "{tu_b_text}"

Possible relationships:
- ELAB: B deepens/specifies A
- BRCH: B starts a new direction from A's setup  
- BACK: B revises or abandons A
- CRIT: B challenges or critiques A
- SUPP: B provides evidence for A
- CONT: B contrasts with A (opposing perspective)
- NONE: No meaningful relationship beyond sequence

Output: {"relation": "CODE or NONE", "confidence": 0.0-1.0}
```

If relation ≠ NONE and confidence ≥ 0.7, add the edge. Remove the `SEQ` edge between adjacent nodes that now have a semantic edge.

**Pass 3: Long-range synthesis detection**  
This pass specifically handles cross-branch connections — the key feature directed graphs have over trees.

For each TU classified as `SYN` (synthesis):
1. Extract the key concepts/claims from the synthesis TU
2. Scan ALL prior TUs (not just the local window) for matching concepts
3. Use embedding similarity (threshold ≥ 0.75) + LLM verification:

```
Does this synthesis thought unit draw on the idea expressed in the candidate source?

Synthesis TU: "{syn_text}"
Candidate source TU: "{candidate_text}"

Output: {"draws_from": true/false, "confidence": 0.0-1.0}
```

If confirmed, add a `SYNT` edge from each verified source to the synthesis node.

**Pass 4: Backtracking detection**  
For each TU classified as `CRT` or containing backtracking markers ("Wait,", "Actually,", "Let me reconsider"):
1. Identify what is being revised/critiqued
2. Scan backward (up to full trace) for the target TU
3. Add `BACK` or `CRIT` edge from the new TU to the original

#### Thought Graph Output Schema
```json
{
  "trace_id": "uuid",
  "nodes": [
    {
      "tu_id": 0,
      "text": "Let me consider the ethical dimensions...",
      "type": "HYP",
      "type_family": "EXPLORATION",
      "start_char": 0,
      "end_char": 147
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "type": "ELAB",
      "confidence": 0.92
    },
    {
      "source": 3,
      "target": 7,
      "type": "SYNT",
      "confidence": 0.85
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
| **Branching factor** (BF) | Mean out-degree of EXPLORATION-family nodes via BRCH edges | How many distinct directions does the model explore? |
| **Unique perspective count** (UPC) | Number of connected components in the subgraph induced by EXPLORATION nodes (after removing SEQ edges) | How many truly independent angles does the model try? |
| **Domain spread** (DS) | For policy/ethics questions: number of distinct stakeholder perspectives mentioned (detected via NER + coreference) | Does the model consider multiple affected parties? |
| **First-idea diversity** (FID) | Cosine distance between embeddings of the first 3 EXPLORATION nodes | How different are the model's initial ideas from each other? |

### 4.2 Depth Metrics

*Note: Depth metrics are computed on the ELAB-only subgraph (which is naturally acyclic — elaboration doesn't loop back). This isolates the depth signal from iterative refinement cycles.*

| Metric | Formula | Intuition |
|---|---|---|
| **Max elaboration chain** (MEC) | Longest path following only ELAB edges from any single root | Deepest single-thread reasoning |
| **Mean branch depth** (MBD) | Average depth across all branches (rooted at EXPLORATION nodes) | Typical elaboration depth |
| **Specificity gradient** (SG) | Pearson correlation between node depth and entity density (entities per token) | Do ideas become more concrete as they're developed? |
| **Reasoning density** (RD) | Ratio of (JUS + IMP + CON nodes) to total nodes | What fraction of thinking is logical reasoning vs. brainstorming? |

### 4.3 Structural Metrics

| Metric | Formula | Intuition |
|---|---|---|
| **Exploration-exploitation ratio** (EER) | (EXPLORATION nodes) / (ELABORATION nodes) | Breadth-first vs. depth-first style |
| **Backtracking rate** (BR) | Proportion of nodes with outgoing BACK edges | How often does the model revise itself? |
| **Cross-branch connectivity** (CBC) | Number of SYNT edges / number of branches | Does the model connect different threads? |
| **Convergence index** (CI) | (SYN nodes in last quartile) / (total SYN nodes) | Does the model synthesize at the end or throughout? |
| **Orphan ratio** (OR) | Proportion of EXPLORATION nodes with no ELAB children | How many ideas are raised but never developed? |
| **Graph density** (GD) | |E| / (|V| × (|V|-1)/2) | Overall connectivity of the thought graph |
| **Cycle count** (CC) | Number of distinct simple cycles (via networkx) | Iterative refinement / oscillation frequency |
| **Mean cycle length** (MCL) | Mean length of all simple cycles | Short (2-3) = local oscillation; long (5+) = large reasoning loops |

### 4.4 Metacognitive Metrics

| Metric | Formula | Intuition |
|---|---|---|
| **Self-reflection rate** (SRR) | Proportion of MET nodes | How often does the model monitor its own reasoning? |
| **Critique-to-hypothesis ratio** (CHR) | CRT nodes / HYP nodes | Does the model evaluate as much as it generates? |
| **Hedging density** (HD) | Proportion of TUs containing uncertainty markers ("might", "could", "it's possible") | How much epistemic humility? |
| **Perspective-taking** (PT) | Proportion of RFR nodes | How often does the model shift viewpoint? |

### 4.5 Efficiency Metrics

| Metric | Formula | Intuition |
|---|---|---|
| **Token-per-idea** (TPI) | Total tokens / UPC | Verbosity relative to idea generation |
| **Redundancy ratio** (RR) | Proportion of TU pairs within same trace with embedding similarity > 0.90 | How repetitive is the reasoning? |

### 4.6 Profile Vector

The full profile vector for a model M on domain D:

```
V(M, D) = [BF, UPC, DS, FID, MEC, MBD, SG, RD, EER, BR, CBC, CI, OR, GD, CC, MCL, SRR, CHR, HD, PT, TPI, RR]
```

**22 dimensions**, computed as means over all traces of model M in domain D.

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

- Parser quality is LLM-dependent (mitigated by validation + gold set)
- Open-ended questions have no answer quality signal (by design — this is the point)
- Models without exposed CoT (e.g., GPT-4o) cannot be evaluated
- Profile stability may vary with prompt sensitivity
- Future: longitudinal profiling (does a model's thinking style change across versions?)
- Future: prescriptive application (route questions to the model with the best-matched thinking style)
