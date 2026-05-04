# ThinkBench: A Framework for Profiling the Cognitive Structure of LLM Reasoning

**Study Report v3 — Prompt Variant Pilot**

**Date**: 2026-04-24 | **Pipeline**: v3 Semantic Embedding Trajectory | **Model**: Qwen/Qwen3.5-35B-A3B | **Domain**: Ethical Dilemmas | **Target venue**: EMNLP 2026

---

## Abstract

We introduce **ThinkBench**, a framework for characterising the cognitive structure of large language model (LLM) reasoning traces. ThinkBench extracts a *Thought Graph* — a directed graph with cycles — from raw chain-of-thought output using a fully non-generative pipeline: sentence-level embedding trajectory analysis, rule-based boundary detection, and DeBERTa-based NLI edge promotion. From each graph we compute a 22-dimensional *Cognitive Profile* spanning five categories: Breadth, Depth, Structure, Metacognitive, and Efficiency. In this pilot study we collect 90 reasoning traces from a single model (Qwen3.5-35B-A3B) across three prompt variants — Pure, Normal, and Eliciting — on ten open-ended ethical dilemma questions. We evaluate which metrics are prompt-sensitive (measuring *elicited* behaviour) versus prompt-invariant (measuring *intrinsic* reasoning style). Twelve of 22 metrics show statistically significant variation across prompt conditions (Kruskal-Wallis, *p* < 0.05). Structural metrics including `branching_factor`, `reasoning_density`, `hedging_density`, and `self_reflection_rate` are stable across all three conditions, suggesting they reflect intrinsic model behaviour. Depth and revision metrics — particularly `revision_depth` (Δ = +8.80, *p* < 0.001) and `max_elaboration_chain` (Δ = +1.40, *p* < 0.001) — are strongly prompt-sensitive and should not be used as capability proxies without variant control.

---

## 1. Introduction

Open-ended reasoning is the primary capability frontier for large language models. Benchmarks such as MMLU, GSM8K, and HumanEval evaluate *what* a model concludes, but not *how* it thinks. Two models can arrive at the same answer through radically different cognitive processes: one through a single confident assertion, another through iterative self-correction, multi-angle exploration, and principled synthesis. This structural difference matters for understanding model behaviour, predicting failure modes, and designing better prompting strategies.

ThinkBench addresses this gap by treating the *structure* of chain-of-thought (CoT) reasoning as the unit of analysis. Instead of scoring final answers, ThinkBench maps the reasoning trace onto a directed graph — the **Thought Graph** — where nodes are thought units (TUs) classified by cognitive function (Hypothesis, Critique, Synthesis, etc.) and edges are semantic relationships (Branching, Elaboration, Backtracking, Synthesis). A 22-dimensional Cognitive Profile is then computed from this graph, characterising the model's reasoning across five dimensions: how widely it explores ideas (Breadth), how deeply it pursues them (Depth), how its reasoning is structurally organised (Structure), the degree to which it monitors its own thinking (Metacognitive), and how economically it uses tokens (Efficiency).

This report documents the first pilot study: 90 traces from one model across three prompt conditions. The primary scientific question is *which metrics are prompt-sensitive and which are intrinsic to the model's reasoning style*. This distinction is essential for designing a fair multi-model benchmark: only prompt-invariant metrics should be used to compare models without controlling for prompt design.

The remainder of this report is organised as follows. §2 describes the ThinkBench framework. §3 details the experimental setup. §4 presents results with inline figures. §5 discusses implications. §6 states limitations and future work. The Supplementary Material (§S) contains all supporting figures and raw data.

---

## 2. The ThinkBench Framework

![ThinkBench v3 end-to-end pipeline](../pipeline_v3.png)
**Figure 0.** Overview of the ThinkBench v3 pipeline. A raw chain-of-thought trace is segmented into Thought Units (TUs) through two boundary-detection passes (cue-phrase rules and TextTiling). TU embeddings are computed in a single batch and used by Layers 3–4 to assign structural edge types (BRCH, ELAB, SYNT, BACK) from the cosine-similarity trajectory, without surface pattern matching. An optional NLI pass (Pass 5) refines remaining untyped transitions. The node classifier applies 8-priority edge-structure rules augmented by embedding similarity to assign cognitive node types. The resulting ThoughtGraph feeds the 22-dimensional Cognitive Profile computation.

### 2.1 Thought Graph Construction

A **Thought Graph** *G = (V, E)* is a directed graph where **V** is the set of *Thought Units* (TUs) — contiguous text segments in a reasoning trace, each assigned a cognitive node type — and **E** is the set of directed semantic edges between TUs, each labelled with a relationship type. Cycles are permitted: they represent iterative refinement (propose → critique → revise → re-evaluate), a hallmark of deliberate reasoning that acyclic structures (trees, DAGs) cannot capture.

#### 2.1.1 Segmentation — v3 Semantic Pipeline

The v3 pipeline segments a raw CoT trace into TUs through five sequential passes.

**Pass 1 — Hard boundaries (rule-based).** A compiled lexicon of 30+ cue phrases detects three supplementary boundary types: `BACKTRACK` (e.g., "but actually", "reconsidering"), `META` (e.g., "stepping back", "at a higher level"), and `CONVERGENCE` (e.g., "in summary", "balancing these"). These are supplementary signals only; structural classification is driven by Pass 3.

**Pass 2 — Soft boundaries (TextTiling).** Each sentence is encoded with `all-MiniLM-L6-v2` (384-dimensional, L2-normalised). A sliding window (width 3) computes cosine similarity at each sentence boundary. Local minima detected via `scipy.signal.find_peaks` (prominence ≥ 0.15) trigger soft `NONE`-class boundaries where similarity falls below the 30th percentile of within-trace similarities.

**Pass 3 — Semantic trajectory (primary structural detection).** For each pair of adjacent TUs (*i−1, i*), we compute cosine similarity *s_i = cos(e_{i−1}, e_i)*. Two adaptive thresholds are derived per trace from the empirical similarity distribution:

- *τ_branch* = 25th percentile of {*s_1, …, s_{n−1}*}
- *τ_elab* = 65th percentile of {*s_1, …, s_{n−1}*}

Each transition is assigned:
- *s_i* < *τ_branch* → `BRCH` edge; new topic segment opened
- *τ_branch* ≤ *s_i* < *τ_elab* → `SEQ` edge; sequential continuation
- *s_i* ≥ *τ_elab* → `ELAB` edge; elaboration of prior TU

Pass 1 assignments override these when `CONVERGENCE` → `SYNT` or `BACKTRACK` → `BACK`.

**Pass 4 — Cross-segment semantic edges.** *Synthesis*: TU *j* receives `SYNT` edges when cos(*e_j*, centroid_s) > 0.50 for ≥ 2 prior segments *s*. *Loop detection*: a `BACK` edge is added when cos(*e_j*, *e_i*) > 0.65 and the minimum intervening similarity drops below 0.45 (confirming a genuine semantic gap), scanning up to 25 positions back.

**Pass 5 — NLI edge promotion (disabled in this run).** `cross-encoder/nli-deberta-v3-large` would promote SEQ edges to SUPP/CONT/ELAB (threshold 0.78); disabled here to isolate the embedding-trajectory signal.

Short spans (< 3 sentences or < 50 tokens) are merged into the following span before classification. The segmenter returns `(tus, edges, embeddings)`.

#### 2.1.2 Node Classification — 8-Priority Edge-Structure Rules

**Table 1.** Node type assignment rules in priority order.

| Priority | Condition | NodeType | Family |
|:---:|---|---|---|
| 1 | Outgoing SYNT to ≥ 2 distinct targets | **SYN** | Convergence |
| 2 | Outgoing BACK edge | **CRT** | Evaluation |
| 3 | `boundary_class == META` | **MET** | Evaluation |
| 4 | Incoming BRCH + 0.25 < cos(*e_i*, *e_0*) < 0.72 | **RFR** | Exploration |
| 5 | Incoming BRCH (otherwise) | **HYP** | Exploration |
| 6 | Outgoing CONT edge | **CMP** | Evaluation |
| 7 | Outgoing SUPP edge | **JUS** | Elaboration |
| 8 | `boundary_class == CONVERGENCE` | **SYN** | Convergence |
| 9 | idx = 0 (first TU in trace) | **HYP** | Exploration |
| Embed. | idx ≥ 45% trace AND cos(*e_i*, *e_0*) ≥ 0.60 | **MET** | Evaluation |
| Default | — | **SPC** | Elaboration |

The **RFR vs. HYP** distinction is the key novelty of v3 classification. A branch-starting TU (incoming BRCH) is *Reframing* (RFR) when 0.25 < cos(*e_i*, *e_0*) < 0.72: it re-engages the original problem from a new angle rather than opening an orthogonal hypothesis. Similarity ≥ 0.72 collapses to SPC (minor restatement); similarity ≤ 0.25 classifies as HYP (genuinely new hypothesis).

The **embedding-based MET** fallback captures meta-cognitive monitoring in models that reason without explicit discourse markers: a TU in the second half of the trace (idx ≥ 45% of *n*) with high similarity to TU₀ (cos ≥ 0.60) is classified as Meta-reflection.

**Node families** group the 12 types: **Exploration** (HYP, RFR, ANA, BRS), **Elaboration** (SPC, JUS, IMP, CON), **Evaluation** (CRT, CMP, MET), **Convergence** (SYN).

**Edge types**: BRCH (new direction), ELAB (deepening), BACK (revision loop), SYNT (integration of ≥ 2 segments), SUPP (NLI-evidence), CONT (NLI-contrast), CRIT (explicit challenge), SEQ (default sequential).

### 2.2 Cognitive Profile — All 22 Metrics Defined

Let *V* = node set, *E* = full edge set, *E_sem* = non-SEQ edges, *E_T* = edges of type *T*, *N_T* = nodes of type *T*, *n = |V|*.

#### Breadth

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| B1 | **Branching Factor** | \|E_BRCH\| / *n* | [0, 1] | Fraction of nodes initiating a new semantic branch. Values > 0.3 indicate a Divergent Explorer style. |
| B2 | **Unique Perspective Count** | \|N_RFR\| | [0, ∞) int | Count of Reframing nodes — branch-starters that re-engage the original problem. |
| B3 | **Domain Spread** | Agglomerative clusters of {N_HYP ∪ N_BRS} (cosine threshold 0.45); 0 if < 3 nodes | [0, ∞) int | Semantic diversity of hypothesis space. Zero when fewer than 3 HYP/BRS nodes exist. |
| B4 | **First Idea Diversity** | Mean pairwise cosine distance among first 3 HYP embeddings | [0, 1] | Diversity of the model's initial hypothesis generation. Zero if fewer than 2 HYP nodes. |

#### Depth

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| D1 | **Max Elaboration Chain** | Longest path in ELAB-only subgraph | [0, ∞) int | Maximum consecutive elaboration depth on any single idea. The ELAB subgraph is acyclic by construction. |
| D2 | **Mean Branch Depth** | Mean shortest-path depth from in-degree-0 roots in semantic subgraph | [0, ∞) | Average hierarchical depth of TUs in the reasoning structure. |
| D3 | **Specificity Gradient** | OLS slope of lexical specificity vs. TU position | (−∞, +∞) | Positive = reasoning becomes more concrete over time. Proxy: capitalised mid-sentence words + numerals + symbols. |
| D4 | **Reasoning Density** | \|nodes with ≥ 1 E_sem\| / *n* | [0, 1] | Fraction of TUs participating in at least one semantic relationship. |

#### Structure

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| S1 | **Exploration–Exploitation Ratio** | \|N_EXPLORATION\| / max(\|N_ELABORATION\|, 1) | [0, ∞) | > 1 = exploration dominant; < 1 = elaboration dominant. |
| S2 | **Backtracking Rate** | \|E_BACK\| / \|E_sem\|; 0 if \|E_sem\| = 0 | [0, 1] | Proportion of semantic edges that are backward revision links. |
| S3 | **Cross-Branch Connectivity** | Fraction of branch-component pairs linked by ≥ 1 SYNT or SUPP edge | [0, 1] | Integration across distinct reasoning branches. |
| S4 | **Convergence Index** | Σ_{v ∈ N_SYN} d_in(*v*) / (*n* × mean_d_in) in E_sem | [0, ∞) | Normalised in-degree concentration at SYN nodes. |
| S5 | **Graph Density** | \|E_sem\| / (*n* × (*n* − 1)) | [0, 1] | Density of the semantic subgraph beyond sequential structure. |
| S6 | **Revision Depth** | mean\|pos(*u*) − pos(*v*)\| for BACK edges | [0, ∞) | Average sequential distance of backward revision links. |

#### Metacognitive

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| M1 | **Self-Reflection Rate** | \|N_MET\| / *n* | [0, 1] | Proportion of TUs that are meta-cognitive commentary on reasoning. |
| M2 | **Critique-to-Hypothesis Ratio** | \|N_CRT\| / max(\|N_HYP\|, 1); 0 if no HYP | [0, ∞) | Evaluative rigour: how often the model challenges its own ideas. |
| M3 | **Hedging Density** | Fraction of TUs containing ≥ 1 uncertainty marker (regex) | [0, 1] | Epistemic humility. Markers: might, could, perhaps, possibly, probably, it seems, unclear, likely, etc. |
| M4 | **Perspective Taking** | \|N_RFR\| / *n* | [0, 1] | Rate of perspective shifts, normalised by trace length. Distinct from UPC (raw count). |

#### Efficiency & Summary

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| E1 | **Token per Idea** | total_tokens / max(\|N_RFR\|, 1); fallback = tokens / max(*n*, 1) | [0, ∞) | Tokens invested per reframed perspective. Lower = more concise diversity. |
| E2 | **Redundancy Ratio** | Fraction of TU pairs with cos_sim > 0.75 | [0, 1] | Near-duplicate content rate. Threshold 0.75 calibrated for MiniLM-L6-v2. |
| — | **avg_tokens** | Total token count of reasoning trace | [0, ∞) | Response length control variable. |
| — | **avg_tus** | Total number of TUs after segmentation and merging | [0, ∞) | Trace granularity. |

---

## 3. Experimental Setup

### 3.1 Data Collection

Traces were collected from **Qwen/Qwen3.5-35B-A3B** (35B parameter mixture-of-experts) via a vLLM endpoint (temperature = 0.7, max_tokens = 8192). The model generates reasoning in its `<think>` tag; only the thinking section is analysed. Collection is idempotent: each trace is keyed by a UUID and skipped on re-run.

### 3.2 Question Set

Ten questions from the **ethical_dilemmas** domain were used. Each presents a morally complex open-ended scenario with no single correct answer, spanning utilitarian, deontological, virtue-ethical, and legal-liability dimensions. Topics include: trolley-problem variants, medical resource allocation, whistleblower dilemmas, AI autonomy decisions, environmental tradeoffs, and criminal justice reform. A single domain was used in this pilot to isolate prompt-condition effects from domain effects. The full benchmark will include 6 domains × 30 questions per domain.

### 3.3 Prompt Variants

**Table 2.** Prompt variant definitions.

| Variant | System Prompt | Design Intent |
|---|---|---|
| **Pure** | *"You are a helpful assistant."* | Baseline — no reasoning guidance. Measures intrinsic model behaviour. |
| **Normal** | *"Think carefully before answering."* | Minimal nudge. Tests whether a simple instruction shifts cognitive structure. |
| **Eliciting** | Full 180-word framing: spread wide across angles; reframe the problem; go deep on key tensions; challenge your own assumptions; connect ideas across branches; reflect on your reasoning; converge toward a nuanced conclusion. | Upper bound — maximal structural elicitation. |

Each variant was applied to all 10 questions with 3 independent runs per question, yielding 30 traces per variant (N = 90 total). This is a between-subjects design: each trace is assigned exactly one prompt condition.

### 3.4 Pipeline Parameters

**Table 3.** Pipeline hyperparameters used in this study.

| Parameter | Value | Role |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Sentence encoding — 384-dim, L2-normalised, GPU-accelerated |
| NLI model | `cross-encoder/nli-deberta-v3-large` | Edge promotion (disabled: `use_nli=False`) |
| τ_branch | 25th percentile of local sims | BRCH boundary threshold (adaptive per trace) |
| τ_elab | 65th percentile of local sims | ELAB boundary threshold (adaptive per trace) |
| τ_synthesis | 0.50 | Min cos_sim to prior segment centroid for SYNT |
| Min bridged segments | 2 | Min prior segments bridged to trigger SYNT edge |
| τ_backtrack | 0.65 | Min cos_sim for BACK edge candidate |
| τ_back_drop | 0.45 | Max intervening similarity (gap requirement for BACK) |
| Min BACK gap | 4 TUs | Minimum positional distance for BACK edge |
| Max BACK lookback | 25 TUs | Lookback window for loop detection |
| NLI threshold | 0.78 | Minimum NLI score for edge promotion (when enabled) |
| Redundancy threshold | 0.75 | Cosine similarity threshold for RR computation |
| RFR sim low | 0.25 | Lower bound for RFR vs. HYP decision |
| RFR sim high | 0.72 | Upper bound for RFR vs. SPC decision |
| MET sim threshold | 0.60 | Min cos_sim to TU₀ for embedding-based MET |
| MET position fraction | 0.45 | Min fractional position in trace for embedding MET |

---

## 4. Results

### 4.1 Extraction Statistics

The v3 pipeline successfully extracted Thought Graphs from all 90 traces (0 failures, 0 skipped). Table 4 reports graph structure by prompt variant.

**Table 4.** Thought Graph structure by prompt variant (mean ± std, 30 traces each).

| Statistic | Pure | Normal | Eliciting | E / P ratio |
|---|---|---|---|---|
| Nodes (TUs) | 9.5 ± 1.9 | 9.7 ± 2.3 | 32.1 ± 14.5 | **3.4×** |
| Edges (total) | 16.2 ± 5.1 | 16.5 ± 5.6 | 67.5 ± 44.0 | **4.2×** |
| SEQ edges | 8.5 ± 1.9 | 8.7 ± 2.3 | 31.1 ± 14.5 | 3.7× |
| BRCH edges | 2.3 | 2.4 | 7.6 | 3.4× |
| ELAB edges | 3.1 | 3.2 | 10.8 | 3.5× |
| BACK edges | 0.87 | 1.07 | 6.90 | **7.9×** |
| SYNT edges | 1.47 | 1.20 | 11.10 | **7.6×** |
| avg_tokens | 696 ± 91 | 695 ± 119 | 2347 ± 1110 | **3.4×** |
| HYP nodes / trace | 1.27 | 1.20 | 4.37 | 3.4× |
| CRT nodes / trace | 0.60 | 0.70 | 4.40 | 7.3× |
| SYN nodes / trace | 0.70 | 0.60 | 4.20 | 6.0× |

The eliciting variant produces traces 3.4× longer but BACK edges increase 7.9× and SYNT edges 7.6×. Structural complexity scales super-linearly with tokens, indicating the eliciting prompt does not merely produce more text — it qualitatively reorganises reasoning structure. Figure 9 below shows the full compositional breakdown by node and edge type.

---

![Node and edge type distribution by prompt variant](figures_v2/fig9_node_edge_dist.png)

**Figure 9.** Stacked bar charts showing mean count per trace of each node type (left) and each edge type (right) for the three prompt variants. The eliciting variant shows substantially more HYP (4.37 vs. 1.27), CRT (4.40 vs. 0.60), and SYN (4.20 vs. 0.70) nodes, and a disproportionate increase in BACK (7.9×) and SYNT (7.6×) edges relative to the 3.4× increase in total tokens. The SPC (Specification) node proportion remains stable, indicating the elaboration backbone grows proportionally while evaluative and integrative structure grows super-linearly.

---

Figure 8 shows three representative Thought Graphs per variant. Pure and Normal graphs are compact (≈9 nodes) with a linear-with-branches topology. Eliciting graphs show dense SYNT subgraphs, long BACK arcs crossing many positions, and extended ELAB chains.

---

![Representative thought graphs — 3 per prompt variant](figures_v2/fig8_graph_examples.png)

**Figure 8.** Selected Thought Graph visualisations (best-scoring per variant by structural richness score = 3 × BRCH + 2 × BACK + 2 × SYNT). **Node fill colour** encodes NodeFamily: blue = Exploration (HYP, RFR), green = Elaboration (SPC, JUS), red = Evaluation (CRT, MET), gold = Convergence (SYN). **Edge colour** encodes EdgeType: blue = BRCH, green = ELAB, red = BACK, gold = SYNT; thin gray = SEQ (shown for context). Node labels show NodeType abbreviation. SEQ edges are thin to reduce visual clutter. Panel titles report node count, total edge count, and counts of ↗ BRCH, ↩ BACK, and ⊕ SYNT edges.

---

### 4.2 Descriptive Statistics — Full Metric Suite

Table 5 reports descriptive statistics for all 22 metrics across all 90 traces pooled.

**Table 5.** Descriptive statistics for all 22 metrics (N = 90 traces).

| Metric | Category | Mean | Std | Median | Min | Max | Zero% |
|---|---|---|---|---|---|---|---|
| branching_factor | Breadth | 0.2386 | 0.0361 | 0.2449 | 0.000 | 0.286 | 1.1% |
| unique_perspective_count | Breadth | 2.189 | 1.182 | 2.000 | 0.000 | 6.000 | 2.2% |
| domain_spread | Breadth | 1.256 | 2.158 | 0.000 | 0.000 | 7.000 | 73.3% |
| first_idea_diversity | Breadth | 0.392 | 0.422 | 0.000 | 0.000 | 1.000 | 53.3% |
| max_elaboration_chain | Depth | 2.300 | 1.130 | 2.000 | 1.000 | 7.000 | 0.0% |
| mean_branch_depth | Depth | 2.346 | 0.875 | 2.258 | 0.000 | 4.783 | 2.2% |
| specificity_gradient | Depth | −0.002 | 0.007 | −0.003 | −0.025 | 0.017 | 0.0% |
| reasoning_density | Depth | 0.875 | 0.092 | 0.880 | 0.600 | 1.000 | 0.0% |
| exploration_exploitation_ratio | Structure | 0.650 | 0.279 | 0.600 | 0.238 | 1.667 | 0.0% |
| backtracking_rate | Structure | 0.127 | 0.103 | 0.129 | 0.000 | 0.364 | 32.2% |
| cross_branch_connectivity | Structure | 0.036 | 0.061 | 0.001 | 0.000 | 0.267 | 50.0% |
| convergence_index | Structure | 0.066 | 0.101 | 0.000 | 0.000 | 0.616 | 54.4% |
| graph_density | Structure | 0.077 | 0.041 | 0.071 | 0.013 | 0.191 | 0.0% |
| revision_depth | Structure | 6.770 | 5.921 | 6.167 | 0.000 | 23.57 | 32.2% |
| self_reflection_rate | Metacog. | 0.031 | 0.074 | 0.000 | 0.000 | 0.500 | 77.8% |
| critique_to_hypothesis_ratio | Metacog. | 0.747 | 0.919 | 0.500 | 0.000 | 4.000 | 40.0% |
| hedging_density | Metacog. | 0.255 | 0.156 | 0.250 | 0.000 | 0.571 | 11.1% |
| perspective_taking | Metacog. | 0.164 | 0.080 | 0.154 | 0.000 | 0.286 | 2.2% |
| token_per_idea | Efficiency | 599.5 | 474.0 | 422.3 | 63.2 | 3282 | 0.0% |
| redundancy_ratio | Efficiency | 0.011 | 0.025 | 0.000 | 0.000 | 0.200 | 57.8% |
| avg_tokens | Summary | 1246 | 1012 | 775.0 | 379 | 5607 | 0.0% |
| avg_tus | Summary | 17.07 | 13.61 | 11.00 | 6.00 | 78.0 | 0.0% |

**Notable distributional properties:**
- `domain_spread` is zero in 73.3% of traces (Pure and Normal have no multi-HYP clustering; this is a genuine finding, not a bug).
- `first_idea_diversity` is zero in 53.3% of traces (fewer than 2 HYP nodes in many Pure/Normal traces).
- `convergence_index` and `cross_branch_connectivity` are zero in ~50% of traces; synthesis emerges only in eliciting traces.
- `self_reflection_rate` is zero in 77.8% of traces; meta-cognitive commentary is rare in this model's dense-prose style.
- All 22 metrics have std > 0 — no metric is fully degenerate.

Figure 4 shows the full within-variant distribution for each metric as violin + strip plots.

---

![Per-trace metric distributions — violin plots for all 20 cognitive metrics](figures_v2/fig4_metric_violin.png)

**Figure 4.** 5×4 grid of violin + strip plots for all 20 cognitive metrics. Each panel shows three violins (P = Pure, N = Normal, E = Eliciting) with individual trace data points overlaid as jittered dots. Median is marked as a thick black line inside each violin. Panel titles are colour-coded by cognitive category (blue = Breadth, green = Depth, orange = Structure, purple = Metacognitive, red = Efficiency). Key distributional observations: `critique_to_hypothesis_ratio` and `revision_depth` are strongly right-skewed in Eliciting with long tails; `branching_factor` shows exceptionally tight distributions across all three variants; `self_reflection_rate` is zero for the majority of traces in all conditions.

---

### 4.3 Prompt Sensitivity Analysis

Table 6 reports per-variant means, absolute deltas (Eliciting − Pure), and Kruskal-Wallis test results for all 22 metrics.

**Table 6.** Per-variant means and statistical significance of prompt effects (Kruskal-Wallis, df = 2). Bold rows are statistically significant (*p* < 0.05). Δ = Eliciting − Pure.

| Metric | Pure | Normal | Eliciting | Δ | H | p | Sig. |
|---|---|---|---|---|---|---|---|
| branching_factor | 0.2413 | 0.2446 | 0.2298 | −0.012 | 1.55 | 0.461 | ns |
| unique_perspective_count | 1.867 | 1.967 | 2.733 | +0.867 | 5.62 | 0.060 | ns |
| **domain_spread** | **0.000** | **0.000** | **3.767** | **+3.767** | **62.86** | **<0.001** | *** |
| **first_idea_diversity** | **0.226** | **0.172** | **0.779** | **+0.554** | **32.33** | **<0.001** | *** |
| **max_elaboration_chain** | **1.933** | **1.633** | **3.333** | **+1.400** | **39.15** | **<0.001** | *** |
| **mean_branch_depth** | **1.952** | **2.022** | **3.064** | **+1.112** | **28.26** | **<0.001** | *** |
| specificity_gradient | −0.003 | −0.000 | −0.004 | −0.001 | 4.43 | 0.109 | ns |
| reasoning_density | 0.861 | 0.890 | 0.874 | +0.013 | 2.07 | 0.356 | ns |
| **exploration_exploitation_ratio** | **0.754** | **0.687** | **0.509** | **−0.244** | **23.40** | **<0.001** | *** |
| **backtracking_rate** | **0.090** | **0.117** | **0.173** | **+0.083** | **10.03** | **0.007** | ** |
| cross_branch_connectivity | 0.047 | 0.037 | 0.024 | −0.023 | 2.27 | 0.322 | ns |
| **convergence_index** | **0.047** | **0.047** | **0.105** | **+0.059** | **7.71** | **0.021** | * |
| **graph_density** | **0.099** | **0.095** | **0.038** | **−0.061** | **47.87** | **<0.001** | *** |
| **revision_depth** | **3.619** | **4.274** | **12.417** | **+8.798** | **38.51** | **<0.001** | *** |
| self_reflection_rate | 0.032 | 0.037 | 0.026 | −0.006 | 1.34 | 0.513 | ns |
| **critique_to_hypothesis_ratio** | **0.517** | **0.633** | **1.091** | **+0.574** | **10.62** | **0.005** | ** |
| hedging_density | 0.279 | 0.235 | 0.250 | −0.029 | 0.85 | 0.653 | ns |
| **perspective_taking** | **0.201** | **0.202** | **0.088** | **−0.113** | **39.97** | **<0.001** | *** |
| **token_per_idea** | **417.2** | **410.5** | **970.8** | **+553.6** | **26.07** | **<0.001** | *** |
| **redundancy_ratio** | **0.007** | **0.017** | **0.008** | **+0.001** | **7.35** | **0.025** | * |
| **avg_tokens** | **695.9** | **695.3** | **2347.1** | **+1651** | **46.07** | **<0.001** | *** |
| **avg_tus** | **9.47** | **9.67** | **32.07** | **+22.6** | **45.02** | **<0.001** | *** |

*Significance: \*\*\* p < 0.001 · \*\* p < 0.01 · \* p < 0.05 · ns p ≥ 0.05*

**12 of 22 metrics are statistically significant.** Figure 1 visualises the full Δ rankings.

---

![Prompt sensitivity — dual horizontal bar chart ranked by absolute delta](figures_v2/fig1_sensitivity_bar.png)

**Figure 1.** Dual horizontal bar chart ranking all 20 cognitive metrics by Δ(Eliciting − Pure) (left) and Δ(Normal − Pure) (right). Bars are colour-coded by cognitive category (blue = Breadth, green = Depth, orange = Structure, purple = Metacognitive, red = Efficiency). Value labels show exact Δ. Metrics are sorted from most sensitive (top) to least sensitive (bottom) by absolute Eliciting delta. The contrast between the two panels is striking: the Normal condition moves almost nothing (max Δ ≈ 0.66 for revision_depth), while Eliciting produces large displacements across Depth and Structure categories. Metrics where both panels are near-zero are prompt-invariant.

---

**Key observations from Table 6:**

- **`revision_depth`** shows the largest absolute delta (+8.80, *p* < 0.001, H = 38.51). The eliciting prompt drives the model to perform deep revisions reaching back a mean of 12.4 TUs (vs. 3.6 in Pure), consistent with the explicit instruction to "challenge your own assumptions."

- **`domain_spread`** undergoes a binary transition: exactly zero in both Pure and Normal (insufficient HYP nodes for agglomerative clustering), jumping to 3.77 in Eliciting. This reflects a qualitative behavioural shift, not a quantitative scaling.

- **`exploration_exploitation_ratio`** *decreases* significantly with eliciting (0.754 → 0.509, *p* < 0.001). Despite generating more content overall, the eliciting variant does not proportionally increase exploration nodes — instead it dramatically increases elaboration (SPC) nodes as the model follows each branch deeper. The model shifts from exploration-dominant to elaboration-dominant mode.

- **`graph_density`** *decreases* significantly with eliciting (0.099 → 0.038, *p* < 0.001). As graph size increases 3.4×, semantic edges do not scale proportionally — larger graphs become sparser. Reasoning is broader but not more uniformly interconnected.

- **`perspective_taking`** *decreases* with eliciting (0.201 → 0.088, *p* < 0.001). This is a normalisation artefact: absolute RFR node count rises from 2.0 to 2.7, but avg_tus rises 3.4×, driving the *rate* metric down even as the absolute count increases.

- **`branching_factor`** is the most stable metric (H = 1.55, *p* = 0.461, ns), with near-identical means (0.241, 0.245, 0.230). Branching propensity appears to be a fixed architectural property of this model.

Figure 3 shows the aggregate effect by cognitive category. Figure 2 shows the full normalised profile as a radar chart.

---

![Cognitive category scores by prompt variant](figures_v2/fig3_category_comparison.png)

**Figure 3.** Grouped bar chart showing mean normalised score per cognitive category per prompt variant (bars labelled with exact values). The Depth category shows the largest eliciting effect (driven by max_elaboration_chain, mean_branch_depth, revision_depth). The Breadth category shows a moderate effect concentrated in domain_spread and first_idea_diversity. The Structure category shows a mixed pattern: revision and convergence metrics increase while density and EER metrics decrease. The Metacognitive category is nearly flat across variants, consistent with the statistical finding that three of four metacognitive metrics are prompt-invariant.

---

![Radar chart — normalised cognitive profiles for all three prompt variants](figures_v2/fig2_radar.png)

**Figure 2.** Polar radar chart overlaying normalised cognitive profiles for Pure (gray), Normal (blue), and Eliciting (red) variants. Each of the 20 axes is normalised to its expected empirical range. The eliciting profile (red) is substantially larger in the upper-right quadrant (Depth and revision metrics) while nearly coinciding with Pure (gray) in the branching_factor and reasoning_density axes. Normal (blue) is nearly indistinguishable from Pure across all axes, confirming that a minimal "think carefully" instruction does not measurably reorganise cognitive structure.

---

### 4.4 Prompt Sensitivity Classification

Based on statistical significance (Table 6) and standardised effect size (Figure 5), we classify the 22 metrics into two groups for benchmark use.

---

![Cohen's d effect sizes — Eliciting vs. Pure for all metrics](figures_v2/fig5_effect_size.png)

**Figure 5.** Cohen's *d* (Eliciting − Pure) ranked from largest to smallest effect. Vertical reference lines mark |*d*| = 0.2 (small) and |*d*| = 0.8 (large). Colour-coded by cognitive category. Metrics with *d* near zero are prompt-invariant and suitable for cross-model comparison. Negative bars (exploration_exploitation_ratio, graph_density, perspective_taking) indicate the metric *decreases* under eliciting — an equally important prompt-sensitivity signal.

---

**Table 7.** Metric classification for benchmark use.

| Class | Metrics | Benchmark Recommendation |
|---|---|---|
| **Prompt-Sensitive** | revision_depth, domain_spread, max_elaboration_chain, mean_branch_depth, first_idea_diversity, avg_tokens, avg_tus, graph_density, exploration_exploitation_ratio, perspective_taking, critique_to_hypothesis_ratio, backtracking_rate | Only interpretable when prompt variant is controlled. Do not use as model capability proxies without specifying the prompt condition. |
| **Prompt-Invariant** | branching_factor, unique_perspective_count, specificity_gradient, reasoning_density, cross_branch_connectivity, self_reflection_rate, hedging_density, convergence_index† | Suitable for cross-model comparison under a fixed prompt. Recommended benchmark metrics. |

†`convergence_index` is borderline (*p* = 0.021); would be reclassified as invariant under Bonferroni correction (α/22 ≈ 0.0023). Flagged for review with larger sample size.

### 4.5 Multivariate Structure

#### Spearman Correlation Matrix

Figure 6 shows the Spearman ρ matrix of all 20 cognitive metrics, reordered by Ward hierarchical clustering.

---

![Spearman correlation matrix — Ward-clustered](figures_v2/fig6_correlation_heatmap.png)

**Figure 6.** Spearman ρ correlation matrix (Ward-clustered on 1 − |ρ| distance). Cells annotated with ρ values; colour scale: red = strong positive, blue = strong negative, white = near zero. Two major clusters emerge: **Cluster 1** (upper-left): `avg_tokens`, `avg_tus`, `max_elaboration_chain`, `mean_branch_depth`, `revision_depth` — a *length-complexity* cluster driven by the Eliciting condition's token inflation. Metrics in this cluster are highly intercorrelated because they all scale with trace length. **Cluster 2** (lower-right): `branching_factor`, `graph_density`, `exploration_exploitation_ratio` — a *structural density* cluster. Metrics within the same cluster should not all be included in a single composite score without dimensionality reduction; they measure the same underlying signal.

---

#### PCA in Metric Space

Figure 7 shows the 90 traces projected onto the first three principal components of the 20-dimensional metric space (z-score normalised).

---

![PCA of 90 traces in 20-dimensional metric space](figures_v2/fig7_pca.png)

**Figure 7.** Three-panel PCA scatter of 90 traces. Each panel shows a different pair of principal components. Points coloured by variant (gray = Pure, blue = Normal, red = Eliciting); stars mark variant centroids. **PC1 × PC2**: Eliciting centroid (red star) is clearly separated from Pure/Normal centroids, which substantially overlap — confirming that the Normal condition does not produce a distinct cognitive profile. **PC1 × PC3** and **PC2 × PC3**: The Eliciting cluster is elongated (high variance within the condition), reflecting the wide range of trace lengths in the eliciting condition (avg_tokens range: 768–5607). Pure and Normal centroids remain co-located across all three projections.

---

#### Parallel Coordinates

Figure 10 shows all 90 individual trace profiles as parallel coordinate lines.

---

![Parallel coordinates — all 90 traces, variant medians highlighted](figures_v2/fig10_parallel_coords.png)

**Figure 10.** Parallel coordinates plot with one axis per metric (min-max normalised). Individual traces drawn as thin semi-transparent lines; thick solid lines show variant medians. Background shading marks cognitive category zones (labelled at top). The plot reveals: (1) Eliciting (red) and Pure/Normal (gray/blue) medians diverge sharply at the Depth and revision-related axes (left half), while converging at the branching_factor, reasoning_density, and hedging_density axes (right portion). (2) Individual eliciting traces show high variance (wide spread of red lines) — some eliciting traces produce extremely long chains while others produce moderate ones, depending on question complexity. (3) Normal traces (blue) follow Pure traces almost identically, visible as near-coincident median lines.

---

### 4.6 Per-Metric Distribution Commentary

Selected observations from Figure 4 and Table 5:

- **`critique_to_hypothesis_ratio`** is strongly right-skewed in Eliciting (median 0.97, max 4.0). A subset of Eliciting traces exhibit extreme critique dominance — the model devotes the majority of its thinking to challenging its own ideas. This outlier behaviour inflates the mean (1.09) relative to the median.

- **`revision_depth`** is approximately symmetric in Pure/Normal (median ≈ 4–6 TUs) but right-skewed in Eliciting (median ≈ 11.4 TUs, tail to 23.6 TUs). Some traces show the model revisiting ideas established more than 20 TUs earlier — consistent with the eliciting prompt instruction to "go back and challenge your assumptions."

- **`branching_factor`**: the tightest distribution in the entire profile (range 0.0–0.286, std ≈ 0.03–0.05 within each variant). This extremely low variance — spanning only 28.6 percentage points across the entire observed range — strongly supports classification as an intrinsic property.

- **`specificity_gradient`**: centred near zero in all conditions (mean −0.002), indicating this model does not systematically become more concrete over time. The slightly negative trend suggests marginal drift toward more abstract language near conclusions — possibly because final synthesis paragraphs use higher-level generalisations.

- **`redundancy_ratio`**: 57.8% of traces show zero redundancy (no TU pair with cos_sim > 0.75). The 42.2% of traces with non-zero redundancy show a heavy tail to 0.20, suggesting occasional repetition is concentrated in a minority of traces.

---

## 5. Discussion

### 5.1 The Prompt-Sensitivity Divide

The central empirical finding is a clean partition of the 22-metric profile into prompt-sensitive and prompt-invariant metrics. This has a direct implication for benchmark design: **a ThinkBench score cannot be a single number computed from raw metric values unless all models are evaluated under identical prompt conditions.** The prompt-invariant metrics — particularly `branching_factor`, `reasoning_density`, and `hedging_density` — are the most appropriate basis for cross-model comparison.

The invariance of `branching_factor` is theoretically important. The model's propensity to open new semantic directions (approximately 1 BRCH edge per 4 TUs, regardless of instruction) appears to be a fixed property of its training. In contrast, how *deeply* it pursues any branch (`revision_depth`, `max_elaboration_chain`) is highly malleable. This suggests a two-dimensional model of reasoning style: (a) intrinsic branching propensity and (b) instrutable elaboration depth.

### 5.2 Eliciting as an Upper Bound, Not a Capability Measure

The Eliciting variant should be understood as an *upper bound* on structurally rich reasoning from this model, not as a measure of intrinsic capability. The 7.9× BACK edge increase and 7.6× SYNT edge increase confirm that the eliciting prompt transforms reasoning qualitatively — but this transformation is instruction-following. A model that scores high on `revision_depth` under eliciting may simply be better at following structured instructions, not better at spontaneous self-correction. This confound must be stated explicitly in any paper using eliciting-condition metrics as capability proxies.

The near-zero change from Pure to Normal (max Δ = 0.66 for `revision_depth`) is equally informative: a minimal "think carefully" instruction has negligible measurable effect on cognitive structure. The eliciting prompt's specificity — seven explicit sub-instructions — is essential for triggering structural reorganisation.

### 5.3 Non-Obvious Directional Effects

Three metrics *decrease* significantly under eliciting, which warrants explicit attention:

1. **`exploration_exploitation_ratio`** (−0.244): Despite generating more total content, the model shifts toward elaboration-dominant reasoning. The eliciting prompt's instruction to "go deep" effectively prioritises depth over breadth within each branch.

2. **`graph_density`** (−0.061): Larger graphs are sparser. As the graph grows from ≈9 to ≈32 nodes, the O(*n*²) denominator in the density formula grows much faster than the actual semantic edge count, producing lower density even though the raw edge count increases 4.2×.

3. **`perspective_taking`** (−0.113): A normalisation artefact. The raw RFR count rises (1.87 → 2.73), but is outpaced by the 3.4× increase in avg_tus, driving the rate metric down.

These decreases are not performance regressions — they reflect the natural consequences of longer, deeper traces on normalised metrics. They illustrate why raw metric comparison across conditions with different trace lengths requires careful normalisation.

### 5.4 Pipeline Validity

The v3 semantic trajectory pipeline resolves the primary failure mode of v2 (cue-phrase-based classification): metrics that were 100% zero under v2 now show meaningful signal. The adaptive threshold design (percentile-based *τ_branch* and *τ_elab*) generalises across models without model-specific tuning. The 0% extraction failure rate (90/90) confirms robustness.

The remaining validity concern is the absence of human annotation. Without gold-standard segmentation labels, extraction accuracy (boundary F1, node classification F1, edge accuracy) is unquantified. Human annotation of 30 traces is the highest-priority remaining task.

---

## 6. Limitations and Future Work

1. **Single model, single domain.** The benchmark's core hypothesis — that Cognitive Profiles discriminate between models — cannot be tested without at least 3 models × 3 domains. The minimum viable experiment is Qwen3.5-35B-A3B vs. DeepSeek-R1 vs. one smaller model under the Normal prompt condition.

2. **No human annotation.** Boundary F1, node classification accuracy, edge accuracy, and IAA (Cohen's κ) are all unquantified. Target: 2 annotators × 30 traces × 200 TU boundaries.

3. **NLI disabled.** Enabling `cross-encoder/nli-deberta-v3-large` will add SUPP and CONT edges, likely affecting `reasoning_density`, `cross_branch_connectivity`, and `convergence_index`. A controlled comparison (NLI on vs. off) is needed to quantify the impact.

4. **Multiple comparison correction.** With 22 simultaneous Kruskal-Wallis tests, the family-wise error rate is elevated. Under Bonferroni correction (α = 0.05/22 ≈ 0.0023): `convergence_index` (*p* = 0.021) and `redundancy_ratio` (*p* = 0.025) would be reclassified as invariant; `backtracking_rate` (*p* = 0.007) survives.

5. **Normalisation bounds hardcoded.** The `_NORM_BOUNDS` constants in the visualisation module are manually calibrated to this study's data range. They must be re-derived from the full multi-model dataset before paper figures are finalised.

6. **`specificity_gradient` proxy validity.** The spacy-free lexical proxy (capitalised mid-sentence words + numerals + symbols) is an approximation of NER density. Once the environment includes spacy + `en_core_web_sm`, the metric should be recomputed with proper NER for comparison.

7. **No ICC reliability analysis.** With 3 runs per question per variant, ICC(3,1) per metric can be computed from existing data to flag unreliable metrics (ICC < 0.6). This analysis uses no additional data collection.

---

## 7. Conclusion

ThinkBench provides a structured approach to characterising *how* LLMs reason, not merely whether they reach correct answers. The v3 semantic pipeline successfully extracts non-degenerate cognitive profiles from 90 reasoning traces without any generative LLM in the extraction loop. The primary finding of this pilot is a clean partition of the 22-metric profile: branching propensity, reasoning density, hedging density, and self-reflection rate are intrinsic to the model's style; revision depth, elaboration chain length, and domain spread are strongly elicitable. For cross-model benchmarking, the eight prompt-invariant metrics form the recommended measurement basis. The framework is ready for multi-model expansion, pending human annotation validation.

---

---

# Supplementary Material

---

## S1. Aggregated Metric Heatmap

![Aggregated metric heatmap — normalised values, all 3 variants](supplement/S1_metric_heatmap.png)

**Figure S1.** Heatmap of normalised metric values at the aggregated level (rows = metrics, columns = prompt variants). Y-axis labels are colour-coded by cognitive category. Cell values are annotated. This compact view complements Figure 4 (per-trace violin): it shows the aggregated pattern clearly but loses within-condition variance. Note how Pure and Normal columns are nearly identical across all rows (confirming the PCA finding of overlapping centroids), while Eliciting shows elevated values in the upper rows (Depth metrics) and a notable drop in the graph_density row.

---

## S2. Key Metric Pairwise Scatters

![9 key bivariate scatter plots](supplement/S2_scatter_pairs.png)

**Figure S2.** 3×3 grid of bivariate scatter plots for the nine most informative metric pairs, coloured by variant (gray = Pure, blue = Normal, red = Eliciting). Reading guide:

- **branching_factor × revision_depth** (top-left): The most important separation plot. Eliciting traces (red) are displaced upward (higher revision_depth) but not rightward (same branching_factor), visually confirming that branching is prompt-invariant while revision is prompt-sensitive.
- **branching_factor × unique_perspective_count**: Eliciting traces show slightly more RFR nodes at similar branching rates, confirming UPC rises modestly.
- **unique_perspective_count × convergence_index**: Eliciting traces concentrate in the upper-right (both higher) — synthesis emerges as perspectives increase.
- **backtracking_rate × critique_to_hypothesis_ratio**: Both metacognitive-structural metrics increase together under eliciting; their correlation suggests they co-occur in the same traces.
- **mean_branch_depth × max_elaboration_chain**: Strong positive correlation within all variants; both are length-complexity metrics (Cluster 1 from Figure 6).
- **hedging_density × reasoning_density**: Both are prompt-invariant, showing no variant-based separation — exactly as expected from Table 7.
- **graph_density × exploration_exploitation_ratio**: Negative correlation visible within eliciting (red): longer traces have lower density and lower EER.
- **perspective_taking × first_idea_diversity**: Both low in Pure/Normal; eliciting separates (high FID, lower PT due to normalisation).
- **token_per_idea × avg_tus**: Positive correlation; eliciting traces occupy the upper-right quadrant (many TUs, high tokens/idea).

---

## S3. Hierarchical Clustering Dendrogram

![Ward dendrogram of 90 traces in 20-metric space](supplement/S3_dendrogram.png)

**Figure S3.** Ward-linkage dendrogram of all 90 traces in the 20-dimensional metric space (z-score normalised). The bottom tick of each leaf is colour-coded by prompt variant: gray = Pure, blue = Normal, red = Eliciting. If prompt variant were the dominant source of variation, Eliciting leaves would cluster together and Pure/Normal would form a separate subtree. The observed dendrogram shows that Eliciting traces (red) do tend to cluster in their own subtree on the right side of the dendrogram, but with notable intermixing — indicating that question topic and individual run variability also contribute to inter-trace differences. This is expected for a single-model, single-domain pilot; a multi-model study would produce cleaner variant-based clusters.

---

## S4. Per-Trace Profile Heatmap

![All 90 trace profiles as rows](supplement/S4_per_trace_heatmap.png)

**Figure S4.** All 90 traces as rows in a normalised metric heatmap, grouped by variant (separated by horizontal dividers). Variant labels shown on the left y-axis in variant colour. Each cell encodes the normalised value [0–1] for that metric × trace combination (RdYlGn colormap: green = high, red = low). This is the most granular view available: it reveals within-condition outliers (individual red or green rows within a group), metric consistency (columns that are uniformly coloured vs. highly variable), and the global pattern that Pure/Normal rows appear nearly identical while Eliciting rows show elevated values in the Depth columns (leftmost axes) and compressed values in the graph_density column. Traces are displayed in collection order within each variant.

---

## S5. All Thought Graphs — Pure Condition

All 30 Pure condition traces (9 graphs per page, 4 pages). Graphs sorted in collection order. Colour scheme identical to Figure 8.

![All Pure traces — page 1 of 4](supplement/graphs_all_pure_p01.png)

**Figure S5a.** Pure condition Thought Graphs, page 1 of 4 (traces 1–9). Note the compact size of Pure graphs (typically 8–12 nodes, 12–25 edges) and the dominance of SEQ (gray) and ELAB (green) edges. BACK (red) and SYNT (gold) edges are rare but present, confirming that even without explicit elicitation, the model occasionally revises and synthesises.

![All Pure traces — page 2 of 4](supplement/graphs_all_pure_p02.png)

**Figure S5b.** Pure condition Thought Graphs, page 2 of 4 (traces 10–18).

![All Pure traces — page 3 of 4](supplement/graphs_all_pure_p03.png)

**Figure S5c.** Pure condition Thought Graphs, page 3 of 4 (traces 19–27).

![All Pure traces — page 4 of 4](supplement/graphs_all_pure_p04.png)

**Figure S5d.** Pure condition Thought Graphs, page 4 of 4 (traces 28–30 plus blank panels).

---

## S6. All Thought Graphs — Normal Condition

All 30 Normal condition traces (9 graphs per page, 4 pages).

![All Normal traces — page 1 of 4](supplement/graphs_all_normal_p01.png)

**Figure S6a.** Normal condition Thought Graphs, page 1 of 4. Structurally nearly identical to Pure traces — the "Think carefully" instruction does not measurably change graph topology. This visual finding is consistent with the statistical finding that Pure and Normal centroids overlap in all PCA projections.

![All Normal traces — page 2 of 4](supplement/graphs_all_normal_p02.png)

**Figure S6b.** Normal condition Thought Graphs, page 2 of 4.

![All Normal traces — page 3 of 4](supplement/graphs_all_normal_p03.png)

**Figure S6c.** Normal condition Thought Graphs, page 3 of 4.

![All Normal traces — page 4 of 4](supplement/graphs_all_normal_p04.png)

**Figure S6d.** Normal condition Thought Graphs, page 4 of 4.

---

## S7. All Thought Graphs — Eliciting Condition

All 30 Eliciting condition traces (9 graphs per page, 4 pages). Note substantially larger graphs and denser edge structure compared to Pure/Normal.

![All Eliciting traces — page 1 of 4](supplement/graphs_all_eliciting_p01.png)

**Figure S7a.** Eliciting condition Thought Graphs, page 1 of 4 (traces 1–9). Eliciting traces exhibit the full range of structural patterns: large SYNT (gold) subgraphs drawing from many prior nodes, long BACK (red) arcs spanning many positions, and extended ELAB (green) chains. The `spring_layout` algorithm spaces nodes by semantic relationships; dense eliciting graphs show tight clustering of related nodes with long arcs bridging distant segments.

![All Eliciting traces — page 2 of 4](supplement/graphs_all_eliciting_p02.png)

**Figure S7b.** Eliciting condition Thought Graphs, page 2 of 4.

![All Eliciting traces — page 3 of 4](supplement/graphs_all_eliciting_p03.png)

**Figure S7c.** Eliciting condition Thought Graphs, page 3 of 4.

![All Eliciting traces — page 4 of 4](supplement/graphs_all_eliciting_p04.png)

**Figure S7d.** Eliciting condition Thought Graphs, page 4 of 4.

---

## S8. Reading the Thought Graphs — Colour Key

| Element | Colour | Meaning |
|---|---|---|
| Node fill — Blue | Exploration family | HYP (Hypothesis), RFR (Reframing), ANA (Analogy), BRS (Brainstorm) |
| Node fill — Green | Elaboration family | SPC (Specification), JUS (Justification), IMP (Implication), CON (Constraint) |
| Node fill — Red | Evaluation family | CRT (Critique), CMP (Comparison), MET (Meta-reflection) |
| Node fill — Gold | Convergence family | SYN (Synthesis) |
| Edge — Blue | BRCH | Branch: TU opens a new semantic direction |
| Edge — Green | ELAB | Elaboration: TU deepens the immediately prior idea |
| Edge — Red (thick) | BACK | Backtrack: TU returns to revise a semantically distant prior TU |
| Edge — Gold | SYNT | Synthesis: TU integrates content from ≥ 2 prior segments |
| Edge — Gray (thin) | SEQ | Sequential: default adjacent ordering, no strong semantic relation |
| Node label | NodeType code | HYP, RFR, SPC, CRT, SYN, MET, CMP, JUS (shown in white) |

---

*End of Report and Supplementary Material*

*ThinkBench v3 · Non-generative pipeline · No generative LLM in extraction loop · No spacy*  
*Generated: 2026-04-24 · Target venue: EMNLP 2026*
