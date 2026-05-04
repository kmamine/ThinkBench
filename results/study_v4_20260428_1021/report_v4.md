# ThinkBench: A Framework for Profiling the Cognitive Structure of LLM Reasoning

**Study Report v4 — Four-Variant Pilot with NLI Enabled**

**Date**: 2026-04-28 | **Pipeline**: v3 Semantic Embedding Trajectory + NLI | **Model**: Qwen/Qwen3.5-35B-A3B | **Domain**: Ethical Dilemmas | **Target venue**: EMNLP 2026

---

## Abstract

We report a four-variant expansion of the ThinkBench cognitive profiling framework applied to 120 reasoning traces from Qwen/Qwen3.5-35B-A3B across four system-prompt conditions: *Empty* (no system message), *Pure* ("You are a helpful assistant"), *Normal* ("Think carefully before answering"), and *Eliciting* (full structured reasoning framing). Relative to the v3 pilot, two changes are introduced: (i) a fourth *Empty* variant establishes a zero-prompt baseline, and (ii) NLI edge promotion (DeBERTa cross-encoder, threshold 0.78) is enabled throughout extraction. Kruskal-Wallis tests identify 15 of 22 metrics as statistically sensitive to prompt condition (*p* < 0.05). The *Empty* and *Pure* and *Normal* conditions produce nearly identical cognitive profiles, confirming that neither role framing nor a minimal reasoning nudge measurably reorganises this model's cognitive structure. The *Eliciting* condition drives super-linear structural expansion: BACK edges increase 8.4×, SYNT edges 40.3×, and CRT nodes 7.5× relative to *Pure*, while response length increases only 3.8×. Enabling NLI inflates `reasoning_density` to 1.000 across all 120 traces — an artifact of DeBERTa's near-universal SUPP classification at threshold 0.78 — confirming that this metric must be redefined before it can serve as a benchmark signal. Seven metrics are prompt-invariant: `branching_factor`, `specificity_gradient`, `exploration_exploitation_ratio`, `convergence_index`, `hedging_density`, `redundancy_ratio`, and the degenerate `reasoning_density`. These form the recommended cross-model benchmark core.

---

## 1. Introduction

Open-ended reasoning is the primary capability frontier for large language models. Benchmarks such as MMLU, GSM8K, and HumanEval evaluate *what* a model concludes, but not *how* it thinks. Two models can arrive at the same answer through radically different cognitive processes: one through a single confident assertion, another through iterative self-correction, multi-angle exploration, and principled synthesis. This structural difference matters for understanding model behaviour, predicting failure modes, and designing better prompting strategies.

ThinkBench addresses this gap by treating the *structure* of chain-of-thought (CoT) reasoning as the unit of analysis. Instead of scoring final answers, ThinkBench maps the reasoning trace onto a directed graph — the **Thought Graph** — where nodes are thought units (TUs) classified by cognitive function (Hypothesis, Critique, Synthesis, etc.) and edges are semantic relationships (Branching, Elaboration, Backtracking, Synthesis). A 22-dimensional Cognitive Profile is then computed from this graph, characterising the model's reasoning across five dimensions: how widely it explores ideas (Breadth), how deeply it pursues them (Depth), how its reasoning is structurally organised (Structure), the degree to which it monitors its own thinking (Metacognitive), and how economically it uses tokens (Efficiency).

This report documents the v4 study: 120 traces from one model across four prompt conditions. Two primary scientific questions are addressed:

1. *Does the complete absence of a system prompt (Empty) produce a different cognitive profile from a minimal role framing (Pure)?* This tests whether even the most minimal prompt scaffolding influences reasoning structure.
2. *What is the impact of enabling NLI edge promotion on the metric suite?* In v3 the NLI pass was disabled; v4 enables it, allowing us to measure the delta in extracted graph structure and metric values.

The findings from v3 — that `branching_factor` is prompt-invariant and `revision_depth` is strongly prompt-sensitive — are replicated and extended with greater statistical power (4 variants × 30 traces each, *df* = 3 in all Kruskal-Wallis tests).

The remainder of this report is organised as follows. §2 describes the ThinkBench framework. §3 details the v4 experimental setup, including the changes from v3. §4 presents results with inline figures. §5 discusses implications. §6 states limitations and future work. Supplementary Material (§S) contains all supporting figures and graph visualisations.

---

## 2. The ThinkBench Framework

![ThinkBench v3 end-to-end pipeline](../../pipeline_v3.png)
**Figure 0.** Overview of the ThinkBench v3 pipeline. A raw chain-of-thought trace is segmented into Thought Units (TUs) through two boundary-detection passes (cue-phrase rules and TextTiling). TU embeddings are computed in a single batch and used by Layers 3–4 to assign structural edge types (BRCH, ELAB, SYNT, BACK) from the cosine-similarity trajectory. Pass 5 (NLI refinement) is **enabled** in v4: untyped SEQ edges are promoted to SUPP/CONT/ELAB by a DeBERTa cross-encoder at threshold 0.78. The node classifier applies 8-priority edge-structure rules augmented by embedding similarity. The resulting ThoughtGraph feeds the 22-dimensional Cognitive Profile computation.

### 2.1 Thought Graph Construction

A **Thought Graph** *G = (V, E)* is a directed graph where **V** is the set of *Thought Units* (TUs) — contiguous text segments in a reasoning trace, each assigned a cognitive node type — and **E** is the set of directed semantic edges between TUs, each labelled with a relationship type. Cycles are permitted: they represent iterative refinement (propose → critique → revise → re-evaluate), a hallmark of deliberate reasoning that acyclic structures (trees, DAGs) cannot capture.

#### 2.1.1 Segmentation — v3 Semantic Pipeline (5 passes)

**Pass 1 — Hard boundaries (rule-based).** A compiled lexicon of 30+ cue phrases detects three supplementary boundary types: `BACKTRACK` (e.g., "but actually", "reconsidering"), `META` (e.g., "stepping back", "at a higher level"), and `CONVERGENCE` (e.g., "in summary", "balancing these"). These are supplementary signals only; primary structural classification is driven by Pass 3.

**Pass 2 — Soft boundaries (TextTiling).** Each sentence is encoded with `all-MiniLM-L6-v2` (384-dimensional, L2-normalised). A sliding window (width 3) computes cosine similarity at each sentence boundary. Local minima detected via `scipy.signal.find_peaks` (prominence ≥ 0.15) trigger soft `NONE`-class boundaries where similarity falls below the 30th percentile of within-trace similarities.

**Pass 3 — Semantic trajectory (primary structural detection).** For each pair of adjacent TUs (*i−1, i*), cosine similarity *s_i = cos(e_{i−1}, e_i)* is computed. Two adaptive thresholds are derived per trace from the empirical similarity distribution:

- *τ_branch* = 25th percentile of {*s_1, …, s_{n−1}*}
- *τ_elab* = 65th percentile of {*s_1, …, s_{n−1}*}

Each transition is assigned:
- *s_i* < *τ_branch* → `BRCH` edge; new topic segment opened
- *τ_branch* ≤ *s_i* < *τ_elab* → `SEQ` edge; sequential continuation
- *s_i* ≥ *τ_elab* → `ELAB` edge; elaboration of prior TU

Pass 1 assignments override these when `CONVERGENCE` → `SYNT` or `BACKTRACK` → `BACK`.

**Pass 4 — Cross-segment semantic edges.** *Synthesis*: TU *j* receives `SYNT` edges when cos(*e_j*, centroid_s) > 0.50 for ≥ 2 prior segments *s*. *Loop detection*: a `BACK` edge is added when cos(*e_j*, *e_i*) > 0.65 and the minimum intervening similarity drops below 0.45 (confirming a genuine semantic gap), scanning up to 25 positions back.

**Pass 5 — NLI edge promotion (enabled in v4).** `cross-encoder/nli-deberta-v3-large` promotes remaining untyped SEQ edges to SUPP/CONT/ELAB (threshold 0.78, window = 3 TUs). All candidate pairs are batched into a single `predict()` call per trace to avoid per-pair inference overhead.

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

**NLI impact on node classification:** Enabling Pass 5 adds SUPP and CONT edges that did not exist in v3. These trigger JUS (Priority 7) and CMP (Priority 6) node assignments respectively for TUs that would otherwise default to SPC. This increases the diversity of node types across all variants and drives the dramatic increase in SYNT edges: with more JUS/CMP nodes present, Layer 4's centroid-similarity synthesis detector fires more frequently across segment boundaries.

#### 2.1.3 Cognitive Profile — All 22 Metrics Defined

Let *V* = node set, *E* = full edge set, *E_sem* = non-SEQ edges, *E_T* = edges of type *T*, *N_T* = nodes of type *T*, *n = |V|*.

##### Breadth

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| B1 | **Branching Factor** | \|E_BRCH\| / *n* | [0, 1] | Fraction of nodes initiating a new semantic branch. Values > 0.3 indicate a Divergent Explorer style. |
| B2 | **Unique Perspective Count** | \|N_RFR\| | [0, ∞) int | Count of Reframing nodes — branch-starters that re-engage the original problem. |
| B3 | **Domain Spread** | Agglomerative clusters of {N_HYP ∪ N_BRS} (cosine threshold 0.45); 0 if < 3 nodes | [0, ∞) int | Semantic diversity of hypothesis space. Zero when fewer than 3 HYP/BRS nodes exist. |
| B4 | **First Idea Diversity** | Mean pairwise cosine distance among first 3 HYP embeddings | [0, 1] | Diversity of the model's initial hypothesis generation. Zero if fewer than 2 HYP nodes. |

##### Depth

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| D1 | **Max Elaboration Chain** | Longest path in ELAB-only subgraph | [0, ∞) int | Maximum consecutive elaboration depth on any single idea. The ELAB subgraph is acyclic by construction. |
| D2 | **Mean Branch Depth** | Mean shortest-path depth from in-degree-0 roots in semantic subgraph | [0, ∞) | Average hierarchical depth of TUs in the reasoning structure. |
| D3 | **Specificity Gradient** | OLS slope of lexical specificity vs. TU position | (−∞, +∞) | Positive = reasoning becomes more concrete over time. Proxy: capitalised mid-sentence words + numerals + symbols. |
| D4 | **Reasoning Density** | \|nodes with ≥ 1 E_sem\| / *n* | [0, 1] | Fraction of TUs participating in at least one semantic relationship. Degenerate (= 1.000) when NLI is enabled at threshold ≤ 0.78. |

##### Structure

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| S1 | **Exploration–Exploitation Ratio** | \|N_EXPLORATION\| / max(\|N_ELABORATION\|, 1) | [0, ∞) | > 1 = exploration dominant; < 1 = elaboration dominant. |
| S2 | **Backtracking Rate** | \|E_BACK\| / \|E_sem\|; 0 if \|E_sem\| = 0 | [0, 1] | Proportion of semantic edges that are backward revision links. |
| S3 | **Cross-Branch Connectivity** | Fraction of branch-component pairs linked by ≥ 1 SYNT or SUPP edge | [0, 1] | Integration across distinct reasoning branches. |
| S4 | **Convergence Index** | Σ_{v ∈ N_SYN} d_in(*v*) / (*n* × mean_d_in) in E_sem | [0, ∞) | Normalised in-degree concentration at SYN nodes. |
| S5 | **Graph Density** | \|E_sem\| / (*n* × (*n* − 1)) | [0, 1] | Density of the semantic subgraph beyond sequential structure. |
| S6 | **Revision Depth** | mean\|pos(*u*) − pos(*v*)\| for BACK edges | [0, ∞) | Average sequential distance of backward revision links. |

##### Metacognitive

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| M1 | **Self-Reflection Rate** | \|N_MET\| / *n* | [0, 1] | Proportion of TUs that are meta-cognitive commentary on reasoning. |
| M2 | **Critique-to-Hypothesis Ratio** | \|N_CRT\| / max(\|N_HYP\|, 1); 0 if no HYP | [0, ∞) | Evaluative rigour: how often the model challenges its own ideas. |
| M3 | **Hedging Density** | Fraction of TUs containing ≥ 1 uncertainty marker (regex) | [0, 1] | Epistemic humility. Markers: might, could, perhaps, possibly, probably, it seems, unclear, likely, etc. |
| M4 | **Perspective Taking** | \|N_RFR\| / *n* | [0, 1] | Rate of perspective shifts, normalised by trace length. Distinct from UPC (raw count). |

##### Efficiency & Summary

| ID | Name | Formula | Range | Interpretation |
|---|---|---|---|---|
| E1 | **Token per Idea** | total_tokens / max(\|N_RFR\|, 1); fallback = tokens / max(*n*, 1) | [0, ∞) | Tokens invested per reframed perspective. Lower = more concise diversity. |
| E2 | **Redundancy Ratio** | Fraction of TU pairs with cos_sim > 0.75 | [0, 1] | Near-duplicate content rate. Threshold 0.75 calibrated for MiniLM-L6-v2. |
| — | **avg_tokens** | Total token count of reasoning trace | [0, ∞) | Response length control variable. |
| — | **avg_tus** | Total number of TUs after segmentation and merging | [0, ∞) | Trace granularity. |

---

## 3. Experimental Setup

### 3.1 Data Collection

Traces were collected from **Qwen/Qwen3.5-35B-A3B** (35B parameter mixture-of-experts) via a vLLM endpoint (temperature = 0.7, max_tokens = 20000). The model generates reasoning in its `<think>` tag; only the thinking section is analysed. Collection is idempotent: each trace is keyed by a UUID and skipped on re-run.

### 3.2 Question Set

Ten questions from the **ethical_dilemmas** domain were used. Each presents a morally complex open-ended scenario with no single correct answer, spanning utilitarian, deontological, virtue-ethical, and legal-liability dimensions. A single domain was used in this pilot to isolate prompt-condition effects from domain effects. The full benchmark will include 6 domains × 30 questions per domain.

### 3.3 Prompt Variants

**Table 2.** Prompt variant definitions for v4.

| Variant | System Message | Design Intent |
|---|---|---|
| **Empty** | *(absent — no system turn in API request)* | Zero-prompt baseline. Tests model behaviour with no framing or instructions of any kind. |
| **Pure** | *"You are a helpful assistant."* | Minimal role framing, no reasoning guidance. Replicates v3 baseline. |
| **Normal** | *"You are a helpful assistant. Think carefully before answering."* | Minimal reasoning nudge. Tests whether a single sentence shifts cognitive structure. |
| **Eliciting** | Full 180-word framing: spread wide across angles; reframe the problem; go deep on key tensions; challenge your own assumptions; connect ideas across branches; reflect on your reasoning; converge toward a nuanced conclusion. | Upper bound — maximal structural elicitation. |

The **Empty** variant is the key addition in v4. It differs from **Pure** in that the API request contains no system turn at all — the model receives only the user message. This tests whether even the most minimal "helpful assistant" role framing shifts cognitive structure relative to a completely unconstrained generation.

Each variant was applied to all 10 questions with 3 independent runs per question, yielding 30 traces per variant (N = 120 total). This is a between-subjects design: each trace is assigned exactly one prompt condition.

### 3.4 Pipeline Parameters

**Table 3.** Pipeline hyperparameters used in v4. Changes from v3 are marked.

| Parameter | Value | Role | Δ from v3? |
|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Sentence encoding — 384-dim, L2-normalised | No |
| NLI model | `cross-encoder/nli-deberta-v3-large` | Edge promotion (Pass 5) | **Enabled** (v3: disabled) |
| NLI threshold | 0.78 | Minimum NLI score for edge promotion | No |
| NLI window | 3 TUs | Candidate pair window for Pass 5 | No |
| τ_branch | 25th percentile of local sims | BRCH boundary threshold (adaptive per trace) | No |
| τ_elab | 65th percentile of local sims | ELAB boundary threshold (adaptive per trace) | No |
| τ_synthesis | 0.50 | Min cos_sim to prior segment centroid for SYNT | No |
| Min bridged segments | 2 | Min prior segments bridged to trigger SYNT edge | No |
| τ_backtrack | 0.65 | Min cos_sim for BACK edge candidate | No |
| τ_back_drop | 0.45 | Max intervening similarity (gap requirement for BACK) | No |
| Min BACK gap | 4 TUs | Minimum positional distance for BACK edge | No |
| Max BACK lookback | 25 TUs | Lookback window for loop detection | No |
| Redundancy threshold | 0.75 | Cosine similarity threshold for RR computation | No |
| RFR sim low | 0.25 | Lower bound for RFR vs. HYP decision | No |
| RFR sim high | 0.72 | Upper bound for RFR vs. SPC decision | No |
| MET sim threshold | 0.60 | Min cos_sim to TU₀ for embedding-based MET | No |
| MET position fraction | 0.45 | Min fractional position in trace for embedding MET | No |

---

## 4. Results

### 4.1 Extraction Statistics

The v3 pipeline successfully extracted Thought Graphs from all 120 traces (0 failures, 0 skipped). Table 4 reports graph structure by prompt variant.

**Table 4.** Thought Graph structure by prompt variant (mean per trace, 30 traces each). The E/P ratio compares Eliciting to Pure.

| Statistic | Empty | Pure | Normal | Eliciting | E / P ratio |
|---|---|---|---|---|---|
| Nodes (TUs) | 11.1 ± 4.2 | 9.6 ± 1.7 | 9.1 ± 2.2 | 38.7 ± 35.0 | **4.0×** |
| Edges (total) | 40.5 ± 22.1 | 33.6 ± 7.9 | 31.7 ± 10.4 | 205.1 ± 359.4 | **6.1×** |
| SEQ edges | 10.1 | 8.6 | 8.1 | 37.7 | 4.4× |
| BRCH edges | 2.6 | 2.3 | 2.2 | 8.8 | 3.8× |
| ELAB edges | 8.5 | 7.6 | 6.6 | 39.0 | 5.1× |
| BACK edges | 1.1 | 0.9 | 0.8 | 7.6 | **8.4×** |
| SYNT edges | 2.4 | 1.4 | 1.5 | 56.4 | **40.3×** |
| SUPP edges *(NLI)* | 10.3 | 8.0 | 7.9 | 38.5 | 4.8× |
| CONT edges *(NLI)* | 5.4 | 4.9 | 4.5 | 17.0 | 3.5× |
| avg_tokens | 763 ± 247 | 693 ± 105 | 685 ± 139 | 2622 ± 2106 | **3.8×** |
| HYP nodes / trace | 0.63 | 0.47 | 0.33 | 3.70 | 7.9× |
| CRT nodes / trace | 0.43 | 0.67 | 0.37 | 5.03 | **7.5×** |
| SYN nodes / trace | 1.10 | 0.67 | 0.73 | 10.20 | **15.2×** |

**Key structural observations:**

- The Eliciting variant produces traces 3.8× longer with BACK edges increasing 8.4× and SYNT edges 40.3×. Structural complexity scales super-linearly with tokens — the eliciting prompt qualitatively reorganises reasoning, not merely lengthens it.
- **NLI impact (SUPP and CONT edges):** All variants now carry substantial NLI-assigned edges (SUPP: 8–38/trace, CONT: 5–17/trace). In v3 these were absent. The presence of SUPP and CONT edges activates the JUS and CMP node classifier paths (Priorities 7 and 6), which substantially increases SYNT edge count: JUS/CMP nodes create cross-segment centroid similarities that trigger Layer 4's synthesis detection. The 40.3× SYNT ratio is partly a cascade effect of NLI enabling.
- **Empty vs. Pure vs. Normal:** The three baseline variants are structurally nearly identical across all edge and node type counts. The Empty variant produces slightly more edges (+7 per trace vs. Pure) and SYN nodes (+0.43/trace) with no system prompt.

---

![Node and edge type distribution by prompt variant](figures_v2/fig9_node_edge_dist.png)

**Figure 9.** Stacked bar charts showing mean count per trace of each node type (left) and each edge type (right) for all four prompt variants. The Empty, Pure, and Normal bars are nearly indistinguishable, confirming structural equivalence. The Eliciting bar shows massive increases in HYP (3.70 vs. 0.47–0.63), CRT (5.03 vs. 0.37–0.67), and SYN (10.20 vs. 0.67–1.10) nodes, and a dramatic increase in SYNT (56.4) and SUPP (38.5) edges. The CONT edge proportions are similar across variants when normalised by trace length, suggesting NLI CONT classification reflects the model's baseline contrastive reasoning density rather than a prompt effect.

---

Figure 8 shows representative Thought Graphs per variant. Empty, Pure, and Normal graphs are compact (≈9–11 nodes) with a linear-with-branches topology and visible SUPP subgraphs from the NLI pass. Eliciting graphs show dense SYNT networks, long BACK arcs, and extended ELAB chains.

---

![Representative thought graphs — 3 per prompt variant](figures_v2/fig8_graph_examples.png)

**Figure 8.** Selected Thought Graph visualisations (best-scoring per variant by structural richness: 3 × BRCH + 2 × BACK + 2 × SYNT). **Node fill colour** encodes NodeFamily: blue = Exploration (HYP, RFR), green = Elaboration (SPC, JUS), red = Evaluation (CRT, MET), gold = Convergence (SYN). **Edge colour** encodes EdgeType: blue = BRCH, green = ELAB, red = BACK, gold = SYNT; thin gray = SEQ. Panel titles report node count, total edge count, and counts of ↗ BRCH, ↩ BACK, and ⊕ SYNT edges. Note the structural similarity between the Empty, Pure, and Normal panels — all three are compact graphs with similar topology.

---

### 4.2 Descriptive Statistics — Full Metric Suite

Table 5 reports descriptive statistics for all 22 metrics across all 120 traces pooled.

**Table 5.** Descriptive statistics for all 22 metrics (N = 120 traces).

| Metric | Category | Mean | Std | Median | Min | Max | Zero% |
|---|---|---|---|---|---|---|---|
| branching_factor | Breadth | 0.2365 | 0.0383 | 0.2442 | 0.000 | 0.286 | 0.8% |
| unique_perspective_count | Breadth | 2.100 | 1.268 | 2.000 | 0.000 | 9.000 | 6.7% |
| domain_spread | Breadth | 0.875 | 1.824 | 0.000 | 0.000 | 8.000 | 79.2% |
| first_idea_diversity | Breadth | 0.188 | 0.330 | 0.000 | 0.000 | 0.942 | 75.0% |
| max_elaboration_chain | Depth | 4.375 | 3.189 | 3.000 | 1.000 | 17.000 | 0.0% |
| mean_branch_depth | Depth | 2.221 | 2.397 | 1.875 | 0.000 | 15.701 | 23.3% |
| specificity_gradient | Depth | −0.002 | 0.009 | −0.001 | −0.031 | 0.027 | 0.0% |
| **reasoning_density** | Depth | **1.000** | **0.000** | **1.000** | **1.000** | **1.000** | **0.0%** |
| exploration_exploitation_ratio | Structure | 0.648 | 0.781 | 0.500 | 0.000 | 8.000 | 0.8% |
| backtracking_rate | Structure | 0.036 | 0.037 | 0.036 | 0.000 | 0.142 | 41.7% |
| cross_branch_connectivity | Structure | 0.249 | 0.139 | 0.240 | 0.000 | 0.800 | 1.7% |
| convergence_index | Structure | 0.091 | 0.130 | 0.051 | 0.000 | 0.965 | 45.8% |
| graph_density | Structure | 0.259 | 0.102 | 0.289 | 0.043 | 0.500 | 0.0% |
| revision_depth | Structure | 5.624 | 5.754 | 6.000 | 0.000 | 20.333 | 41.7% |
| self_reflection_rate | Metacog. | 0.016 | 0.042 | 0.000 | 0.000 | 0.200 | 85.8% |
| critique_to_hypothesis_ratio | Metacog. | 0.518 | 1.304 | 0.000 | 0.000 | 12.000 | 69.2% |
| hedging_density | Metacog. | 0.240 | 0.139 | 0.222 | 0.000 | 0.571 | 6.7% |
| perspective_taking | Metacog. | 0.162 | 0.080 | 0.167 | 0.000 | 0.286 | 6.7% |
| token_per_idea | Efficiency | 495.9 | 366.3 | 387.0 | 57.0 | 3086 | 0.0% |
| redundancy_ratio | Efficiency | 0.018 | 0.070 | 0.000 | 0.000 | 0.697 | 60.8% |
| avg_tokens | Summary | 1190.9 | 1347.5 | 765.5 | 171 | 12345 | 0.0% |
| avg_tus | Summary | 17.14 | 21.64 | 10.00 | 3 | 204 | 0.0% |

**`reasoning_density` = 1.000 (std = 0.000) across all 120 traces.** This is a confirmed NLI artifact: at threshold 0.78, DeBERTa classifies adjacent reasoning sentences as SUPP entailment for essentially every pair, ensuring every TU participates in at least one semantic edge. The metric has zero discriminative power in this configuration and is flagged for redefinition (§6, Limitation 2).

**Notable distributional properties:**
- `domain_spread` is zero in 79.2% of traces (baseline variants have insufficient HYP nodes for agglomerative clustering).
- `first_idea_diversity` is zero in 75.0% of traces (fewer than 2 HYP nodes in most baseline traces).
- `self_reflection_rate` is zero in 85.8% of traces; meta-cognitive commentary is rare in this model's dense-prose style and is not systematically induced by any prompt except through organic variation.
- `critique_to_hypothesis_ratio` is zero in 69.2% of traces — the distribution is bimodal: most baseline traces have no HYP nodes (formula returns 0), while Eliciting traces show high CHR (max 12.0).
- All 22 metrics have std ≥ 0 (21 with std > 0; `reasoning_density` is the degenerate exception).

Figure 4 shows the full within-variant distribution for each metric.

---

![Per-trace metric distributions — violin plots for all 20 cognitive metrics](figures_v2/fig4_metric_violin.png)

**Figure 4.** 5×4 grid of violin + strip plots for all 20 cognitive metrics (excluding `reasoning_density` — degenerate). Each panel shows four violins (Empty, Pure, Normal, Eliciting) with individual trace data points overlaid as jittered dots. Median is marked as a thick black line inside each violin. Panel titles are colour-coded by cognitive category (blue = Breadth, green = Depth, orange = Structure, purple = Metacognitive, red = Efficiency). Key distributional observations: (1) Empty, Pure, and Normal violins are nearly identical in every panel; (2) `critique_to_hypothesis_ratio` in Eliciting shows extreme right-skew (tail to 12.0); (3) `revision_depth` is approximately symmetric in baseline conditions but strongly right-skewed in Eliciting; (4) `branching_factor` shows the tightest distribution across all four conditions.

---

### 4.3 Prompt Sensitivity Analysis

Table 6 reports per-variant means, absolute deltas (Eliciting − Pure), and Kruskal-Wallis test results for all 22 metrics. Tests use *df* = 3 (four groups), more conservative than v3's *df* = 2.

**Table 6.** Per-variant means and statistical significance of prompt effects (Kruskal-Wallis, df = 3). Bold rows are statistically significant (*p* < 0.05). Δ = Eliciting − Pure.

| Metric | Empty | Pure | Normal | Eliciting | Δ | H | p | Sig. |
|---|---|---|---|---|---|---|---|---|
| branching_factor | 0.237 | 0.236 | 0.240 | 0.233 | −0.004 | 1.59 | 0.661 | ns |
| **unique_perspective_count** | **1.833** | **1.667** | **1.800** | **3.100** | **+1.433** | **22.11** | **<0.001** | *** |
| **domain_spread** | **0.067** | **0.100** | **0.000** | **3.333** | **+3.233** | **76.63** | **<0.001** | *** |
| **first_idea_diversity** | **0.075** | **0.025** | **0.026** | **0.625** | **+0.600** | **70.42** | **<0.001** | *** |
| **max_elaboration_chain** | **3.267** | **3.200** | **2.933** | **8.100** | **+4.900** | **39.26** | **<0.001** | *** |
| **mean_branch_depth** | **1.717** | **1.347** | **1.523** | **4.297** | **+2.949** | **14.58** | **0.002** | ** |
| specificity_gradient | −0.002 | −0.000 | −0.001 | −0.004 | −0.004 | 5.36 | 0.148 | ns |
| reasoning_density | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | — | — | degenerate |
| exploration_exploitation_ratio | 0.524 | 0.561 | 0.722 | 0.784 | +0.223 | 1.61 | 0.656 | ns |
| **backtracking_rate** | **0.027** | **0.033** | **0.026** | **0.057** | **+0.024** | **13.53** | **0.004** | ** |
| **cross_branch_connectivity** | **0.272** | **0.298** | **0.301** | **0.125** | **−0.173** | **41.89** | **<0.001** | *** |
| convergence_index | 0.071 | 0.075 | 0.073 | 0.144 | +0.069 | 5.52 | 0.137 | ns |
| **graph_density** | **0.278** | **0.308** | **0.322** | **0.127** | **−0.181** | **55.33** | **<0.001** | *** |
| **revision_depth** | **4.139** | **3.882** | **2.889** | **11.586** | **+7.704** | **35.88** | **<0.001** | *** |
| **self_reflection_rate** | **0.026** | **0.003** | **0.031** | **0.002** | **−0.001** | **9.11** | **0.028** | * |
| **critique_to_hypothesis_ratio** | **0.167** | **0.311** | **0.100** | **1.495** | **+1.184** | **44.93** | **<0.001** | *** |
| hedging_density | 0.243 | 0.266 | 0.224 | 0.227 | −0.039 | 1.51 | 0.679 | ns |
| **perspective_taking** | **0.171** | **0.179** | **0.200** | **0.098** | **−0.081** | **30.31** | **<0.001** | *** |
| **token_per_idea** | **406.1** | **406.0** | **407.6** | **764.2** | **+358.3** | **18.23** | **<0.001** | *** |
| redundancy_ratio | 0.011 | 0.008 | 0.023 | 0.030 | +0.022 | 6.86 | 0.077 | ns |
| **avg_tokens** | **763** | **693** | **685** | **2622** | **+1929** | **50.59** | **<0.001** | *** |
| **avg_tus** | **11.1** | **9.6** | **9.1** | **38.7** | **+29.1** | **53.64** | **<0.001** | *** |

*Significance: \*\*\* p < 0.001 · \*\* p < 0.01 · \* p < 0.05 · ns p ≥ 0.05*

**15 of 22 metrics are statistically significant** (treating `reasoning_density` as degenerate). Figure 1 visualises the full Δ rankings.

---

![Prompt sensitivity — ranked by absolute delta](figures_v2/fig1_sensitivity_bar.png)

**Figure 1.** Horizontal bar chart ranking all 20 cognitive metrics by |Δ(Eliciting − Pure)|, coloured by cognitive category (blue = Breadth, green = Depth, orange = Structure, purple = Metacognitive, red = Efficiency). Value labels show exact Δ. `token_per_idea` (Δ = +358.3) dominates, followed by `revision_depth` (+7.70), `max_elaboration_chain` (+4.90), and `domain_spread` (+3.23). Negative bars (`cross_branch_connectivity`, `graph_density`, `perspective_taking`) indicate metrics that *decrease* under eliciting — an equally informative prompt-sensitivity signal driven by graph sparsification as trace length increases.

---

**Key observations from Table 6:**

- **`token_per_idea`** shows the largest absolute delta (+358.3). The eliciting prompt produces 3.8× more tokens but only 1.86× more reframing perspectives (UPC: 3.1 vs 1.67), making each perspective more expensive, not less. The eliciting condition is verbose but not proportionally idea-dense.

- **`revision_depth`** is the second-largest delta (+7.70, *p* < 0.001, H = 35.88). The eliciting prompt drives the model to revisit ideas established a mean of 11.6 TUs earlier (vs. 3.9 in Pure), consistent with the explicit "challenge your own assumptions" instruction. This replicates the v3 finding (Δ = +8.80).

- **`domain_spread`** transitions from near-zero in all three baseline variants (0.0–0.1) to 3.33 in Eliciting — a qualitative behavioural shift, not a quantitative scaling.

- **`cross_branch_connectivity`** *decreases* significantly with eliciting (0.298 → 0.125, H = 41.89, *p* < 0.001). Despite having more branches and more synthesis events, the Eliciting graph is *less* cross-connected as a fraction of possible branch pairs: with ~39 branches (vs. ~2.3 in Pure), the denominator in the fraction grows O(*k²*) while the numerator grows only O(*k*). This is a normalisation effect.

- **`graph_density`** *decreases* significantly (0.308 → 0.127, H = 55.33, *p* < 0.001). As the graph grows from ≈9.6 to ≈38.7 nodes, semantic edges do not scale with the O(*n*²) denominator. Reasoning grows broader but not more uniformly interconnected.

- **`self_reflection_rate`** is statistically significant (*p* = 0.028) but shows a counter-intuitive pattern: Empty (0.026) and Normal (0.031) are substantially higher than Pure (0.003) and Eliciting (0.002). SRR rises when the model has neither role framing nor explicit structure — suggesting organic metacognitive asides occur when not suppressed by framing. The role framing in Pure ("you are a helpful assistant") and the explicit cognitive structure in Eliciting both suppress meta-commentary relative to the unscaffolded conditions.

- **`perspective_taking`** *decreases* with eliciting (0.179 → 0.098, *p* < 0.001). This is a normalisation artefact: absolute RFR count rises (1.67 → 3.10), but avg_tus rises 4.0×, driving the rate metric down.

- **`branching_factor`** remains the most stable metric (H = 1.59, *p* = 0.661, ns). The model's branching propensity (~1 BRCH edge per 4 TUs) is invariant to all four prompt conditions, replicating the v3 finding.

Figure 3 shows the aggregate effect by cognitive category. Figure 2 shows the full normalised profile as a radar chart.

---

![Cognitive category scores by prompt variant](figures_v2/fig3_category_comparison.png)

**Figure 3.** Grouped bar chart showing mean normalised score per cognitive category per prompt variant. The Depth category shows the largest eliciting effect (max_elaboration_chain, mean_branch_depth, revision_depth). The Breadth category shows a moderate effect concentrated in domain_spread and first_idea_diversity. Structure shows a mixed pattern: revision and convergence metrics increase while density and connectivity metrics decrease. The Metacognitive category is nearly flat across Empty/Pure/Normal/Eliciting when aggregated, obscuring the significant SRR inversion pattern identified in Table 6.

---

![Radar chart — normalised cognitive profiles for all four prompt variants](figures_v2/fig2_radar.png)

**Figure 2.** Polar radar chart overlaying normalised cognitive profiles for Empty (gray), Pure (blue), Normal (green), and Eliciting (red) variants. The Empty, Pure, and Normal profiles (gray, blue, green) substantially overlap across all 20 axes, confirming that neither role framing nor a minimal "think carefully" instruction measurably reorganises cognitive structure. The Eliciting profile (red) is larger in the upper-right quadrant (Depth and revision axes) while converging with the baseline profiles at branching_factor and hedging_density.

---

### 4.4 Empty vs. Pure: Does the Absence of a System Prompt Matter?

The Empty and Pure variants differ by exactly one thing: whether the API request contains a system message. Key pairwise comparisons:

**Table 7.** Empty vs. Pure metric comparison (30 traces each). Δ = Empty − Pure.

| Metric | Empty | Pure | Δ | Notable? |
|---|---|---|---|---|
| avg_tokens | 763 | 693 | +70 | +10% more tokens with no framing |
| avg_tus | 11.1 | 9.6 | +1.5 | Slightly more TUs |
| unique_perspective_count | 1.833 | 1.667 | +0.166 | Marginal |
| self_reflection_rate | 0.026 | 0.003 | **+0.023** | **9× higher without framing** |
| critique_to_hypothesis_ratio | 0.167 | 0.311 | −0.144 | Fewer critiques without role framing |
| SYN nodes / trace | 1.10 | 0.67 | +0.43 | More synthesis nodes |
| SUPP edges / trace | 10.3 | 8.0 | +2.3 | More NLI-detected support edges |
| branching_factor | 0.237 | 0.236 | +0.001 | Negligible |

The dominant finding: `self_reflection_rate` is 9× higher in the Empty condition (0.026 vs. 0.003). Without a "helpful assistant" role frame, this model is more likely to produce organic metacognitive commentary ("let me reconsider", "stepping back"). The role framing appears to suppress unsolicited self-monitoring.

All other differences are small in absolute terms and likely within sampling noise for n=30. No Kruskal-Wallis test on the 4-variant pooled data separates Empty from Pure as a distinct cluster — the two are statistically indistinguishable on 21 of 22 metrics.

### 4.5 Prompt Sensitivity Classification

Based on statistical significance (Table 6) and standardised effect size (Figure 5), we classify the 22 metrics into two groups.

---

![Cohen's d effect sizes — Eliciting vs. Pure for all metrics](figures_v2/fig5_effect_size.png)

**Figure 5.** Cohen's *d* (Eliciting − Pure) ranked from largest to smallest effect. Vertical reference lines mark |*d*| = 0.2 (small) and |*d*| = 0.8 (large). Colour-coded by cognitive category. Metrics with *d* near zero are prompt-invariant. Negative bars (`cross_branch_connectivity`, `graph_density`, `perspective_taking`) indicate metrics that *decrease* under eliciting. `reasoning_density` is omitted (degenerate, d = 0).

---

**Table 8.** Metric classification for benchmark use.

| Class | Metrics | Benchmark Recommendation |
|---|---|---|
| **Prompt-Sensitive** (KW *p* < 0.05) | `unique_perspective_count`, `domain_spread`, `first_idea_diversity`, `max_elaboration_chain`, `mean_branch_depth`, `backtracking_rate`, `cross_branch_connectivity`, `graph_density`, `revision_depth`, `self_reflection_rate`, `critique_to_hypothesis_ratio`, `perspective_taking`, `token_per_idea`, `avg_tokens`, `avg_tus` | Only interpretable when prompt variant is controlled. Do not use as model capability proxies without specifying the prompt condition. |
| **Prompt-Invariant** (KW *p* ≥ 0.05) | `branching_factor`, `specificity_gradient`, `exploration_exploitation_ratio`, `convergence_index`, `hedging_density`, `redundancy_ratio` | Suitable for cross-model comparison under a fixed prompt. Recommended benchmark metrics. |
| **Degenerate** | `reasoning_density` | Always = 1.000 when NLI is enabled at threshold ≤ 0.78. Requires redefinition before use. |

Comparing v3 and v4 sensitivity classifications:
- `branching_factor` is invariant in both studies — the most robust prompt-invariant signal.
- `self_reflection_rate` was invariant in v3 (*p* = 0.513) but is significant in v4 (*p* = 0.028) — due to the Empty condition's elevated SRR. With three baseline conditions (Empty, Pure, Normal) vs. one in v3, the Empty outlier pulls the KW statistic above threshold.
- `convergence_index` was borderline in v3 (*p* = 0.021, significant) but is not significant in v4 (*p* = 0.137), reclassified as invariant.
- `reasoning_density` went from prompt-invariant (v3: *p* = 0.356, mean ≈ 0.87) to fully degenerate (v4: mean = 1.000, std = 0) due to NLI activation.

### 4.6 Multivariate Structure

#### Spearman Correlation Matrix

Figure 6 shows the Spearman ρ matrix of all 20 cognitive metrics, reordered by Ward hierarchical clustering.

---

![Spearman correlation matrix — Ward-clustered](figures_v2/fig6_correlation_heatmap.png)

**Figure 6.** Spearman ρ correlation matrix (Ward-clustered on 1 − |ρ|  distance), computed across all 120 traces. Two major clusters emerge: **Length-Complexity Cluster** (upper-left): `avg_tokens`, `avg_tus`, `max_elaboration_chain`, `mean_branch_depth`, `revision_depth` — all scale with trace length and are dominated by the Eliciting condition's token inflation. **Structural Density Cluster** (lower-right): `branching_factor`, `graph_density`, `cross_branch_connectivity` — metrics that measure connectivity relative to graph size and are negatively impacted by Eliciting's larger, sparser graphs. Metrics within the same cluster share underlying signal; they should not all be included in a composite score without dimensionality reduction.

---

#### PCA in Metric Space

Figure 7 shows the 120 traces projected onto the first three principal components.

---

![PCA of 120 traces in 20-dimensional metric space](figures_v2/fig7_pca.png)

**Figure 7.** Three-panel PCA scatter of 120 traces, coloured by variant. **PC1 × PC2**: The Eliciting cluster (red) is clearly separated from the three baseline clusters (gray=Empty, blue=Pure, green=Normal), which substantially overlap — confirming that Empty, Pure, and Normal produce statistically indistinguishable cognitive profiles. **PC1 × PC3** and **PC2 × PC3**: The Eliciting cluster is elongated (high within-condition variance in avg_tokens: range 171–12345 tokens), while all three baseline centroids remain co-located.

---

#### Parallel Coordinates

Figure 10 shows all 120 individual trace profiles as parallel coordinate lines.

---

![Parallel coordinates — all 120 traces, variant medians highlighted](figures_v2/fig10_parallel_coords.png)

**Figure 10.** Parallel coordinates plot with one axis per metric (min-max normalised). Individual traces drawn as thin semi-transparent lines; thick solid lines show variant medians. The plot reveals: (1) Eliciting (red) and baseline (gray/blue/green) medians diverge sharply at the Depth axes while converging at `branching_factor` and `hedging_density` axes. (2) Individual Eliciting traces show very high variance — some eliciting traces produce enormous elaboration chains (max MEC = 17) while others are moderate, depending on question complexity. (3) The three baseline variant medians (Empty, Pure, Normal) are nearly coincident across all axes, confirming the PCA finding.

---

### 4.7 Comparison with v3 (NLI Disabled → Enabled)

The primary pipeline change between v3 and v4 is enabling NLI edge promotion (Pass 5). Table 9 compares equivalent metrics across studies for the Pure condition.

**Table 9.** v3 vs. v4 comparison — Pure condition (30 traces each). v3: NLI disabled. v4: NLI enabled (DeBERTa, threshold 0.78).

| Metric | v3 Pure | v4 Pure | Δ | Interpretation |
|---|---|---|---|---|
| avg_tokens | 696 | 693 | −3 | Equivalent (same model, same questions roughly) |
| avg_tus | 9.5 | 9.6 | +0.1 | Equivalent |
| SUPP edges / trace | 0 | 8.0 | **+8.0** | NLI adds ~8 SUPP edges per trace |
| CONT edges / trace | 0 | 4.9 | **+4.9** | NLI adds ~5 CONT edges per trace |
| SYNT edges / trace | 1.47 | 1.40 | −0.07 | Nearly unchanged |
| reasoning_density | 0.861 | 1.000 | **+0.139** | NLI forces all TUs into semantic edges |
| cross_branch_connectivity | 0.047 | 0.298 | **+0.251** | Large increase — SUPP edges now bridge branches |
| graph_density | 0.099 | 0.308 | **+0.209** | Large increase — SUPP/CONT edges increase edge count |
| critique_to_hypothesis_ratio | 0.517 | 0.311 | −0.206 | Fewer CRT nodes despite CONT edges |
| backtracking_rate | 0.090 | 0.033 | −0.057 | Rate drops because denominator (E_sem) increases from NLI edges |

The most significant NLI impact is on `reasoning_density` (1.000 everywhere), `graph_density` (+0.209), and `cross_branch_connectivity` (+0.251). These three metrics are fundamentally different objects in v4 vs. v3, driven by the massive SUPP/CONT edge injection rather than genuine changes in reasoning structure. Results for these metrics are not comparable between the two studies without NLI normalisation.

---

## 5. Discussion

### 5.1 The Three-Way Equivalence of Empty, Pure, and Normal

The central empirical finding of v4 is that **Empty, Pure, and Normal produce statistically indistinguishable cognitive profiles** across 21 of 22 metrics. For this model, the following prompt conditions are cognitively equivalent:

1. No system prompt at all
2. "You are a helpful assistant."
3. "You are a helpful assistant. Think carefully before answering."

This has two implications for benchmark design. First, the benchmark can use any of these as the "baseline" prompt with equivalent results. Second, the "think carefully" nudge — often used in prompting practice — has no measurable effect on the structural properties of this model's reasoning. Whether this generalises to other models is an open empirical question.

The one exception is `self_reflection_rate`: the Empty condition shows 9× higher SRR than Pure (0.026 vs. 0.003). This suggests that role framing suppresses organic metacognitive commentary. The mechanism may be that "helpful assistant" framing directs the model toward task completion rather than self-monitoring. This is a subtle but potentially important finding for benchmarking: a model's intrinsic metacognitive tendencies may be partially hidden by role framing.

### 5.2 Eliciting as an Upper Bound, Not a Capability Measure

The Eliciting condition should be understood as an *upper bound* on structurally rich reasoning from this model under explicit instruction, not as a measure of intrinsic capability. The 8.4× BACK edge increase and 40.3× SYNT edge increase confirm that the eliciting prompt transforms reasoning qualitatively — but this transformation is instruction-following. A model that scores high on `revision_depth` under eliciting may simply be better at following structured instructions, not better at spontaneous self-correction. This confound must be stated explicitly in any paper using eliciting-condition metrics as capability proxies.

The `token_per_idea` dominance of the sensitivity ranking (Δ = +358, 46× larger than the next metric) reveals the core efficiency problem with elicited reasoning: the model generates 3.8× more tokens for 1.86× more perspectives. The eliciting variant is expanding elaboration depth (MEC: +4.9) and revision distance (revision_depth: +7.7) more than genuine perspective diversity (UPC: +1.4).

### 5.3 Non-Obvious Directional Effects

Three metrics decrease significantly under eliciting:

1. **`cross_branch_connectivity`** (−0.173): With ~39 branches in Eliciting vs. ~2.3 in Pure, the number of branch-component pairs grows ~O(k²), but actual cross-branch SYNT/SUPP edges grow only O(k). The fraction falls even as absolute connectivity increases.

2. **`graph_density`** (−0.181): As graph size increases 4.0×, the O(n²) denominator in the density formula grows much faster than actual semantic edge count, producing lower density despite 6.1× more total edges.

3. **`perspective_taking`** (−0.081): A normalisation artefact. The raw RFR count rises (1.67 → 3.10), but avg_tus rises 4.0×, driving the rate metric down.

These decreases are not performance regressions — they are natural consequences of longer, deeper traces on normalised metrics, and must be interpreted accordingly.

### 5.4 NLI Impact: A Double-Edged Sword

Enabling NLI in v4 produces two effects: one expected and one pathological.

**Expected:** SUPP and CONT edges add genuine semantic information for the 30%–40% of adjacent TU pairs that have typed relationships not captured by the embedding trajectory alone. This increased cross_branch_connectivity from 0.047 (v3 Pure) to 0.298 (v4 Pure) — a meaningful structural enrichment.

**Pathological:** `reasoning_density` collapses to 1.000 for all 120 traces because DeBERTa classifies virtually every adjacent reasoning sentence as SUPP entailment at threshold 0.78. This eliminates the metric's discriminative power entirely. The fix is either to raise the threshold substantially (≥ 0.90) or to redefine the metric to count only ELAB + CRIT + SYNT + CONT node participation, excluding the near-universal SUPP.

### 5.5 Recommended Benchmark Metric Core

Combining the prompt-sensitivity classification from Table 8 with the NLI-degenerate finding:

**Recommended prompt-invariant benchmark metrics** (6 + 1 pending fix):
- `branching_factor` — robustly invariant across v3 and v4
- `hedging_density` — stable epistemic humility marker
- `specificity_gradient` — trajectory of concreteness (near-zero for this model)
- `exploration_exploitation_ratio` — exploration vs. elaboration balance
- `convergence_index` — synthesis node concentration
- `redundancy_ratio` — near-duplicate content rate
- `reasoning_density` — *pending redefinition*

---

## 6. Limitations and Future Work

1. **Single model, single domain.** Proving that Cognitive Profiles discriminate between models requires at minimum 3 models × 3 domains. None of the multi-model claims in the paper can be supported from this dataset.

2. **`reasoning_density` is degenerate.** At NLI threshold 0.78, DeBERTa classifies virtually every adjacent sentence pair as SUPP, pinning the metric at 1.000. Fix: redefine as `|ELAB + CRIT + SYNT + CONT nodes| / |V|` (exclude SUPP), or raise the threshold to ≥ 0.90 and re-run.

3. **No multiple-comparison correction.** With 22 simultaneous KW tests, Bonferroni threshold is α = 0.0023. Under this correction `self_reflection_rate` (p=0.028), `backtracking_rate` (p=0.004), and `mean_branch_depth` (p=0.002) revert to invariant. The 12 metrics surviving Bonferroni are the safer benchmark candidates.

4. **No human annotation.** Boundary F1, node classification F1, edge accuracy, and Cohen's κ are all unmeasured. Until a gold set exists, extraction quality is unquantified.

5. **Eliciting confounds instruction-following with reasoning depth.** Metrics that increase under Eliciting measure prompt responsiveness, not intrinsic capacity. They must not be used as model capability proxies without caveat.

6. **No ICC reliability analysis.** ICC(3,1) per metric is computable from the 3 existing runs per question. Metrics with ICC < 0.6 should be flagged as unreliable before inclusion in a benchmark table.

---

## 7. Conclusion

ThinkBench v4 extends the three-variant pilot study with a zero-prompt (Empty) condition and NLI edge promotion enabled. The primary findings are:

1. **Empty ≈ Pure ≈ Normal.** For Qwen/Qwen3.5-35B-A3B, the presence or absence of a system message, and the presence or absence of a "think carefully" nudge, are cognitively equivalent on 21 of 22 metrics. The only exception is `self_reflection_rate`, where role framing suppresses organic metacognitive asides.

2. **Eliciting drives super-linear structural expansion.** BACK edges increase 8.4×, SYNT edges 40.3×, and CRT nodes 7.5× relative to Pure — all at 3.8× the token cost.

3. **NLI enables `reasoning_density` degeneracy.** At threshold 0.78, DeBERTa produces SUPP edges for virtually all adjacent pairs, collapsing `reasoning_density` to 1.000 across all 120 traces. This metric requires redefinition before it can serve as a benchmark signal.

4. **`branching_factor` remains the most stable prompt-invariant metric** — replicated from v3. The model's branching propensity (~1 BRCH edge per 4 TUs) is a fixed property regardless of prompt condition.

The framework is ready for multi-model expansion. The minimum viable next experiment is: 2–3 additional models under the Normal prompt condition, with the prompt-invariant metric core (Table 8) as the primary comparison basis.

---

---

# Supplementary Material

---

## S1. Aggregated Metric Heatmap

![Aggregated metric heatmap — normalised values, all 4 variants](supplement/S1_metric_heatmap.png)

**Figure S1.** Heatmap of normalised metric values at the aggregated level (rows = metrics, columns = prompt variants). The Empty, Pure, and Normal columns are nearly identical across all rows, confirming the three-way equivalence finding from §4.1. The Eliciting column shows elevated values in the Depth rows (upper portion) and depressed values in the graph_density and cross_branch_connectivity rows (mid-section), consistent with the sparsification effect of larger graphs.

---

## S2. Key Metric Pairwise Scatters

![9 key bivariate scatter plots](supplement/S2_scatter_pairs.png)

**Figure S2.** 3×3 grid of bivariate scatter plots for nine informative metric pairs, coloured by variant (gray = Empty, blue = Pure, green = Normal, red = Eliciting). Key panels: **branching_factor × revision_depth** (top-left): Eliciting traces (red) are displaced upward (higher revision_depth) but not rightward (same branching_factor), confirming branching is prompt-invariant while revision is prompt-sensitive. The three baseline clusters (gray, blue, green) are co-located — visually confirming the three-way equivalence. **backtracking_rate × critique_to_hypothesis_ratio**: Both increase together under eliciting, concentrated in the upper-right quadrant.

---

## S3. Hierarchical Clustering Dendrogram

![Ward dendrogram of 120 traces in 20-metric space](supplement/S3_dendrogram.png)

**Figure S3.** Ward-linkage dendrogram of all 120 traces in the 20-dimensional metric space (z-score normalised). Leaf ticks colour-coded by variant: gray = Empty, blue = Pure, green = Normal, red = Eliciting. Eliciting traces (red) tend to cluster in one subtree on the right side, while the three baseline variants intermix on the left — consistent with the PCA finding that Empty/Pure/Normal occupy the same metric-space region.

---

## S4. Per-Trace Profile Heatmap

![All 120 trace profiles as rows](supplement/S4_per_trace_heatmap.png)

**Figure S4.** All 120 traces as rows in a normalised metric heatmap, grouped by variant (separated by horizontal dividers). Each cell encodes the normalised value [0–1] for that metric × trace. Within each of the three baseline groups (Empty, Pure, Normal), rows appear nearly identical — reflecting the three-way equivalence. The Eliciting group shows elevated values in the Depth columns and elevated `critique_to_hypothesis_ratio` for high-revision traces. Within-Eliciting variance is highest (wide range of trace lengths from 171 to 12345 tokens).

---

## S5. All Thought Graphs — Empty Condition

All 30 Empty condition traces (9 graphs per page, 4 pages).

![All Empty traces — page 1 of 4](supplement/graphs_all_empty_p01.png)

**Figure S5a.** Empty condition Thought Graphs, page 1 of 4. No system prompt — the model receives only the user question. Graphs are structurally similar to Pure condition: compact (≈11 nodes), moderate SUPP and CONT edge density from NLI, occasional BACK arcs. Slightly more SYN nodes on average than Pure (1.10 vs. 0.67/trace).

---

## S6. All Thought Graphs — Pure Condition

All 30 Pure condition traces (9 graphs per page, 4 pages). Colour scheme identical to Figure 8.

![All Pure traces — page 1 of 4](supplement/graphs_all_pure_p01.png)

**Figure S6a.** Pure condition Thought Graphs, page 1 of 4. Note the compact size (typically 8–12 nodes) and the presence of SUPP (green) and CONT edges from the enabled NLI pass — absent in v3 graphs. BACK (red) and SYNT (gold) edges are occasional but present, confirming that even under Pure conditions the model revises and synthesises.

![All Pure traces — page 2 of 4](supplement/graphs_all_pure_p02.png)

**Figure S6b.** Pure condition Thought Graphs, page 2 of 4.

![All Pure traces — page 3 of 4](supplement/graphs_all_pure_p03.png)

**Figure S6c.** Pure condition Thought Graphs, page 3 of 4.

![All Pure traces — page 4 of 4](supplement/graphs_all_pure_p04.png)

**Figure S6d.** Pure condition Thought Graphs, page 4 of 4.

---

## S7. All Thought Graphs — Normal Condition

All 30 Normal condition traces (9 graphs per page, 4 pages).

![All Normal traces — page 1 of 4](supplement/graphs_all_normal_p01.png)

**Figure S7a.** Normal condition Thought Graphs, page 1 of 4. Structurally nearly identical to Empty and Pure traces — the "Think carefully" instruction does not measurably change graph topology. This visual finding is consistent with the statistical finding that all three baseline variants occupy the same region of metric space.

![All Normal traces — page 2 of 4](supplement/graphs_all_normal_p02.png)

**Figure S7b.** Normal condition Thought Graphs, page 2 of 4.

![All Normal traces — page 3 of 4](supplement/graphs_all_normal_p03.png)

**Figure S7c.** Normal condition Thought Graphs, page 3 of 4.

![All Normal traces — page 4 of 4](supplement/graphs_all_normal_p04.png)

**Figure S7d.** Normal condition Thought Graphs, page 4 of 4.

---

## S8. All Thought Graphs — Eliciting Condition

All 30 Eliciting condition traces (9 graphs per page, 4 pages). Note substantially larger graphs and denser edge structure.

![All Eliciting traces — page 1 of 4](supplement/graphs_all_eliciting_p01.png)

**Figure S8a.** Eliciting condition Thought Graphs, page 1 of 4. Eliciting traces exhibit the full range of structural patterns: large SYNT (gold) subgraphs drawing from many prior nodes, long BACK (red) arcs spanning many positions, and extended ELAB (green) chains. The NLI-enabled SUPP (green) and CONT edges add further density. Some traces are extremely large (max 204 TUs in this study), displayed at reduced node size.

![All Eliciting traces — page 2 of 4](supplement/graphs_all_eliciting_p02.png)

**Figure S8b.** Eliciting condition Thought Graphs, page 2 of 4.

![All Eliciting traces — page 3 of 4](supplement/graphs_all_eliciting_p03.png)

**Figure S8c.** Eliciting condition Thought Graphs, page 3 of 4.

![All Eliciting traces — page 4 of 4](supplement/graphs_all_eliciting_p04.png)

**Figure S8d.** Eliciting condition Thought Graphs, page 4 of 4.

---

## S9. Reading the Thought Graphs — Colour Key

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
| Edge — Orange | SUPP | NLI-detected support/evidence relationship |
| Edge — Purple | CONT | NLI-detected contrast/contradiction relationship |
| Edge — Gray (thin) | SEQ | Sequential: default adjacent ordering, no strong semantic relation |
| Node label | NodeType code | HYP, RFR, SPC, CRT, SYN, MET, CMP, JUS (shown in white) |

---

*End of Report and Supplementary Material*

*ThinkBench v4 · Non-generative pipeline · NLI enabled (DeBERTa, threshold 0.78) · No generative LLM in extraction loop*  
*Generated: 2026-04-28 · Target venue: EMNLP 2026*
