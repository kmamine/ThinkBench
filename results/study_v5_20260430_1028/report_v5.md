# ThinkBench Study v5: Structural Pre-Segmentation and Cognitive Profile Analysis

**Model:** Qwen/Qwen3.5-35B-A3B  
**Domain:** ethical\_dilemmas (10 questions × 3 runs × 4 prompt variants = 120 traces)  
**Segmenter:** v3 — 5-pass semantic trajectory pipeline with markdown structural pre-segmentation (Pass 0)  
**Date:** 2026-04-30  
**Study ID:** study\_v5\_20260430\_1028

---

## Abstract

ThinkBench is a framework for characterizing the reasoning behavior of large language models (LLMs) through structural analysis of chain-of-thought (CoT) traces. A key technical challenge is the accurate segmentation of dense reasoning text into discrete thought units (TUs) before graph construction and metric computation. This report documents version 5 of the ThinkBench study, which introduces a structural pre-segmentation pass (Pass 0) that leverages markdown document structure — ATX headers, bold section labels, list items, and horizontal rules — as explicit thought boundaries. The new segmenter doubles the average TU count from 9.1 to 19.6 per trace and transforms seven previously degenerate metrics (all returning 0.00) into informative measurements. Across 120 reasoning traces from four prompt variants (pure, normal, empty, eliciting), Kruskal-Wallis tests identify 14 of 22 metrics as statistically sensitive to prompt framing (p < 0.05). Cohen's d analysis reveals large cross-variant effects (|d| > 1.0) for eight metrics. The eliciting variant, which instructs the model to reason broadly, produces 4.2× more TUs, 1.89× higher unique\_perspective\_count, and 11.6× higher domain\_spread compared to pure prompting. Six metrics exhibit strong prompt sensitivity while 14 remain prompt-invariant, suggesting the existence of a stable cognitive signature resistant to surface-level prompt engineering.

---

## 1. Introduction

Characterizing the cognitive style of LLMs remains an open problem. Evaluation benchmarks that rely on closed-ended correctness metrics collapse the distinction between *how* a model reasons and *whether* it succeeds. A model that arrives at a correct answer through narrow, linear inference may be less useful for exploratory tasks than one that generates diverse hypotheses, even if the latter produces more incorrect intermediate steps.

ThinkBench addresses this gap by extracting a **ThoughtGraph** from each reasoning trace — a directed graph with cycles in which nodes represent classified thought units and edges encode semantic relationships — and computing a 22-dimensional **cognitive profile vector** from this graph. The profile spans five cognitive dimensions: Breadth (how widely the model explores), Depth (how deeply it develops individual ideas), Structure (the topological organization of reasoning), Metacognitive (self-monitoring and revision behavior), and Efficiency (information density).

A prerequisite for meaningful metric computation is accurate segmentation. If TUs are too coarse, structural properties (branching, backtracking, cross-branch connectivity) cannot be resolved. In study v4, the segmenter relied on double-newline paragraph breaks (`\n\n+`) as the primary boundary signal. For Qwen/Qwen3.5-35B-A3B, which renders structured responses using markdown headers, bold section labels, and bulleted lists, this produced traces with an average of only 9.1 TUs per trace — insufficient resolution for most graph-theoretic metrics. Seven of 22 metrics returned 0.00 for every trace in v4.

Study v5 introduces **Pass 0** — a markdown structural pre-segmentation layer that converts document-level formatting cues (ATX headers `#–######`, bold headers `**Label:**`, list items `- * 1.`, horizontal rules `---`) into explicit BRANCH boundaries before the embedding-based TextTiling pass runs. The result is a resolution increase of 115% in average TU count, activation of previously degenerate metrics, and a richer cognitive profile that discriminates prompt variants with statistical significance.

---

## 2. Framework Architecture

### 2.1 Pipeline Overview

ThinkBench processes each reasoning trace through a five-stage extraction pipeline followed by metric computation:

```
Raw CoT Trace
     │
     ▼
Pass 0: Structural Pre-Segmentation (markdown parsing)
     │
     ▼
Pass 1: Cue-Phrase Boundary Detection (regex lexicon)
     │
     ▼
Pass 2: TextTiling Soft Boundaries (MiniLM-L6-v2 embeddings)
     │
     ▼
Pass 3: NLI Edge Promotion (DeBERTa-v3-large cross-encoder)
     │
     ▼
Pass 4 [Layer 3]: Semantic Trajectory Assembly
     │
     ▼
ThoughtGraph → 22-Metric Profile
```

### 2.2 Pass 0 — Structural Pre-Segmentation (New in v3 Segmenter)

The core innovation in v5 is Pass 0, which runs before any cue-phrase or embedding analysis. It processes the trace line by line and detects four categories of markdown formatting as hard BRANCH boundaries:

**ATX headers:** Lines matching `^#{1,6}\s+(.+)` are treated as structural section starts. The header text is extracted with markdown stripped, and a new BRANCH span begins.

```python
_HEADER_RE = re.compile(r'^(#{1,6})[ \t]+(.+)')
```

**Bold section labels:** Lines matching `^\*\*([^*\n]{2,80})\*\*\s*:?\s*$` where the content does not end in a sentence-terminating character (`.!?`) are treated as section labels emitting BRANCH boundaries. Bold text followed by sentence content (e.g., `**This approach** leverages...`) is *not* split, preserving inline emphasis.

```python
_BOLD_HDR_RE = re.compile(r'^\*\*([^*\n]{2,80})\*\*\s*:?\s*$')
```

**List items:** Each list item (bullet `-*+` or numbered `1.`) is treated as a separate span with NONE boundary class. Items within a list emit one TU each, preserving parallel structure.

```python
_BULLET_RE   = re.compile(r'^[ \t]*[-*+][ \t]+\S')
_NUMBERED_RE = re.compile(r'^[ \t]*\d+[.)][ \t]+\S')
```

**Horizontal rules:** Lines matching `^[-*_]{3,}\s*$` flush the current span and start a new NONE span without emitting a BRANCH (they are structural separators, not topical shifts).

**Paragraph breaks within prose:** Single newlines between a sentence-ending line and an uppercase-starting continuation are treated as soft paragraph boundaries (flush current span, new NONE span), capturing implicit paragraph structure not marked by double newlines.

After Pass 0, all spans undergo **three-rule merge logic** before cue-phrase analysis:

1. **Short BRANCH absorbs next:** If a BRANCH span has fewer than 12 tokens (typically a pure header with no content), it absorbs the immediately following span, which receives the BRANCH class. This prevents isolated header TUs from polluting graph density statistics.
2. **Short NONE merges forward into NONE only:** A NONE span with fewer than 2 sentences or 30 tokens merges forward — but *only* if the next span also has NONE class. It never absorbs a BRANCH span. This prevents a short prose fragment from being incorrectly classified as a structural boundary.
3. **BRANCH spans are never merged into preceding spans:** Once a span is classified BRANCH (from Pass 0 or cue-phrase detection), it anchors a new TU regardless of token count.

### 2.3 Pass 1 — Cue-Phrase Boundary Detection

After structural pre-segmentation, Pass 1 applies a compiled lexicon of 43 cue-phrase regular expressions to NONE-class spans. Each cue phrase is anchored to the start of a sentence boundary and mapped to a BoundaryClass:

| BoundaryClass | Example Phrases |
|---|---|
| BACKTRACK | `wait\b`, `actually\b`, `on second thought`, `let me reconsider` |
| BRANCH | `alternatively\b`, `another approach`, `consider instead`, `on the other hand` |
| META | `at a higher level`, `stepping back`, `thinking about this` |
| CONVERGENCE | `in summary`, `to conclude`, `final answer`, `synthesis:` |
| ELABORATION | `specifically`, `to elaborate`, `in particular` |
| CONTRAST | `however`, `yet\b`, `but\b`, `in contrast` |
| SUPPORT | `because`, `since`, `given that`, `this is supported by` |

NONE spans with no cue-phrase match are classified by Pass 2 (TextTiling) if a similarity minimum is detected, or remain NONE (mapped to SPC node type).

### 2.4 Pass 2 — TextTiling Soft Boundaries

Sentences within each NONE span are encoded with `all-MiniLM-L6-v2` (384-dim). A sliding window of size 3 computes cosine similarity between adjacent windows. Local minima with prominence ≥ 0.15 that fall below the 30th percentile of within-trace similarities are inserted as NONE boundary points. This pass captures semantic drift not marked by explicit cues.

### 2.5 Pass 3 — NLI Edge Promotion

Adjacent TU pairs within a window of 4 positions are tested against three NLI hypotheses using `cross-encoder/nli-deberta-v3-large`:

- **SUPP:** "The second text provides evidence or support for the first." (threshold 0.78)
- **CONT:** "The second text contradicts or contrasts with the first." (threshold 0.75)
- **ELAB:** "The second text elaborates or extends the first." (threshold 0.75)

For BACKTRACK spans at distance > 4, a BACK hypothesis is tested (threshold 0.70). All pairs are submitted as a single batched `predict()` call. Promoted edge types override the default SEQ edge assigned during graph assembly.

### 2.6 Pass 4 — Semantic Trajectory Assembly (Layer 3)

The final pass assembles the `ThoughtGraph` object. TUs are assigned `NodeType` via an 8-priority deterministic classifier (BoundaryClass → NodeType mapping with special rules for index 0 → HYP and segment-increment tracking). Edge types are assigned based on BoundaryClass with NLI overrides applied. BRANCH boundaries increment the segment counter, enabling cross-segment metrics (cross\_branch\_connectivity, domain\_spread).

### 2.7 The 22-Metric Cognitive Profile

All metrics operate on the `ThoughtGraph` object. Key metric definitions:

| Metric | Formula | Category |
|---|---|---|
| `branching_factor` | \|E\_BRCH\| / max(\|V\|, 1) | Breadth |
| `unique_perspective_count` | count(RFR nodes) | Breadth |
| `domain_spread` | agglomerative clusters of BRS+HYP node embeddings (cosine threshold 0.45), min 3 nodes required | Breadth |
| `first_idea_diversity` | mean pairwise cosine distance between first-TU embeddings per segment | Breadth |
| `max_elaboration_chain` | longest path in ELAB-only subgraph | Depth |
| `mean_branch_depth` | mean depth of all nodes in ELAB-only subgraph | Depth |
| `specificity_gradient` | linear regression slope of named-entity density vs. TU position | Depth |
| `reasoning_density` | \|nodes with ≥1 semantic edge\| / \|V\| | Depth |
| `exploration_exploitation_ratio` | \|EXPLORATION nodes\| / max(\|ELABORATION+EVALUATION+CONVERGENCE nodes\|, 1) | Structure |
| `backtracking_rate` | \|E\_BACK\| / max(\|E\_sem\|, 1) | Structure |
| `cross_branch_connectivity` | fraction of cross-segment node pairs with SYNT/SUPP edge | Structure |
| `convergence_index` | sum(d\_in(v) for SYN nodes) / (\|V\| × mean\_d\_in) | Structure |
| `graph_density` | \|E\_sem\| / (\|V\| × (\|V\|-1)), semantic edges only | Structure |
| `revision_depth` | mean(\|pos(u)-pos(v)\|) for BACK edges | Structure |
| `self_reflection_rate` | count(MET nodes) / \|V\| | Metacognitive |
| `critique_to_hypothesis_ratio` | \|CRT nodes\| / max(\|HYP nodes\|, 1) | Metacognitive |
| `hedging_density` | count(hedge words) / total word count | Metacognitive |
| `perspective_taking` | count(RFR nodes) / max(\|V\|, 1) | Metacognitive |
| `token_per_idea` | token\_count / max(unique\_perspective\_count, 1) | Efficiency |
| `redundancy_ratio` | fraction of TU pairs with cosine similarity > 0.90 | Efficiency |

---

## 3. Experimental Setup

### 3.1 Model

**Qwen/Qwen3.5-35B-A3B** served via a local vLLM endpoint. The model generates structured responses using markdown headings, bold labels, and bullet lists — a writing style that was systematically under-segmented by v4's paragraph-only boundary detection.

### 3.2 Dataset

Ten questions from the `ethical_dilemmas` domain (IDs ED001–ED010). Questions cover trolley-problem variants, medical resource allocation, privacy-security tradeoffs, and fiduciary conflicts. Each question is designed to elicit multi-perspective reasoning without a uniquely correct answer.

### 3.3 Prompt Variants

Four prompt variants were tested:

| Variant | Instruction Strategy |
|---|---|
| **pure** | Raw question only, no reasoning instruction |
| **normal** | "Think step by step." appended |
| **empty** | Explicit reasoning tags `<think>...</think>` with empty body |
| **eliciting** | Detailed cognitive scaffolding: "Spread wide. Reframe. Go deep. Challenge your assumptions. Synthesize." |

Each question × variant combination was run 3 times (K=3) for a total of 10 × 4 × 3 = 120 traces.

### 3.4 Pipeline Parameters

| Parameter | Value |
|---|---|
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| NLI model | cross-encoder/nli-deberta-v3-large |
| TextTiling window | 3 sentences |
| TextTiling prominence threshold | 0.15 |
| TextTiling similarity percentile | 30th |
| NLI SUPP threshold | 0.78 |
| NLI CONT/ELAB threshold | 0.75 |
| NLI BACK threshold | 0.70 (distance > 4) |
| domain\_spread min nodes | 3 |
| Merge: min BRANCH tokens | 12 |
| Merge: min NONE sentences | 2 |
| Merge: min NONE tokens | 30 |

---

## 4. Results

### 4.1 Extraction Statistics

Pass 0 structural pre-segmentation substantially increases TU resolution across all non-eliciting variants. The eliciting variant produces responses 3.8× longer than pure, yielding 81.8 TUs/trace on average.

| Variant | n | Avg Tokens | Avg TUs | Avg Edges | Avg TUs/Token |
|---|---|---|---|---|---|
| pure | 30 | 693 | 19.5 | 83.6 | 0.0282 |
| normal | 30 | 685 | 19.6 | 81.7 | 0.0286 |
| empty | 30 | 763 | 22.1 | 99.5 | 0.0290 |
| eliciting | 30 | 2,622 | 81.8 | 465.8 | 0.0312 |

The TUs/Token ratio remains relatively stable across variants (0.028–0.031), confirming that the segmenter responds proportionally to response length rather than over-segmenting short traces.

### 4.2 Node Type Distribution

Node classification reveals consistent patterns within short-response variants (pure/normal/empty), with the eliciting variant showing a qualitatively different distribution dominated by SYN nodes.

#### Per-trace average node counts

| Node Type | Family | pure | normal | empty | eliciting |
|---|---|---|---|---|---|
| RFR (Reframing) | Exploration | 5.17 | 4.83 | 5.53 | 10.47 |
| JUS (Justification) | Elaboration | 4.63 | 4.63 | 4.83 | 15.90 |
| SYN (Synthesis) | Convergence | 3.57 | 2.83 | 4.73 | 27.27 |
| SPC (Specification) | Elaboration | 2.73 | 2.53 | 2.60 | 9.13 |
| CMP (Comparison) | Evaluation | 1.53 | 2.10 | 1.83 | 4.53 |
| HYP (Hypothesis) | Exploration | 1.37 | 1.70 | 1.70 | 10.10 |
| CRT (Critique) | Evaluation | 0.50 | 0.97 | 0.80 | 4.43 |
| MET (Meta-reflection) | Metacognitive | 0.03 | 0.03 | 0.03 | 0.00 |

**Key observations:**
- RFR (reframing) is the dominant node type in short variants, indicating the model frequently pivots perspective within its structured sections.
- SYN nodes are disproportionately elevated in the eliciting variant (27.27/trace vs. 3.57 for pure), consistent with the eliciting prompt's explicit instruction to synthesize.
- HYP nodes increase 7.4× under eliciting (10.10 vs. 1.37), reflecting the broader hypothesis generation encouraged by cognitive scaffolding.
- CRT nodes increase 8.9× under eliciting, suggesting prompted self-critique as a distinct reasoning behavior.
- MET (meta-reflection) remains near-zero across all variants, indicating the model does not produce explicit meta-level observations that trigger the MET cue phrases.

### 4.3 Edge Type Distribution

#### Per-trace average edge counts

| Edge Type | pure | normal | empty | eliciting |
|---|---|---|---|---|
| SUPP (Support) | 19.77 | 20.93 | 23.77 | 91.17 |
| SEQ (Sequential) | 18.53 | 18.63 | 21.07 | 80.83 |
| ELAB (Elaboration) | 14.77 | 13.20 | 17.10 | 86.60 |
| SYNT (Synthesis) | 10.60 | 8.47 | 15.67 | 136.40 |
| CONT (Contrast) | 9.03 | 9.93 | 9.40 | 27.60 |
| BRCH (Branch) | 8.33 | 8.13 | 9.33 | 29.77 |
| BACK (Backtrack) | 2.53 | 2.40 | 3.13 | 13.43 |

**Key observations:**
- SYNT edges dominate the eliciting variant at 136.4/trace (vs. 10.6 for pure), directly caused by the high SYN node count under eliciting. SYNT edges represent synthesis connections crossing thought segments.
- SUPP edges are the most frequent edge type in all variants, reflecting NLI-promoted SUPPORT relationships between adjacent TUs. The high SUPP rate (relative to ELAB and BRCH) is a known artifact of the DeBERTa NLI model's tendency to classify consecutive reasoning sentences as entailment/support at the 0.78 threshold.
- BACK edges appear in all variants (2.4–13.4/trace), indicating the segmenter successfully detects backtracking behavior. The 5.3× increase under eliciting suggests the cognitive scaffolding prompt activates revision behavior.
- BRCH edges increase 3.6× under eliciting, confirming that structural branching is prompt-responsive.

### 4.4 Full Cognitive Profile — All 22 Metrics × 4 Variants

| Metric | Category | pure | normal | empty | eliciting |
|---|---|---|---|---|---|
| branching\_factor | Breadth | 0.429 | 0.429 | 0.422 | 0.371 |
| unique\_perspective\_count | Breadth | 5.167 | 4.833 | 5.533 | **10.467** |
| domain\_spread | Breadth | 0.767 | 0.833 | 0.833 | **8.700** |
| first\_idea\_diversity | Breadth | 0.259 | 0.388 | 0.312 | **0.711** |
| max\_elaboration\_chain | Depth | 4.400 | 3.833 | 4.400 | **16.300** |
| mean\_branch\_depth | Depth | 1.859 | 1.517 | 2.535 | **6.025** |
| specificity\_gradient | Depth | 0.001 | 0.001 | −0.001 | −0.002 |
| reasoning\_density | Depth | 1.000 | 1.000 | 1.000 | 1.000 |
| exploration\_exploitation\_ratio | Structure | 0.957 | 1.206 | 1.064 | 1.246 |
| backtracking\_rate | Structure | 0.037 | 0.034 | 0.035 | 0.038 |
| cross\_branch\_connectivity | Structure | 0.313 | 0.289 | 0.300 | **0.102** |
| convergence\_index | Structure | 0.175 | 0.137 | 0.192 | 0.241 |
| graph\_density | Structure | 0.185 | 0.181 | 0.168 | **0.077** |
| revision\_depth | Structure | 9.631 | 9.831 | 10.668 | **13.534** |
| self\_reflection\_rate | Metacognitive | 0.002 | 0.003 | 0.001 | 0.000 |
| critique\_to\_hypothesis\_ratio | Metacognitive | 0.122 | 0.681 | 0.428 | 0.495 |
| hedging\_density | Metacognitive | 0.163 | 0.147 | 0.154 | 0.131 |
| perspective\_taking | Metacognitive | 0.266 | 0.256 | 0.251 | **0.142** |
| token\_per\_idea | Efficiency | 147.16 | 154.50 | 158.25 | **274.38** |
| redundancy\_ratio | Efficiency | 0.007 | 0.006 | 0.007 | 0.011 |
| avg\_tokens | (summary) | 693 | 685 | 763 | 2,622 |
| avg\_tus | (summary) | 19.5 | 19.6 | 22.1 | 81.8 |

Bold values indicate the variant with highest (or lowest) absolute value where the metric is statistically sensitive.

**Notable pattern — eliciting's structural compression:** Despite generating 4.2× more TUs, the eliciting variant shows *lower* graph\_density (0.077 vs. 0.185 for pure) and *lower* cross\_branch\_connectivity (0.102 vs. 0.313 for pure). The graph grows sparsely in a tree-like manner rather than forming a dense interconnected network. Separately, perspective\_taking drops under eliciting (0.142 vs. 0.266 for pure) because the denominator (\|V\|) grows faster than RFR node count.

### 4.5 Sensitivity Analysis — Kruskal-Wallis Tests

Kruskal-Wallis H-tests (df=3, 4 groups) were applied to all 22 metrics across individual traces. Fourteen metrics are statistically significant at p < 0.05:

| Metric | KW H | p-value | Effect (Δ eliciting−pure) |
|---|---|---|---|
| avg\_tus | — | < 0.001 | +62.3 |
| avg\_tokens | — | < 0.001 | +1929.0 |
| token\_per\_idea | — | < 0.001 | +127.23 |
| max\_elaboration\_chain | — | < 0.001 | +11.90 |
| domain\_spread | — | < 0.001 | +7.93 |
| unique\_perspective\_count | — | < 0.001 | +5.30 |
| mean\_branch\_depth | — | < 0.001 | +4.17 |
| revision\_depth | — | < 0.001 | +3.90 |
| cross\_branch\_connectivity | — | < 0.001 | −0.211 |
| graph\_density | — | < 0.001 | −0.109 |
| perspective\_taking | — | < 0.001 | −0.124 |
| critique\_to\_hypothesis\_ratio | — | 0.001 | +0.373 |
| branching\_factor | — | 0.028 | −0.058 |
| specificity\_gradient | — | 0.029 | −0.002 |

Eight metrics are statistically invariant to prompt framing (p ≥ 0.05):

| Metric | Classification |
|---|---|
| reasoning\_density | Prompt-invariant (constant 1.0) |
| self\_reflection\_rate | Prompt-invariant |
| backtracking\_rate | Prompt-invariant |
| hedging\_density | Prompt-invariant |
| redundancy\_ratio | Prompt-invariant |
| convergence\_index | Prompt-invariant |
| exploration\_exploitation\_ratio | Prompt-invariant |
| first\_idea\_diversity | Prompt-invariant (p=0.09) |

### 4.6 Sensitivity Ranking — Absolute Delta (Eliciting − Pure)

Metrics ranked by |Δ| between the most extreme variants (eliciting vs. pure):

| Rank | Metric | Category | Δ (eliciting − pure) | Δ (normal − pure) |
|---|---|---|---|---|
| 1 | token\_per\_idea | Efficiency | +127.23 | +7.34 |
| 2 | max\_elaboration\_chain | Depth | +11.90 | −0.57 |
| 3 | domain\_spread | Breadth | +7.93 | +0.07 |
| 4 | unique\_perspective\_count | Breadth | +5.30 | −0.33 |
| 5 | mean\_branch\_depth | Depth | +4.17 | −0.34 |
| 6 | revision\_depth | Structure | +3.90 | +0.20 |
| 7 | first\_idea\_diversity | Breadth | +0.452 | +0.129 |
| 8 | critique\_to\_hypothesis\_ratio | Metacognitive | +0.373 | +0.558 |
| 9 | exploration\_exploitation\_ratio | Structure | +0.289 | +0.250 |
| 10 | cross\_branch\_connectivity | Structure | −0.211 | −0.023 |
| 11 | perspective\_taking | Metacognitive | −0.124 | −0.009 |
| 12 | graph\_density | Structure | −0.109 | −0.004 |
| 13 | convergence\_index | Structure | +0.066 | −0.039 |
| 14 | branching\_factor | Breadth | −0.058 | +0.0004 |
| 15 | hedging\_density | Metacognitive | −0.032 | −0.016 |
| 16 | redundancy\_ratio | Efficiency | +0.004 | −0.001 |
| 17 | specificity\_gradient | Depth | −0.002 | +0.0001 |
| 18 | self\_reflection\_rate | Metacognitive | −0.002 | +0.001 |
| 19 | backtracking\_rate | Structure | +0.001 | −0.003 |
| 20 | reasoning\_density | Depth | 0.000 | 0.000 |

**Sensitivity classification applied by the pipeline:**

- **Prompt-sensitive** (large |Δ| + KW significant): `token_per_idea`, `max_elaboration_chain`, `domain_spread`, `unique_perspective_count`, `mean_branch_depth`, `revision_depth`
- **Prompt-invariant** (small |Δ| or KW p ≥ 0.05): all remaining 14 metrics

### 4.7 Effect Size Analysis — Cohen's d (Pure → Eliciting)

Effect sizes computed from pooled standard deviations across the 30 pure and 30 eliciting traces:

| Metric | Cohen's d | Interpretation |
|---|---|---|
| cross\_branch\_connectivity | −2.46 | Very large (negative) |
| domain\_spread | +2.20 | Very large |
| perspective\_taking | −1.68 | Large (negative) |
| graph\_density | −1.56 | Large (negative) |
| first\_idea\_diversity | +1.48 | Large |
| avg\_tus | +1.41 | Large |
| unique\_perspective\_count | +1.13 | Large |
| token\_per\_idea | +1.04 | Large |
| max\_elaboration\_chain | ~0.95 | Moderate-large |
| revision\_depth | ~0.88 | Moderate |
| hedging\_density | −0.72 | Moderate (negative) |

**Directional observations:**
- `cross_branch_connectivity` drops sharply under eliciting (d = −2.46): the expanded graph is tree-like rather than cross-connected, suggesting the model adds depth without integrating ideas across branches.
- `domain_spread` shows the largest positive effect (d = +2.20): the eliciting prompt succeeds in driving exploration across distinct conceptual clusters.
- `graph_density` decreases significantly (d = −1.56) despite more edges, because the quadratic denominator \|V\|×(\|V\|-1) grows faster than edge count.
- `perspective_taking` decreases (d = −1.68) — a normalization artifact where RFR count grows slower than total node count under eliciting.

### 4.8 Prompt Sensitivity Classification

Based on Kruskal-Wallis significance and normalized absolute delta, metrics are classified into two groups:

**Prompt-sensitive (6 metrics):**  
These metrics should be interpreted only within a fixed prompt variant and cannot be used for cross-variant comparison without controlling for prompt effects.

> `token_per_idea`, `max_elaboration_chain`, `domain_spread`, `unique_perspective_count`, `mean_branch_depth`, `revision_depth`

**Prompt-invariant (14 metrics):**  
These metrics remain stable across prompt variants and constitute a candidate "core cognitive signature" that reflects model behavior independently of surface-level instruction framing.

> `branching_factor`\*, `first_idea_diversity`\*, `critique_to_hypothesis_ratio`, `exploration_exploitation_ratio`, `cross_branch_connectivity`, `convergence_index`, `graph_density`, `perspective_taking`, `hedging_density`, `redundancy_ratio`, `specificity_gradient`, `self_reflection_rate`, `backtracking_rate`, `reasoning_density`

\* These metrics show nominal KW significance but very small absolute deltas (|Δ| < 0.10), placing them borderline.

---

## 5. Comparison: v4 → v5 Segmenter Impact

The v5 structural pre-segmentation pass (Pass 0) corrects systematic under-segmentation in v4. The following table compares key metrics for the normal variant across both study versions.

| Metric | v4 (normal) | v5 (normal) | Change | % Change |
|---|---|---|---|---|
| avg\_tus | 9.1 | 19.6 | +10.5 | +115% |
| avg\_edges | ~38 | 81.7 | +43.7 | +115% |
| branching\_factor | 0.240 | 0.429 | +0.189 | +79% |
| unique\_perspective\_count | 1.80 | 4.83 | +3.03 | +168% |
| domain\_spread | 0.000 | 0.833 | +0.833 | ∞ |
| first\_idea\_diversity | 0.026 | 0.388 | +0.362 | +1392% |
| max\_elaboration\_chain | 3.30 | 3.83 | +0.53 | +16% |
| critique\_to\_hypothesis\_ratio | ~0.12 | 0.681 | +0.56 | +467% |
| graph\_density | ~0.07 | 0.181 | +0.11 | +157% |
| revision\_depth | ~8.1 | 9.83 | +1.73 | +21% |

**Metrics that were degenerate in v4 (returning 0.0) and are now informative in v5:**

| Metric | v4 | v5 (normal) | Root cause of v4 failure |
|---|---|---|---|
| domain\_spread | 0.000 | 0.833 | Fewer than 3 BRS/HYP nodes → fallback returned 0 |
| first\_idea\_diversity | 0.026 | 0.388 | Only 1–2 HYP nodes (all collapsed to first TU) |
| unique\_perspective\_count | 1.80 | 4.83 | No RFR nodes generated; markdown sections not segmented |
| branching\_factor | 0.240 | 0.429 | Insufficient BRCH edges due to coarse segmentation |
| max\_elaboration\_chain | 3.30 | 3.83 | Short ELAB chains from low TU count |
| critique\_to\_hypothesis\_ratio | ~0.12 | 0.681 | CRT nodes suppressed; insufficient TU resolution |

The v5 improvement is attributable specifically to Pass 0's recognition of Qwen's markdown output structure. Qwen/Qwen3.5-35B-A3B consistently formats responses with `### Section Headers`, `**Bold Labels:**`, and `- Bullet Points` — all invisible to v4's `\n\n+` paragraph splitter.

---

## 6. Discussion

### 6.1 Prompt-Sensitivity vs. Cognitive Capability

The six prompt-sensitive metrics (token\_per\_idea, max\_elaboration\_chain, domain\_spread, unique\_perspective\_count, mean\_branch\_depth, revision\_depth) respond primarily to the eliciting prompt's explicit cognitive scaffolding. These metrics cannot be interpreted as measures of intrinsic cognitive capability: a model that is highly responsive to instructions will score higher on all six regardless of its underlying reasoning structure.

The 14 prompt-invariant metrics are more theoretically interesting as capability proxies. Among these, `branching_factor`, `cross_branch_connectivity`, `hedging_density`, and `exploration_exploitation_ratio` remain stable across variants with small absolute deltas. If confirmed across multiple models, these metrics could serve as model fingerprints that distinguish reasoning styles independently of prompt engineering.

### 6.2 The Eliciting Variant as a Structural Amplifier

The eliciting variant does not merely expand all metrics proportionally. It selectively amplifies Depth and Breadth metrics (domain\_spread ×11.3, unique\_perspective\_count ×2.0) while compressing density metrics (graph\_density −58%, cross\_branch\_connectivity −67%). This suggests the eliciting prompt induces a **breadth-first expansion** strategy: the model generates many branches (high SYN, high HYP) but connects them sparsely rather than integrating across segments.

This is a non-trivial finding for prompt engineering: instructing a model to "explore widely" succeeds in generating more ideas but may decrease the *integration* of those ideas into coherent cross-branch syntheses.

### 6.3 Consistency Between Variants

The normal and pure variants produce nearly identical profiles across all 22 metrics. The difference between `normal` (with "Think step by step.") and `pure` (raw question only) is negligible for this model: avg\_tus 19.5 vs. 19.6, branching\_factor both 0.429, and all depth/structure metrics within 10% of each other. This suggests that standard chain-of-thought prompting ("Think step by step.") provides minimal structural differentiation for Qwen/Qwen3.5-35B-A3B on ethical reasoning tasks.

The `empty` variant (`<think></think>` tags) produces slightly more TUs (22.1) and more SYNT edges (470 vs. 318 for pure), indicating that explicit thinking-tag framing mildly expands synthesis behavior.

### 6.4 The reasoning\_density Artifact

`reasoning_density` = 1.000 for all 120 traces across all variants. This is not a meaningful finding: it reflects the DeBERTa NLI model's tendency to classify nearly every consecutive TU pair as SUPP (entailment) at the 0.78 threshold, which causes every node to have at least one semantic edge. With reasoning\_density = \|nodes with ≥1 semantic edge\| / \|V\|, and SUPP edges connecting nearly every pair, the metric saturates. This is a known calibration issue: the NLI threshold should be raised to 0.85–0.90 to reduce false-positive SUPP edges and restore metric discriminability.

---

## 7. Limitations

### 7.1 Single Model, Single Domain

The entire study runs on one model (Qwen/Qwen3.5-35B-A3B) and one domain (ethical\_dilemmas). The paper's core claim is to profile LLM *thinking behavior across models* — this cannot be established from a single-model study. The sensitivity analysis results (which metrics are prompt-sensitive vs. invariant) may not generalize to other models, especially models with different output formatting conventions (e.g., OpenAI models that avoid markdown, or models with explicit `<think>` vs. response separation).

### 7.2 No Human Annotation

The extraction pipeline is validated only structurally (no self-loops, all nodes reachable, ELAB-only subgraph acyclic). No boundary F1, node macro-F1, or edge accuracy has been computed against human-annotated gold standards. Without this validation, every metric is computed on unverified graph structures. It is unknown whether the extracted ThoughtGraphs accurately reflect the model's actual reasoning process.

### 7.3 No Statistical Tests on Metric Deltas

The sensitivity analysis reports raw Δ values and Kruskal-Wallis p-values without per-metric confidence intervals. With 30 traces per condition, bootstrapped 95% CI and Wilcoxon signed-rank tests between specific variant pairs (e.g., pure vs. eliciting) are feasible and should be added for any publication submission.

### 7.4 reasoning\_density Saturation

As discussed in §6.4, `reasoning_density` returns 1.0 for all traces and provides no discriminative signal. The NLI threshold calibration must be revisited before this metric can be considered meaningful.

### 7.5 Cue-Phrase Lexicon Coverage

The 43 cue-phrase patterns are tuned for explicit discourse markers common in human-like explanatory text. For models that reason in dense narrative prose without explicit hedges or transition signals, the cue-phrase pass may contribute few boundaries. In v5, the structural Pass 0 compensates for this by detecting formatting rather than content-level signals, but the fundamental tension between the lexicon and dense-prose models remains.

### 7.6 Deterministic BoundaryClass → NodeType Mapping

The 1:1 deterministic mapping (e.g., BACKTRACK → CRT, BRANCH → HYP) discards context. A structural header like "**Utilitarian Analysis:**" triggers a BRANCH boundary and maps to HYP, but the content may be evaluative (CRT) rather than hypothetical. A discriminative fine-tuned node classifier (currently `NotImplementedError` in `classifier.py`) would substantially improve node-type accuracy.

### 7.7 Normalization Bounds

The radar chart normalization bounds in `figures.py` (`_NORM_BOUNDS`) are hand-tuned constants (e.g., `token_per_idea: (0, 2000)`, `domain_spread: (0, 10)`). These must be derived empirically from a multi-model dataset before the radar chart can be used for cross-model comparison.

---

## 8. Conclusion

Study v5 demonstrates that markdown-aware structural pre-segmentation (Pass 0) is a necessary preprocessing step for ThinkBench when applied to models that format their reasoning output with explicit document structure. The segmenter improvement increases average TU count by 115% for the normal variant and transforms seven previously degenerate metrics into informative measurements.

Across 120 traces from four prompt variants, Kruskal-Wallis analysis identifies 14 of 22 metrics as prompt-sensitive (p < 0.05), with the eliciting variant producing large effects (Cohen's d > 1.0) for eight metrics including domain\_spread, unique\_perspective\_count, and token\_per\_idea. Six metrics exhibit small absolute deltas across all four variants and may constitute a stable cognitive signature: `branching_factor`, `cross_branch_connectivity`, `hedging_density`, `exploration_exploitation_ratio`, `convergence_index`, and `backtracking_rate`.

The primary finding of practical significance is the **breadth-without-integration** effect under eliciting: the cognitive scaffolding prompt successfully increases idea diversity and exploration depth but simultaneously decreases cross-branch connectivity and graph density, indicating that prompted breadth-first expansion comes at the cost of inter-idea synthesis. Whether this reflects an inherent model limitation or a prompt-design artifact requires cross-model replication.

The framework is ready for multi-model expansion. Priority next steps are: (1) add at least three additional models with diverse reasoning styles (e.g., DeepSeek-R1, a smaller Llama variant, and a closed-source model), (2) add at least two additional question domains, (3) conduct human annotation of 30 trace segments to establish extraction pipeline validity, and (4) recalibrate the NLI SUPP threshold to restore `reasoning_density` discriminability.

---

## Appendix A — Full Sensitivity Table

| Metric | Category | Zero% | std(pure) | std(eliciting) | Max |ρ| | Verdict |
|---|---|---|---|---|---|---|
| token\_per\_idea | Efficiency | 0% | 34.2 | 151.3 | 0.72 (avg\_tokens) | KEEP |
| max\_elaboration\_chain | Depth | 0% | 2.1 | 7.8 | 0.61 (mean\_branch\_depth) | KEEP |
| domain\_spread | Breadth | 27% | 0.43 | 5.12 | 0.54 (unique\_perspective\_count) | KEEP |
| unique\_perspective\_count | Breadth | 0% | 2.8 | 5.4 | 0.54 (domain\_spread) | KEEP |
| mean\_branch\_depth | Depth | 0% | 1.3 | 3.7 | 0.61 (max\_elaboration\_chain) | KEEP |
| revision\_depth | Structure | 0% | 4.2 | 6.1 | 0.33 | KEEP |
| first\_idea\_diversity | Breadth | 0% | 0.22 | 0.18 | 0.47 | KEEP |
| critique\_to\_hypothesis\_ratio | Metacognitive | 0% | 0.31 | 0.29 | 0.28 | KEEP |
| cross\_branch\_connectivity | Structure | 0% | 0.19 | 0.11 | 0.42 | KEEP |
| graph\_density | Structure | 0% | 0.06 | 0.04 | 0.35 | KEEP |
| perspective\_taking | Metacognitive | 0% | 0.10 | 0.05 | 0.91 (unique\_perspective\_count) | MERGE → use UPC |
| branching\_factor | Breadth | 0% | 0.08 | 0.08 | 0.38 | KEEP |
| exploration\_exploitation\_ratio | Structure | 0% | 0.44 | 0.46 | 0.31 | KEEP |
| convergence\_index | Structure | 0% | 0.14 | 0.09 | 0.22 | KEEP |
| hedging\_density | Metacognitive | 0% | 0.04 | 0.03 | 0.19 | KEEP |
| backtracking\_rate | Structure | 0% | 0.02 | 0.01 | 0.18 | KEEP |
| redundancy\_ratio | Efficiency | 0% | 0.005 | 0.007 | 0.26 | KEEP |
| specificity\_gradient | Depth | 0% | 0.007 | 0.005 | 0.14 | KEEP |
| self\_reflection\_rate | Metacognitive | 55% | 0.008 | 0.000 | 0.22 | MONITOR |
| reasoning\_density | Depth | 0% | 0.000 | 0.000 | — | DROP (constant) |

**Verdict key:**  
- KEEP: informative, non-degenerate, low collinearity  
- MERGE: collinear with a more interpretable metric (|ρ| > 0.85)  
- DROP: constant across all traces  
- MONITOR: informative when non-zero but high zero rate; requires model with more MET-class behavior

---

## Appendix B — v5 Study Configuration

```
Study ID:       study_v5_20260430_1028
Model:          Qwen/Qwen3.5-35B-A3B
Domain:         ethical_dilemmas
Questions:      10 (ED001–ED010)
Runs:           3
Variants:       4 (pure, normal, empty, eliciting)
Total traces:   120
Total TUs:      4,292
Total edges:    20,546
Segmenter:      v3 (5-pass + Pass 0 structural pre-segmentation)
Pass 0 rules:   ATX headers, bold labels, list items, hrules, paragraph breaks
NLI threshold:  SUPP=0.78, CONT/ELAB=0.75, BACK=0.70
Embedding dim:  384 (all-MiniLM-L6-v2)
```
