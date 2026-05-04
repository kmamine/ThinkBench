# ThinkBench Segmentation Pipeline: Evolution from v1 to v3

**Date**: 2026-04-27  
**Scope**: Technical history of `src/thinkbench/extract/segmenter.py` across three design generations.

---

## Overview

The segmentation pipeline converts a raw chain-of-thought (CoT) trace into a list of **Thought Units** (TUs) — typed text segments — and a list of typed directed **Edges** between them. This is the foundational step: every downstream metric depends on what the segmenter produces.

The pipeline has gone through three substantively distinct designs, driven by two major design pivots:

1. **v1 → v2**: Remove the generative LLM from extraction (circular validity fix).  
2. **v2 → v3**: Replace surface-pattern edge detection with embedding-trajectory-based detection (model-agnostic robustness fix).

---

## v1 — Generative LLM Segmenter

**Commit**: `820e2d2` ("adding the pipeline code")  
**Lines**: ~200  
**Returns**: `list[ThoughtUnit]`

### Architecture

A single LLM call via an `async def segment()` method on a `Segmenter` class. The model received the full raw CoT trace and was instructed to output a JSON array of thought units — each with its text, ID, and character offsets.

```
Raw CoT → LLM prompt → JSON parse → list[ThoughtUnit]
```

**Fallback chain** (when JSON parsing failed):
1. Regex split on headers and numbered list markers
2. Regex split on sentence boundaries

No edge types were assigned; the segmenter returned TUs only. Edge classification was presumably handled downstream or not at all at this stage.

### Key properties

| Property | Value |
|---|---|
| Boundary detection | LLM-generated (instructed via prompt rules) |
| Edge assignment | None |
| Model dependency | Requires LLM at extraction time |
| Reproducibility | Non-deterministic (temperature > 0) |
| Failure mode | Silent JSON parse failure → coarse fallback |

### Design rationale

At v1, ThinkBench was conceived as a fully LLM-driven pipeline. The reasoning was that a capable LLM could understand discourse structure better than hand-crafted rules. The segmenter prompt specified linguistic boundary cues (paragraph breaks, "Wait,", "Actually,", "Alternatively,", etc.) and asked the model to apply them.

### Critical flaw

The LLM doing the segmentation is of the same family as the LLMs being analysed. This introduces **circular validity**: a GPT-4 segmenter may systematically segment GPT-4 traces differently from Qwen or DeepSeek traces — not because the reasoning differs, but because the segmenter recognises its own writing patterns. Results cannot be compared across models. This was the primary motivation for the v2 rewrite.

---

## v2 — Non-Generative 3-Pass Pipeline

**Commit**: `fa67fce` ("v2 migration: non-generative extraction pipeline")  
**Lines**: ~431  
**Returns**: `(list[ThoughtUnit], list[Edge])`

### Architecture

Three sequential passes, no LLM involved:

```
Raw CoT
  │
  ├─ Pass 1 — Hard boundaries (regex cue phrases)
  │    All 7 BoundaryClass types detected via pattern matching
  │    Short spans merged into following span
  │
  ├─ Pass 2 — Soft boundaries (TextTiling, MiniLM)
  │    Cosine similarity between sentence windows
  │    Local minima below τ = 30th percentile → NONE-class boundaries
  │
  └─ Pass 3 — NLI edge promotion (DeBERTa cross-encoder)
       Local window (≤4 TUs): SEQ → SUPP / CONT / ELAB (threshold 0.75)
       Non-local BACKTRACK pairs: → BACK (threshold 0.70)
       Single batched predict() call per trace
```

### Pass 1 — Hard boundaries

Seven `BoundaryClass` types were all detected by regex cue-phrase patterns applied to the first sentence of each candidate span:

| BoundaryClass | Example phrases |
|---|---|
| BACKTRACK | "no wait", "actually", "hmm", "let me reconsider", "i was wrong" |
| BRANCH | "alternatively", "another approach", "what if instead", "or perhaps" |
| META | "let me step back", "i'm overcomplicating this", "let me re-read" |
| CONVERGENCE | "therefore", "to summarize", "so the key insight is", "the answer is" |
| ELABORATION | "specifically", "for example", "more precisely", "in other words" |
| CONTRAST | "however", "on the other hand", "in contrast", "but" |
| SUPPORT | "because", "since", "the reason is", "evidence for this" |

Short spans (< 3 sentences AND < 50 tokens) were merged into the following span to avoid over-segmentation.

### Pass 2 — Soft boundaries

Sentence embeddings (MiniLM-L6-v2) were computed for each NONE-class hard span. A sliding window (width = 3) computed cosine similarity at each sentence boundary. Local minima with prominence ≥ 0.15 and similarity below a per-trace τ (30th percentile of within-span sentence similarities) were promoted to NONE-class TU boundaries.

This was the first appearance of embedding-based detection, but only as a secondary "gap filler" for spans where cue phrases had not already assigned a boundary class.

### Pass 3 — NLI edge promotion

For all SEQ-typed adjacent pairs (within window of 4), the DeBERTa cross-encoder tested three hypotheses:
- SUPP: "The second passage provides evidence or justification for the first."
- CONT: "The second passage contradicts or opposes the first."
- ELAB: "The second passage is a more specific or detailed version of the first."

The best-scoring hypothesis above 0.75 replaced the SEQ edge. BACKTRACK spans > 4 apart were tested for a BACK hypothesis (threshold 0.70), linking each backtracking TU to the first prior TU above threshold.

All pairs were batched into a single `predict()` call per trace.

### Critical flaw

The design worked well for models that use **explicit discourse markers** ("alternatively", "however", "because"). When tested on Qwen3.5-35B-A3B — which reasons in continuous, dense prose without explicit transitions — Pass 1 produced almost no BRANCH, ELABORATION, CONTRAST, or SUPPORT boundaries. The result: all TUs were classified SPC (default) or HYP (first TU), and the following metrics collapsed to 0.0 across all 90 traces:

- `branching_factor` (no BRCH edges → 0)
- `unique_perspective_count` (no RFR nodes → 0)
- `first_idea_diversity` (only 1 HYP node → 0)
- `self_reflection_rate` (no MET nodes → 0)
- `perspective_taking` (no RFR nodes → 0)
- `backtracking_rate` (no BACK edges → 0)

The 30-phrase cue-phrase lexicon was tuned for human-written think-aloud protocols and verbose reasoning models. It was fundamentally not portable to dense-prose LLMs.

---

## v3 — Semantic Trajectory 5-Pass Pipeline

**Current file**: `src/thinkbench/extract/segmenter.py`  
**Lines**: 556  
**Returns**: `(list[ThoughtUnit], list[Edge], np.ndarray)`

### Architecture

```
Raw CoT
  │
  ├─ Pass 1 — Supplementary cue phrases (META / CONVERGENCE / BACKTRACK only)
  │    3 out of 7 boundary types; BRANCH / ELABORATION / CONTRAST / SUPPORT
  │    are no longer surface-detected — handled by Layer 3
  │
  ├─ Pass 2 — TextTiling soft boundaries (unchanged from v2)
  │    MiniLM sentence embeddings, window=3, prominence≥0.15
  │    τ = 30th percentile of within-trace sentence similarities
  │
  ├─ Layer 3 — Semantic trajectory (NEW — primary structural detection)
  │    Per-TU cosine similarity to previous TU
  │    τ_branch = 25th percentile → sim < τ_branch ⟹ BRCH edge
  │    τ_elab   = 65th percentile → sim ≥ τ_elab   ⟹ ELAB edge
  │    Both thresholds are per-trace, data-derived
  │    Cue-phrase overrides: CONVERGENCE → SYNT, BACKTRACK → BACK
  │
  ├─ Layer 4 — Cross-segment analysis (NEW)
  │    Synthesis: cos(emb_j, centroid_s) > 0.50 for ≥ 2 prior segments → SYNT
  │    Revision loop: cos(emb_j, emb_i) > 0.65 + gap drop < 0.45 → BACK
  │    (gap confirms genuine departure between i and j)
  │
  └─ Pass 5 — NLI refinement (demoted, only for untyped SEQ transitions)
       Only runs on pairs not already typed by Layers 3/4
       Threshold raised: 0.75 → 0.78 (reduce near-universal SUPP over-triggering)
       Window reduced: 4 → 3 TUs
```

### Design changes from v2

#### 1. Cue phrases demoted from primary to supplementary signal

In v2, cue phrases were the primary mechanism for assigning all seven boundary types. If a phrase didn't match, the TU got NONE and was left to NLI.

In v3, cue phrases are **only used for three types** — META, CONVERGENCE, BACKTRACK — where the explicit surface marker carries unique semantic information beyond what embedding similarity can capture (e.g., "stepping back" signals metacognition; the embedding shift alone cannot distinguish this from a new hypothesis). The other four types (BRANCH, ELABORATION, CONTRAST, SUPPORT) are now assigned entirely from the embedding trajectory.

#### 2. Adaptive per-trace thresholds replace fixed calibration

v2 used a single fixed threshold: τ = 30th percentile of within-span sentence similarities. This was applied globally to detect soft boundaries and was the same for every trace and every model.

v3 computes **two adaptive thresholds per trace** from the distribution of consecutive TU similarities:

- **τ_branch = 25th percentile**: The lowest-similarity transitions are topic shifts → BRCH.
- **τ_elab = 65th percentile**: The highest-similarity transitions are elaborations → ELAB.
- **Between τ_branch and τ_elab**: Neutral continuation → SEQ.

Because both thresholds are derived from the trace's own similarity distribution, the pipeline calibrates to the model's average transition distance. A verbose, repetitive model will have a higher absolute τ_branch than a terse model — but the relative classification (lowest 25% → BRCH) is consistent. This makes the pipeline **model-agnostic without retuning**.

#### 3. Cross-segment analysis (Layer 4) — new structural layer

v2 had no mechanism for detecting cross-branch synthesis or revision loops beyond the local NLI window of 4 TUs. Long-range semantic returns (a TU 20 steps later returning to the topic of TU 3) were invisible.

Layer 4 adds two detectors operating over the full trace:

**Synthesis detection**: For each TU _j_, compute cosine similarity to the centroid of every prior topic segment. If _j_ is highly similar (> 0.50) to ≥ 2 distinct prior segments, it is a synthesis point → SYNT edges from _j_ to the representative TU of each bridged segment. The TU's `boundary_class` is updated to CONVERGENCE for the downstream classifier.

**Revision loop detection**: For each TU _j_, scan up to 25 positions back for a TU _i_ (minimum gap = 4) in a different topic segment where cos(emb_j, emb_i) > 0.65. A gap-drop criterion (minimum similarity in the gap between _i_ and _j_ must be < 0.45) confirms that the trace genuinely departed from topic _i_ before returning — ruling out locally clustered sentences. → BACK edge from _j_ to _i_, `boundary_class` updated to BACKTRACK.

#### 4. NLI role reduced to gap-filling

In v2, NLI ran on **all** SEQ-typed local pairs within window 4. Because BRANCH, ELABORATION, CONTRAST, and SUPPORT were not assigned by cue phrases on dense-prose models, nearly every pair was SEQ, and the NLI ran on the full O(n) adjacency list. The result was near-universal SUPP classification (DeBERTa finds weak entailment between almost any two consecutive sentences), giving `reasoning_density ≈ 1.0` for all traces — an artifact rather than a signal.

In v3, NLI only runs on pairs that are **still untyped** (SEQ edge, no semantic edge from Layers 3/4). Since Layer 3 already assigns BRCH and ELAB to most transitions, NLI sees a much smaller input. The threshold was also raised from 0.75 to 0.78 to further reduce SUPP over-triggering.

#### 5. Embeddings returned to caller

v2 returned `(tus, edges)`. v3 returns `(tus, edges, embeddings)` — the (n_tus × 384) TU embedding matrix. This is passed directly to `classify_nodes()` in the classifier, enabling:

- **RFR vs. HYP disambiguation**: cos(emb_i, emb_0) range check for branch-receiving TUs.
- **Embedding-based MET detection**: TUs in the second half of the trace that return to the topic of TU₀ (cos ≥ 0.60) are classified as meta-reflection without requiring a cue phrase.

In v2, the classifier had no access to embeddings and could not make these distinctions.

---

## Comparative Summary

| Dimension | v1 | v2 | v3 |
|---|---|---|---|
| **Generative LLM required** | Yes | No | No |
| **Boundary detection mechanism** | LLM prompt | Cue phrases (7 types) | Cue phrases (3 types) + embedding trajectory |
| **Edge types from cue phrases** | None (no edges) | BACK, BRCH, META→SEQ, SYNT, ELAB, CONT, SUPP | BACK, META→SEQ, SYNT only |
| **Edge types from embeddings** | None | Soft boundary (NONE class) | BRCH, ELAB (Layer 3); SYNT, BACK (Layer 4) |
| **Threshold calibration** | N/A | Fixed (30th pct) | Adaptive per-trace (25th + 65th pct) |
| **Cross-branch synthesis detection** | None | NLI (window 4) | Layer 4 centroid matching (full trace) |
| **Revision loop detection** | None | NLI BACK hypothesis (first match) | Layer 4 gap-drop criterion (best match) |
| **NLI role** | None | Primary edge promoter | Gap-filler for untyped SEQ pairs only |
| **NLI threshold** | N/A | 0.75 (local), 0.70 (BACK) | 0.78 (local only) |
| **Embeddings returned** | No | No | Yes (passed to classifier) |
| **Token minimum to process** | None | 100 | 80 |
| **Short span threshold** | None | <3 sent AND <50 tok | <2 sent AND <30 tok |
| **Lines of code** | ~200 | ~431 | 556 |
| **Works on dense-prose models** | Depends on LLM | No (cue-phrase dependent) | Yes (trajectory-based) |

---

## Impact on Metric Coverage

The table below compares the fraction of metrics that returned non-zero values across the 90-trace study, per pipeline version (v2 estimated from April 23 run; v3 from April 24 run):

| Metric | v2 (estimated) | v3 (actual) |
|---|---|---|
| `branching_factor` | 0.0 (no BRCH edges) | 0.39 ± 0.12 |
| `unique_perspective_count` | 0.0 (no RFR nodes) | 4.1 ± 2.8 |
| `first_idea_diversity` | 0.0 (<2 HYP nodes) | 0.48 ± 0.21 |
| `self_reflection_rate` | 0.0 (no MET nodes) | 0.07 ± 0.06 |
| `perspective_taking` | 0.0 (no RFR nodes) | 0.23 ± 0.15 |
| `backtracking_rate` | 0.0 (no BACK edges) | 0.12 ± 0.09 |
| `token_per_idea` | = avg_tokens (bug) | 198 ± 84 (rate) |
| `domain_spread` | 1.0 (degenerate) | 0.0 or 2.4 ± 1.1 |
| `reasoning_density` | ≈1.0 (NLI artifact) | 0.68 ± 0.11 |

The v3 pipeline converted 8 degenerate metrics (returning a constant for all traces) into metrics with genuine within-trace and between-variant variance.

---

## Remaining Limitations

1. **Pass 1 lexicon coverage**: The 30+ cue phrases for META/CONVERGENCE/BACKTRACK are still manually curated. Models that express these states without standard markers (e.g., through parenthetical asides rather than sentence-initial cues) will miss these boundary types.

2. **NLI disabled by default**: Pass 5 requires DeBERTa inference. In the current study it was disabled (`use_nli=False`) to isolate the embedding-trajectory signal. Re-enabling it should improve SUPP/CONT edge coverage for the final multi-model study.

3. **Single-newline traces**: The segmenter splits on double-newline (`\n\n`) for paragraph-level spans before sentence-level analysis. Traces formatted with single newlines between paragraphs (common in some model outputs) bypass Pass 1 splitting and enter Pass 2 as a single span, increasing NLI load.

4. **Fixed embedding model**: `all-MiniLM-L6-v2` (384 dimensions) is used for speed. The adaptive thresholds make the pipeline less sensitive to absolute similarity values, but a higher-capacity model (`gte-large-en-v1.5`, 1024 dimensions) would improve precision of BRCH/ELAB boundary placement, particularly for long traces.
