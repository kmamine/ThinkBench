# ThinkBench — File Description

> Complete listing of every file in the project with its path and functional description.
> Excludes auto-generated caches (`__pycache__`, `.ipynb_checkpoints`), `.git/`, `.agent/`, `.claude/`, and large binary result directories.

---

## Root-Level Configuration & Documentation

| File | Description |
|---|---|
| `CLAUDE.md` | Master implementation specification. Defines the full project architecture, tech stack, directory layout, all Pydantic data models (`ThoughtUnit`, `Edge`, `ThoughtGraph`, `CognitiveProfile`), the 12 node types / 4 node families / 8 edge types, the 22-metric computation order, build phases, CLI commands, and critical design notes (why non-generative, why directed graphs with cycles, why open-ended questions). The authoritative source of truth for implementation. |
| `AGENT.md` | Agent-mode operational instructions for Claude Code. Describes how to autonomously execute experiments, which scripts to call in what order, and how to interpret outputs. |
| `NEW_METHOD.md` | Conceptual notes on the new segmentation approach (v3 semantic-trajectory pipeline). Records the rationale for replacing cue-phrase-only segmentation with embedding-based BRCH/ELAB detection. |
| `README.md` | Public-facing project overview: what ThinkBench is, how to install, quick-start usage. |
| `THINKBENCH_METHOD.md` | Detailed method writeup describing the theoretical framework — Thought Graphs, cognitive profile dimensions, and the benchmark design. |
| `pyproject.toml` | Python package declaration (`thinkbench`, version 0.1.0, requires Python ≥ 3.11). Lists all runtime dependencies: `openai`, `pydantic`, `networkx`, `sentence-transformers`, `transformers`, `torch`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `typer`, etc. Configures pytest to run from `tests/`. |
| `thinkbench.toml` | Additional project-level configuration (experiment defaults, endpoint settings). |
| `file_description.md` | This file. |

---

## Root-Level Test Scripts

These scripts live at the project root and are standalone integration tests / smoke tests that hit the live vLLM endpoint.

| File | Description |
|---|---|
| `test_collection_params.py` | Smoke test for collection parameter variants (different temperature, max_tokens, prompt variants). Verifies the `LLMClient` correctly applies each parameter. |
| `test_collector.py` | End-to-end test of `TraceCollector.collect_batch()`. Loads questions, collects a small batch, and inspects the returned records for required fields (`trace_id`, `raw_cot`, `token_count`). |
| `test_extraction.py` | Integration test for the full extraction pipeline on existing traces. Calls `segment()` → `classify_nodes()` → `build_graph()` and prints node/edge counts per graph. |
| `test_metrics.py` | Unit test for metric computation. Loads a single graph JSON from `data/graphs/graph_test.json`, runs `compute_profile()`, and prints all 22 metric values. |

---

## Data

### `data/questions/`

| File | Description |
|---|---|
| `data/questions/ethical_dilemmas.jsonl` | 10 open-ended ethical dilemma questions used in all benchmark runs. Each line is a JSON object with fields: `id` (`D1_001`–`D1_010`), `domain` (`ethical_dilemmas`), `text` (the question), `expected_angles` (list of reasoning perspectives the question should elicit), `difficulty` (`medium` / `hard`). |
| `data/questions/test_2q.jsonl` | Minimal 2-question subset used for fast pipeline smoke tests and development iteration. |

### `data/archive/`

| File | Description |
|---|---|
| `data/archive/2025-04-15_README.md` | Notes on the April 2025 archival run — which model, what parameters, what was learned. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B.jsonl` | Baseline traces (3 runs per question, default temperature). |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_high_k3.jsonl` | Traces collected with `thinking_effort=HIGH`, k=3 runs. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_high_k3_q10.jsonl` | High-effort traces, 10 questions, k=3. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_high_k5.jsonl` | High-effort traces, k=5 runs. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_low_k3.jsonl` | Low-effort traces, k=3. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_low_k3_q10.jsonl` | Low-effort, 10 questions, k=3. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_low_k5.jsonl` | Low-effort, k=5. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_max_k5.jsonl` | Max-effort traces, k=5. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_medium_k3.jsonl` | Medium-effort traces, k=3. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_medium_k3_q10.jsonl` | Medium-effort, 10 questions, k=3. |
| `data/archive/2025-04-15_traces/traces_Qwen-Qwen3.5-35B-A3B_medium_k5.jsonl` | Medium-effort, k=5. |

---

## Source Package — `src/thinkbench/`

### Package Root

| File | Description |
|---|---|
| `src/thinkbench/__init__.py` | Package init (empty or minimal imports). |
| `src/thinkbench/cli.py` | Typer-based CLI entrypoint. Exposes the `thinkbench collect` command: takes `--questions`, `--model`, `--runs`, `--output`, `--api-key`, `--endpoint` arguments, instantiates `LLMClient` + `TraceCollector`, and runs `collect_batch()`. Entry point for `thinkbench` binary defined in `pyproject.toml`. |

---

### `src/thinkbench/collect/` — CoT Trace Collection

| File | Description |
|---|---|
| `src/thinkbench/collect/__init__.py` | Re-exports `LLMClient`, `TraceCollector`, `PROMPT_VARIANTS`. |
| `src/thinkbench/collect/models.py` | `LLMClient` — async OpenAI-compatible client for vLLM endpoints. Defines the four prompt variants: `empty` (no system turn), `pure` ("You are a helpful assistant."), `normal` (+ "Think carefully before answering."), `eliciting` (full structured reasoning framing with 7 explicit cognitive instructions). `SYSTEM_PROMPTS` dict and `PROMPT_VARIANTS` list are imported throughout. `chat()` method strips and re-injects the system turn, calls the vLLM endpoint, and handles reasoning-model responses (`.reasoning` field fallback). `chat_with_retries()` wraps with exponential backoff. |
| `src/thinkbench/collect/collector.py` | `TraceCollector` — async batch trace collection. `collect_single()` generates a UUID `trace_id`, sends the question to `LLMClient.chat()`, and returns a record dict with `trace_id`, `model`, `prompt_variant`, `question_id`, `domain`, `run`, `raw_cot`, `token_count`, `collected_at`. `collect_batch()` iterates questions × runs, saves in batches of `questions_per_batch` for crash resilience. `_save_traces()` deduplicates by `trace_id` and appends to JSONL files named `traces_{model_slug}_{variant}.jsonl`. |

---

### `src/thinkbench/extract/` — Thought Graph Extraction

| File | Description |
|---|---|
| `src/thinkbench/extract/__init__.py` | Re-exports the main extraction symbols. |
| `src/thinkbench/extract/schemas.py` | All Pydantic data models for the extraction pipeline. **Enums**: `NodeType` (12 types: HYP, RFR, ANA, BRS, JUS, SPC, IMP, CON, CRT, CMP, MET, SYN), `NodeFamily` (EXPLORATION / ELABORATION / EVALUATION / CONVERGENCE), `EdgeType` (ELAB, BRCH, BACK, SYNT, CRIT, SUPP, CONT, SEQ), `BoundaryClass` (BACKTRACK / BRANCH / META / CONVERGENCE / ELABORATION / CONTRAST / SUPPORT / NONE). **Maps**: `NODE_FAMILY_MAP` (NodeType → NodeFamily), `BOUNDARY_EDGE_MAP` (BoundaryClass → EdgeType), `BOUNDARY_NODE_MAP` (BoundaryClass → NodeType, with NONE → HYP for the first TU). **Models**: `ThoughtUnit` (tu_id, text, char offsets, token_count, boundary_class, node_type, node_family, confidence), `Edge` (source, target, edge_type, confidence, is_sequential), `ThoughtGraph` (trace_id, model, question_id, domain, run, nodes, edges, token_count), `CognitiveProfile` (all 22 metric fields + `to_vector()`). |
| `src/thinkbench/extract/segmenter.py` | 5-pass non-generative segmentation pipeline. **Pass 1** — supplementary cue-phrase detection: regex patterns for META (`"let me step back"`, `"i'm going in circles"`, etc.), CONVERGENCE (`"in summary"`, `"therefore"`, `"putting this together"`, etc.), and BACKTRACK (`"wait"`, `"but actually"`, `"no wait"`, etc.). Splits text at these boundaries and merges spans shorter than 2 sentences / 30 tokens. **Pass 2** — TextTiling: encodes sentences with `all-MiniLM-L6-v2`, slides a window of 3, computes cosine similarity, finds local minima via `scipy.signal.find_peaks` (prominence ≥ 0.15). `τ` is calibrated at the 30th percentile of within-trace similarities. **Layer 3** — semantic trajectory: for each consecutive TU pair, classifies the transition as BRCH (sim < 25th pct), ELAB (sim ≥ 65th pct), or SEQ. Cue-phrase classes (CONVERGENCE → SYNT, BACKTRACK → BACK, META → SEQ) override the cosine classification. Tracks segment IDs for each TU. **Layer 4** — cross-segment analysis: SYNT detection (TU with cosine sim > 0.50 to centroids of ≥2 prior segments), BACK detection (gap-drop criterion: similarity dropped in the 3 TUs after the target before returning). **Pass 5** — NLI refinement: runs `cross-encoder/nli-deberta-v3-large` on untyped SEQ pairs within window=3, promotes to SUPP/CONT/ELAB at threshold 0.78. All NLI pairs batched in a single `predict()` call. Public API: `segment(raw_cot, use_nli=True) → (tus, edges, embeddings)`. |
| `src/thinkbench/extract/classifier.py` | 8-priority rule-based node type classifier. **Priority 1**: outgoing SYNT edges to ≥2 distinct targets → SYN. **Priority 2**: outgoing BACK edge → CRT. **Priority 3**: `boundary_class == META` → MET. **Priority 4/5**: incoming BRCH edge + embedding similarity to TU₀ in (0.25, 0.72) → RFR (reframing); outside range → HYP. **Priority 6**: outgoing CONT edge → CMP. **Priority 7**: outgoing SUPP edge → JUS. **Priority 8**: `boundary_class == CONVERGENCE` → SYN. **Default**: idx==0 → HYP; embedding-based MET detection (sim to TU₀ > 0.60 in second half of trace); else → SPC. Confidence is 1.0 for hard-boundary TUs, 0.5 for soft. `load_deberta_classifier()` raises `NotImplementedError` (planned fine-tuned node classifier). Public API: `classify_nodes(tus, edges, embeddings) → tus`. |
| `src/thinkbench/extract/linker.py` | Graph assembler. `build_graph(trace, thought_units, edges) → ThoughtGraph` — constructs the `ThoughtGraph` Pydantic object from the pre-computed TUs and edges produced by the segmenter, filling in metadata from the trace dict. |

---

### `src/thinkbench/metrics/` — Cognitive Profile Computation

| File | Description |
|---|---|
| `src/thinkbench/metrics/__init__.py` | Re-exports all 20 metric functions and `compute_profile` / `aggregate_profiles`. |
| `src/thinkbench/metrics/breadth.py` | **4 breadth metrics**. `branching_factor`: `|E_BRCH| / |V|`. `unique_perspective_count`: count of RFR nodes. `domain_spread`: agglomerative clustering (cosine threshold 0.45) on embeddings of BRS+HYP nodes — returns number of semantic clusters (0.0 if fewer than 3 target nodes). `first_idea_diversity`: mean pairwise cosine distance among the first 3 HYP nodes. All embedding calls use the `get_embed_model()` singleton. |
| `src/thinkbench/metrics/depth.py` | **4 depth metrics**. `max_elaboration_chain`: longest directed path in the ELAB-only subgraph (via `nx.dag_longest_path_length`). `mean_branch_depth`: average BFS depth from root nodes (in-degree=0) in the semantic subgraph. `specificity_gradient`: linear regression slope of concrete-token ratio (numbers, capitalised mid-sentence words, symbols) vs sequential position — positive slope = reasoning becomes more grounded. `reasoning_density`: fraction of nodes participating in at least one non-SEQ edge. |
| `src/thinkbench/metrics/structure.py` | **6 structure metrics**. `exploration_exploitation_ratio`: `|EXPLORATION nodes| / max(|ELABORATION nodes|, 1)`. `backtracking_rate`: `|E_BACK| / max(|E_sem|, 1)` (returns 0.0 when no semantic edges). `cross_branch_connectivity`: fraction of branch-component pairs connected by at least one SYNT or SUPP edge. `convergence_index`: weighted in-degree of SYN nodes normalised by graph size and mean in-degree. `graph_density`: `|E_sem| / (|V| * (|V|-1))`. `revision_depth`: mean `|source - target|` over all BACK edges. |
| `src/thinkbench/metrics/metacognitive.py` | **4 metacognitive metrics**. `self_reflection_rate`: proportion of MET nodes. `critique_to_hypothesis_ratio`: `|CRT| / |HYP|` (returns 0.0 when no HYP nodes). `hedging_density`: proportion of TUs matching any of 16 uncertainty-marker regex patterns (`might`, `maybe`, `could`, `perhaps`, `possibly`, `probably`, `it seems`, `unclear`, `not sure`, `likely`, etc.). `perspective_taking`: proportion of RFR nodes. |
| `src/thinkbench/metrics/efficiency.py` | **2 efficiency metrics**. `token_per_idea`: `total_tokens / |N_RFR|` — falls back to `total_tokens / max(|V|, 1)` when no RFR nodes. `redundancy_ratio`: fraction of TU pairs with cosine similarity > 0.75 (near-duplicate content detection). Uses `get_embed_model()` singleton. |
| `src/thinkbench/metrics/profile.py` | **Profile aggregation**. `compute_profile(graph, model, domain) → CognitiveProfile`: calls all 20 metric functions and packages results into a `CognitiveProfile` with `avg_tokens = graph.token_count` and `avg_tus = len(graph.nodes)`. `aggregate_profiles(profiles, trace_model_map, trace_prompt_map) → list[dict]`: groups profiles by `(model, prompt_variant)`, averages all numeric fields, and returns one dict per group with `num_traces` count — the format consumed by analysis and figure functions. |

---

### `src/thinkbench/analysis/` — Sensitivity Analysis & Visualization

| File | Description |
|---|---|
| `src/thinkbench/analysis/__init__.py` | Re-exports everything from `compare.py` and `figures.py`. |
| `src/thinkbench/analysis/compare.py` | Prompt sensitivity analysis. **Constants**: `COGNITIVE_METRICS` (ordered list of 20 metric names, excluding `avg_tokens`/`avg_tus`), `METRIC_CATEGORIES` (dict mapping category name to metric list), `CATEGORY_OF` (flat reverse map). **`compute_deltas(profiles_agg, baseline="pure")`**: computes per-metric delta of each variant vs. the baseline variant. **`prompt_sensitivity(profiles_agg, baseline="pure")`**: ranks metrics by `|Δ(eliciting − pure)|`, returns list sorted by `abs_sensitivity` descending with fields `metric`, `category`, `delta_eliciting`, `delta_normal`, `abs_sensitivity`. **`classify_metrics(sensitivity, top_n=6)`**: splits into `prompt_sensitive` (top N by Δ) and `prompt_invariant` (rest). |
| `src/thinkbench/analysis/figures.py` | All 14+ publication-quality matplotlib figures. **Constants**: `VARIANT_COLOR`, `VARIANT_LABEL`, `VARIANTS` (controls which variants all functions iterate — currently `["empty", "pure", "normal", "eliciting"]`), `COGNITIVE_METRICS`, `_NORM_BOUNDS` (per-metric normalisation bounds for radar plots). **Main figures**: `radar_comparison` (overlaid radar chart, all 20 metrics, all variants), `sensitivity_bar` (horizontal dual-panel bar ranked by Δ), `metric_heatmap` (variants × metrics heatmap, min-max normalised), `category_comparison` (grouped bar per category), `metric_violin` (per-metric violin plots coloured by variant — dynamically adapts to `len(VARIANTS)`), `delta_violin` (Cohen's d effect-size bars per metric), `correlation_heatmap` (Spearman ρ heatmap across per-trace profiles), `pca_traces` (2D PCA scatter coloured by variant), `parallel_coordinates` (per-metric parallel axis plot). **Graph figures**: `graph_examples` (3-column grid of representative graphs for each variant), `node_edge_distributions` (stacked bar of node-type and edge-type distributions per variant), `graph_grid_all` (all graphs in a paginated grid written to supplement directory — one page per variant). **Supplement**: `scatter_key_pairs` (pairwise scatter of key metrics), `cluster_dendrogram` (Ward-linkage dendrogram on (1−|ρ|) distance), `per_trace_profile_heatmap` (each trace as a row, each metric as a column). `generate_all()` calls every figure function in sequence. |

---

### `src/thinkbench/utils/` — Shared Infrastructure

| File | Description |
|---|---|
| `src/thinkbench/utils/__init__.py` | Empty init. |
| `src/thinkbench/utils/models.py` | Lazy global singletons for the two expensive ML models. `get_embed_model()`: loads `all-MiniLM-L6-v2` via `sentence_transformers.SentenceTransformer` on first call, returns the cached instance on all subsequent calls. `get_nli_model()`: loads `cross-encoder/nli-deberta-v3-large` via `sentence_transformers.CrossEncoder(num_labels=3)` on first call. Both use module-level globals (`_embed_model`, `_nli_model`) to ensure the models are loaded exactly once per process across all calling modules. |

---

## Scripts

| File | Description |
|---|---|
| `scripts/run_study.py` | **Primary orchestration script** for a full multi-variant prompt study. Runs all four prompt variants sequentially (empty → pure → normal → eliciting), then runs extraction, profile computation, sensitivity analysis, figures, and writes a markdown report — all in a single command. Arguments: `--questions`, `--runs`, `--output`, `--skip-collect`, `--skip-extract`. Reads `VLLM_API_KEY`, `VLLM_ENDPOINT`, `VLLM_MODEL` from environment. Output: `results/study_{timestamp}/` with subdirectories `traces/`, `graphs/`, `profiles/`, `figures/`, and `report.md`. |
| `scripts/run_experiment.py` | **Modular experiment runner** (one prompt variant at a time). Exposes three independent phases via flags: `--collect`, `--extract`, `--compute` (or `--all` for the full pipeline). Supports a `--prompt` argument to select a single variant. Produces output in `data/traces/`, `data/graphs/`, `data/profiles/`. Used for iterative development and single-variant reruns. |
| `scripts/extract_graphs.py` | **Batch graph extraction** script. Reads all `traces_*.jsonl` files from an input directory, calls `segment()` → `classify_nodes()` → `build_graph()` for each trace, and writes one `{trace_id}.json` to the output directory. Skips already-extracted traces (idempotent). Supports `--no-nli` flag to skip Pass 5 for faster iteration. Prints per-file and totals summary with success/fail/skip counts. |
| `scripts/compute_profiles.py` | **Batch profile computation** script. Loads all graph JSONs from a directory, calls `compute_profile()` on each, then aggregates by `thinking_effort` level (reads effort from trace files). Writes `effort_comparison.json`. Older script — predates the prompt-variant study design; groups by effort level rather than prompt variant. |
| `scripts/generate_report.py` | **Single-profile report generator**. Reads `data/profiles/effort_comparison.json`, generates a radar chart (min-max normalised against hardcoded `_BOUNDS`), and writes a Markdown report to `docs/benchmark_report.md`. Produces a self-contained single-profile summary — used for early development inspection. |
| `scripts/report_to_pdf.py` | **Markdown-to-PDF converter** using WeasyPrint. Reads a study report markdown file, embeds all referenced images as base64 data URIs, converts to HTML via the `markdown` library (with tables, fenced_code, footnotes, toc, sane_lists extensions), applies a full academic CSS stylesheet (Source Serif 4 font, A4 page size, page numbers, dark-header tables), and renders to PDF. Configured for `results/study_v3_20260424_2036/report_v2.md` but reusable for any report. |
| `scripts/metric_analysis.py` | **Metric-space analysis** script. Loads all per-trace profiles from a study directory, builds a (90, 22) metric matrix, and performs: (1) per-metric statistics (mean/std/min/max/zero%), (2) full Spearman ρ correlation matrix with high-pair flagging (|ρ| > 0.70), (3) Ward hierarchical clustering on (1 − |ρ|) distance, (4) variance and Cohen's d effect-size per metric (pure vs. eliciting), (5) ANOVA F-statistic across variants, (6) composite ranking and KEEP/REVIEW/MERGE/DROP verdict for each metric. Standalone script — not imported by any other module. |
| `scripts/make_pipeline_figure.py` | **Publication-quality pipeline diagram** generator. Draws the ThinkBench v3 pipeline as a vertical sequence of colour-coded blocks (Pass 1 purple, Pass 2 blue, Layer 3 burnt orange, Layer 4 crimson, Pass 5 dashed gray, Classifier forest green, Graph navy, Metrics deep purple) with region groupings (Segmentation / Edge Assignment / Graph & Metrics), arrows with data-flow labels, and a legend. Saves to `results/pipeline_v3.png` at 220 DPI. Uses `matplotlib.patches.FancyBboxPatch`. |
| `scripts/thinkbench_full.py` | Earlier standalone full-pipeline script (pre-`run_study.py`). Runs collection + extraction + profiling in a single file. Now superseded by `run_study.py` but retained for reference. |
| `scripts/run_k3_experiment.py` | Early single-run experiment script using k=3 traces. Predates the multi-variant study design. |
| `scripts/run_k3_high.py` | Variant of the k=3 runner with `thinking_effort=HIGH` parameter. |
| `scripts/run_k3_10q.py` | k=3 experiment with 10 questions (full question set). |

---

## Documentation

| File | Description |
|---|---|
| `docs/benchmark_report.md` | Auto-generated single-model benchmark report (produced by `scripts/generate_report.py`). Shows all 22 metric values in tabular form plus a radar chart. Reflects the early effort-level comparison design. |
| `docs/preliminary_report.md` | Preliminary analysis report from early experimental runs. Contains early observations on metric behaviour and pipeline calibration. |

---

## Package Build Artifacts — `src/thinkbench.egg-info/`

Auto-generated by `pip install -e .`. Do not edit manually.

| File | Description |
|---|---|
| `src/thinkbench.egg-info/SOURCES.txt` | Lists all source files included in the installed package. |
| `src/thinkbench.egg-info/dependency_links.txt` | Dependency link overrides (empty for this project). |
| `src/thinkbench.egg-info/requires.txt` | Serialised dependency list mirroring `pyproject.toml`. |
| `src/thinkbench.egg-info/top_level.txt` | Contains `thinkbench` — the single top-level package. |

---

## Key Data Flow

```
data/questions/*.jsonl
        │
        ▼  scripts/run_study.py  (or run_experiment.py)
collect/collector.py  →  results/{study}/traces/traces_{model}_{variant}.jsonl
        │
        ▼  extract_graphs.py  (or run_study.py Phase 2)
extract/segmenter.py  →  TUs + Edges + Embeddings
extract/classifier.py →  node_type per TU
extract/linker.py     →  ThoughtGraph
        │
        ▼  results/{study}/graphs/{trace_id}.json
        │
        ▼  compute_profiles.py  (or run_study.py Phase 3)
metrics/profile.py  →  CognitiveProfile × N traces
        │
        ▼  results/{study}/profiles/profiles_agg.json
        │
        ▼  analysis/compare.py  (sensitivity analysis)
        ▼  analysis/figures.py  (all 14 figures)
        │
        ▼  results/{study}/figures_v2/  +  supplement/
        ▼  results/{study}/report_v4.md  +  report_v4.pdf
```
