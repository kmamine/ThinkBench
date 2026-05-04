#!/usr/bin/env python
"""ThinkBench automated multi-prompt study runner.

Runs all three prompt variants (pure / normal / eliciting), extracts graphs,
computes profiles, performs sensitivity analysis, and generates all paper figures
plus a markdown report — all in a single command.

Usage:
    python scripts/run_study.py --questions data/questions/ethical_dilemmas.jsonl --runs 3
    python scripts/run_study.py --questions data/questions/ethical_dilemmas.jsonl --runs 5 --output results/my_study
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thinkbench.collect.models import LLMClient, PROMPT_VARIANTS
from thinkbench.collect.collector import TraceCollector
from thinkbench.extract.segmenter import segment
from thinkbench.extract.classifier import classify_nodes
from thinkbench.extract.linker import build_graph
from thinkbench.extract.schemas import ThoughtGraph
from thinkbench.metrics import compute_profile, aggregate_profiles
from thinkbench.analysis import (
    compute_deltas, prompt_sensitivity, classify_metrics,
    radar_comparison, sensitivity_bar, metric_heatmap, category_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

async def _collect_variant(questions: list[dict], variant: str, runs: int,
                            traces_dir: Path, model: str,
                            api_key: str, endpoint: str) -> int:
    client = LLMClient(api_key=api_key, endpoint=endpoint, model=model,
                       prompt_variant=variant)
    collector = TraceCollector(client=client, output_dir=traces_dir)
    traces = await collector.collect_batch(
        questions=questions, model=model, runs=runs, questions_per_batch=5
    )
    logger.info(f"  [{variant}] collected {len(traces)} traces")
    return len(traces)


async def collect_all_variants(questions: list[dict], runs: int,
                                traces_dir: Path, model: str,
                                api_key: str, endpoint: str) -> None:
    for variant in PROMPT_VARIANTS:
        logger.info(f"Collecting traces — variant: {variant}")
        await _collect_variant(questions, variant, runs, traces_dir, model,
                               api_key, endpoint)


def extract_graphs(traces_dir: Path, graphs_dir: Path) -> dict:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    trace_files = list(traces_dir.glob("traces_*.jsonl"))
    if not trace_files:
        logger.error(f"No trace files in {traces_dir}")
        return {}

    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    for trace_file in trace_files:
        with open(trace_file) as f:
            traces = [json.loads(line) for line in f]
        logger.info(f"Extracting {len(traces)} traces from {trace_file.name}")

        for trace in traces:
            stats["total"] += 1
            trace_id = trace["trace_id"]
            output_path = graphs_dir / f"{trace_id}.json"
            if output_path.exists():
                stats["skipped"] += 1
                continue
            try:
                tus, edges, embeddings = segment(trace["raw_cot"], use_nli=True)
                if not tus:
                    stats["failed"] += 1
                    continue
                classify_nodes(tus, edges, embeddings)
                graph = build_graph(trace, tus, edges)
                with open(output_path, "w") as f:
                    json.dump(graph.model_dump(), f, indent=2)
                stats["success"] += 1
            except Exception as e:
                logger.error(f"  FAILED {trace_id[:8]}: {e}")
                stats["failed"] += 1

    logger.info(f"Extraction stats: {stats}")
    return stats


def compute_profiles(traces_dir: Path, graphs_dir: Path,
                     profiles_dir: Path) -> list[dict]:
    graph_files = list(graphs_dir.glob("*.json"))
    if not graph_files:
        logger.error(f"No graphs in {graphs_dir}")
        return []

    trace_model: dict[str, str] = {}
    trace_prompt: dict[str, str] = {}
    for tf in traces_dir.glob("traces_*.jsonl"):
        with open(tf) as f:
            for line in f:
                t = json.loads(line)
                trace_model[t["trace_id"]] = t.get("model", "unknown")
                trace_prompt[t["trace_id"]] = t.get("prompt_variant", "normal")

    profiles = []
    for gf in graph_files:
        try:
            with open(gf) as f:
                graph = ThoughtGraph(**json.load(f))
            profiles.append(compute_profile(graph, model=graph.model,
                                            domain=graph.domain))
        except Exception as e:
            logger.error(f"  Error {gf.name}: {e}")

    aggregated = aggregate_profiles(profiles, trace_model, trace_prompt)
    for row in aggregated:
        logger.info(
            f"  [{row['prompt_variant']}] {row['model']} — {row['num_traces']} traces"
        )

    profiles_dir.mkdir(parents=True, exist_ok=True)
    out = profiles_dir / "profiles_agg.json"
    with open(out, "w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved aggregated profiles → {out}")
    return aggregated


def run_analysis(profiles_agg: list[dict], profiles_dir: Path) -> tuple[list, list, dict]:
    deltas = compute_deltas(profiles_agg)
    sensitivity = prompt_sensitivity(profiles_agg)
    classification = classify_metrics(sensitivity)

    with open(profiles_dir / "deltas.json", "w") as f:
        json.dump(deltas, f, indent=2)
    with open(profiles_dir / "sensitivity.json", "w") as f:
        json.dump(sensitivity, f, indent=2)
    with open(profiles_dir / "metric_classification.json", "w") as f:
        json.dump(classification, f, indent=2)

    logger.info(f"Prompt-sensitive metrics: {classification['prompt_sensitive']}")
    logger.info(f"Prompt-invariant metrics:  {classification['prompt_invariant']}")
    return deltas, sensitivity, classification


def generate_figures(profiles_agg: list[dict], sensitivity: list[dict],
                     figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    radar_comparison(profiles_agg, figures_dir / "radar_comparison.png")
    logger.info("  radar_comparison.png")

    sensitivity_bar(sensitivity, figures_dir / "sensitivity_ranking.png")
    logger.info("  sensitivity_ranking.png")

    metric_heatmap(profiles_agg, figures_dir / "metric_heatmap.png")
    logger.info("  metric_heatmap.png")

    category_comparison(profiles_agg, figures_dir / "category_comparison.png")
    logger.info("  category_comparison.png")


def write_report(profiles_agg: list[dict], sensitivity: list[dict],
                 classification: dict, stats: dict,
                 output_dir: Path, questions_path: str, runs: int) -> None:
    model = profiles_agg[0].get("model", "unknown") if profiles_agg else "unknown"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build sensitivity table rows
    sens_rows = "\n".join(
        f"| {r['metric']} | {r['category']} | {r['delta_eliciting']:+.4f} "
        f"| {r['delta_normal']:+.4f} | {r['abs_sensitivity']:.4f} |"
        for r in sensitivity[:10]
    )

    # Build variant summary table
    by_variant = {p["prompt_variant"]: p for p in profiles_agg}
    variant_rows = ""
    for v in ["empty", "pure", "normal", "eliciting"]:
        p = by_variant.get(v)
        if p:
            variant_rows += (
                f"| {v} | {p['num_traces']} "
                f"| {p.get('branching_factor', 0):.3f} "
                f"| {p.get('unique_perspective_count', 0):.2f} "
                f"| {p.get('reasoning_density', 0):.3f} "
                f"| {p.get('self_reflection_rate', 0):.3f} |\n"
            )

    sensitive_list = ", ".join(f"`{m}`" for m in classification["prompt_sensitive"])
    invariant_list = ", ".join(f"`{m}`" for m in classification["prompt_invariant"])

    report = f"""# ThinkBench Prompt Variant Study Report

**Generated**: {timestamp}
**Model**: {model}
**Questions**: {questions_path}
**Runs per variant**: {runs}
**Extraction**: {stats.get('success', '?')} graphs extracted ({stats.get('failed', '?')} failed, {stats.get('skipped', '?')} skipped)

---

## Experimental Design

Four prompt variants were tested to separate **intrinsic model capability** from
**prompted reasoning behavior**:

| Variant | Description |
|---|---|
| `empty` | No system prompt (completely absent from the request) |
| `pure` | Bare system prompt: "You are a helpful assistant." |
| `normal` | Minimal guidance: "Think carefully before answering." |
| `eliciting` | Full framing: explicit instructions to spread wide, reframe, go deep, challenge, connect, reflect, converge |

---

## Variant Summary

| Variant | N traces | Branching Factor | Perspectives | Reasoning Density | Self-Reflection |
|---|---|---|---|---|---|
{variant_rows}

---

## Prompt Sensitivity Analysis

Metrics ranked by |Δ(eliciting − pure)|. Positive = increases with more guidance.

| Metric | Category | Δ Eliciting | Δ Normal | Abs Sensitivity |
|---|---|---|---|---|
{sens_rows}

---

## Metric Classification

**Prompt-sensitive** (top 6 by Δ — likely reflect prompted behavior):
{sensitive_list}

**Prompt-invariant** (lower Δ — likely reflect intrinsic capability):
{invariant_list}

---

## Figures

| Figure | Description |
|---|---|
| `figures/radar_comparison.png` | Overlaid radar chart — all 20 metrics, all 3 variants |
| `figures/sensitivity_ranking.png` | Horizontal bar chart ranking metrics by Δ (dual panel) |
| `figures/metric_heatmap.png` | Heatmap: metrics × variants, normalized [0–1] |
| `figures/category_comparison.png` | Grouped bar: mean category score per variant |

---

## Interpretation Notes

- Metrics with **large positive Δ** under `eliciting` but not `normal` are
  specifically responsive to explicit framing instructions — these measure
  *prompted* rather than *intrinsic* behavior.
- Metrics that remain **stable across all three variants** reflect the model's
  baseline cognitive style regardless of how it is prompted.
- The `pure → normal` gap reveals the effect of a simple "think carefully"
  nudge; the `normal → eliciting` gap reveals the effect of structured
  reasoning framing.
"""

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="ThinkBench automated prompt variant study")
    parser.add_argument("--questions", type=str, required=True,
                        help="Path to questions JSONL file")
    parser.add_argument("--runs", type=int, default=3,
                        help="Traces per question per variant")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: results/study_YYYYMMDD_HHMM)")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip collection, use existing traces in output dir")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip graph extraction, use existing graphs")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output) if args.output else Path(f"results/study_{timestamp}")
    traces_dir = output_dir / "traces"
    graphs_dir = output_dir / "graphs"
    profiles_dir = output_dir / "profiles"
    figures_dir = output_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Study output directory: {output_dir}")

    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    with open(args.questions) as f:
        questions = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(questions)} questions from {args.questions}")

    # Phase 1: Collection
    if not args.skip_collect:
        logger.info("=" * 60)
        logger.info("PHASE 1: Collecting traces for all prompt variants")
        logger.info("=" * 60)
        await collect_all_variants(questions, args.runs, traces_dir, model,
                                   api_key, endpoint)
    else:
        logger.info("Skipping collection (--skip-collect)")

    # Phase 2: Graph extraction
    if not args.skip_extract:
        logger.info("=" * 60)
        logger.info("PHASE 2: Extracting thought graphs")
        logger.info("=" * 60)
        extraction_stats = extract_graphs(traces_dir, graphs_dir)
    else:
        logger.info("Skipping extraction (--skip-extract)")
        extraction_stats = {}

    # Phase 3: Profile computation
    logger.info("=" * 60)
    logger.info("PHASE 3: Computing cognitive profiles")
    logger.info("=" * 60)
    profiles_agg = compute_profiles(traces_dir, graphs_dir, profiles_dir)
    if not profiles_agg:
        logger.error("No profiles computed — aborting analysis.")
        return

    # Phase 4: Sensitivity analysis
    logger.info("=" * 60)
    logger.info("PHASE 4: Prompt sensitivity analysis")
    logger.info("=" * 60)
    deltas, sensitivity, classification = run_analysis(profiles_agg, profiles_dir)

    # Phase 5: Figures
    logger.info("=" * 60)
    logger.info("PHASE 5: Generating figures")
    logger.info("=" * 60)
    generate_figures(profiles_agg, sensitivity, figures_dir)

    # Phase 6: Report
    logger.info("=" * 60)
    logger.info("PHASE 6: Writing study report")
    logger.info("=" * 60)
    write_report(profiles_agg, sensitivity, classification, extraction_stats,
                 output_dir, args.questions, args.runs)

    logger.info("=" * 60)
    logger.info(f"Study complete. Results in: {output_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
