#!/usr/bin/env python3
"""ThinkBench Full Pipeline (v2)

Usage:
    python scripts/thinkbench_full.py --questions data/questions/ethical_dilemmas.jsonl --runs 3

Pipeline:
  1. Collect traces from LLM (collection only uses the LLM endpoint)
  2. Extract cognitive graphs (non-generative: regex + TextTiling + NLI)
  3. Compute profiles (22 metrics)
  4. Generate report & visualizations
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from thinkbench.collect.models import LLMClient
from thinkbench.collect.collector import TraceCollector
from thinkbench.extract.segmenter import segment
from thinkbench.extract.classifier import classify_nodes
from thinkbench.extract.linker import build_graph
from thinkbench.extract.schemas import ThoughtGraph
from thinkbench.metrics import compute_profile, aggregate_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_KEY = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# v2: empirical bounds derived from benchmark results (placeholder — will be
# replaced with actual min/max after the full benchmark run)
_NORM_BOUNDS: dict[str, tuple[float, float]] = {
    "branching_factor": (0, 0.5),
    "unique_perspective_count": (0, 10),
    "domain_spread": (0, 10),
    "first_idea_diversity": (0, 1),
    "max_elaboration_chain": (0, 15),
    "mean_branch_depth": (0, 10),
    "specificity_gradient": (-0.1, 0.1),
    "reasoning_density": (0, 1),
    "exploration_exploitation_ratio": (0, 5),
    "backtracking_rate": (0, 1),
    "cross_branch_connectivity": (0, 1),
    "convergence_index": (0, 1),
    "graph_density": (0, 0.5),
    "revision_depth": (0, 20),
    "self_reflection_rate": (0, 0.3),
    "critique_to_hypothesis_ratio": (0, 1),
    "hedging_density": (0, 1),
    "perspective_taking": (0, 0.3),
    "token_per_idea": (0, 2000),
    "redundancy_ratio": (0, 0.5),
    "avg_tokens": (0, 20000),
    "avg_tus": (0, 50),
}

_RADAR_METRICS = [
    "branching_factor", "unique_perspective_count", "domain_spread", "first_idea_diversity",
    "max_elaboration_chain", "mean_branch_depth", "specificity_gradient", "reasoning_density",
    "exploration_exploitation_ratio", "backtracking_rate", "cross_branch_connectivity",
    "convergence_index", "graph_density", "revision_depth",
    "self_reflection_rate", "critique_to_hypothesis_ratio", "hedging_density", "perspective_taking",
    "token_per_idea", "redundancy_ratio",
]


def normalize_value(value: float, metric_name: str) -> float:
    lo, hi = _NORM_BOUNDS.get(metric_name, (0, 1))
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


async def collect_traces(questions, runs, output_dir):
    logger.info("PHASE 1: Collecting traces")
    client = LLMClient(api_key=API_KEY, endpoint=ENDPOINT, model=MODEL)
    collector = TraceCollector(client=client, output_dir=output_dir)
    traces = await collector.collect_batch(
        questions=questions, model=MODEL, runs=runs, questions_per_batch=5
    )
    logger.info(f"Collected {len(traces)} traces")
    return traces


def extract_graphs(traces_dir: Path, graphs_dir: Path) -> dict:
    """Phase 2: Non-generative graph extraction."""
    logger.info("PHASE 2: Extracting graphs (non-generative)")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    trace_files = list(traces_dir.glob("traces_*.jsonl"))
    if not trace_files:
        logger.error(f"No trace files found in {traces_dir}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    for trace_file in trace_files:
        with open(trace_file) as f:
            traces = [json.loads(line) for line in f]

        for trace in traces:
            stats["total"] += 1
            trace_id = trace["trace_id"]
            output_path = graphs_dir / f"{trace_id}.json"

            if output_path.exists():
                stats["skipped"] += 1
                continue

            try:
                tus, edges = segment(trace["raw_cot"])
                if not tus:
                    logger.warning(f"No TUs for {trace_id}")
                    stats["failed"] += 1
                    continue

                classify_nodes(tus)
                graph = build_graph(trace, tus, edges)

                with open(output_path, "w") as f:
                    json.dump(graph.model_dump(), f, indent=2)

                stats["success"] += 1
                logger.info(f"  OK: {trace_id[:8]} ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")

            except Exception as e:
                logger.error(f"  FAILED: {trace_id[:8]}: {e}")
                stats["failed"] += 1

    logger.info(f"Extraction: {stats}")
    return stats


def compute_profiles(graphs_dir: Path, traces_dir: Path) -> list[dict]:
    logger.info("PHASE 3: Computing profiles")

    graph_files = list(graphs_dir.glob("*.json"))
    if not graph_files:
        logger.error(f"No graphs in {graphs_dir}")
        return []

    trace_model: dict[str, str] = {}
    for tf in traces_dir.glob("traces_*.jsonl"):
        with open(tf) as f:
            for line in f:
                t = json.loads(line)
                trace_model[t["trace_id"]] = t.get("model", "unknown")

    profiles = []
    for gf in graph_files:
        try:
            with open(gf) as f:
                graph = ThoughtGraph(**json.load(f))
            profiles.append(compute_profile(graph, model=graph.model, domain=graph.domain))
        except Exception as e:
            logger.error(f"  Error {gf.name}: {e}")

    aggregated = aggregate_profiles(profiles, trace_model)
    logger.info(f"Aggregated {len(profiles)} profiles into {len(aggregated)} groups")
    return aggregated


def generate_report(profiles: list[dict], output_dir: Path, args) -> None:
    logger.info("PHASE 4: Generating report")
    if not profiles:
        logger.warning("No profiles — skipping report")
        return

    profile = profiles[0]

    # Radar chart (20 core metrics, exclude summary avg_tokens / avg_tus)
    values = [normalize_value(profile.get(m, 0.0), m) for m in _RADAR_METRICS]
    angles = np.linspace(0, 2 * np.pi, len(_RADAR_METRICS), endpoint=False).tolist()
    values_plot = np.concatenate([values, [values[0]]])
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#2E86AB")
    ax.fill(angles_plot, values_plot, alpha=0.25, color="#2E86AB")
    ax.set_xticks(angles)
    ax.set_xticklabels(_RADAR_METRICS, size=7)
    ax.set_title(
        f"Cognitive Profile: {profile.get('model', MODEL)}\n({profile['num_traces']} traces)",
        size=14, fontweight="bold",
    )
    plt.tight_layout()

    radar_path = output_dir / f"benchmark_radar_{TIMESTAMP}.png"
    plt.savefig(radar_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved radar chart: {radar_path}")

    # Markdown report
    report = f"""# ThinkBench Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Model:** {profile.get('model', MODEL)}
**Configuration:** K={args.runs}, N={profile['num_traces']} traces

---

## Breadth
| Metric | Value |
|--------|-------|
| branching_factor | {profile.get('branching_factor', 0):.4f} |
| unique_perspective_count | {profile.get('unique_perspective_count', 0):.2f} |
| domain_spread | {profile.get('domain_spread', 0):.2f} |
| first_idea_diversity | {profile.get('first_idea_diversity', 0):.4f} |

## Depth
| Metric | Value |
|--------|-------|
| max_elaboration_chain | {profile.get('max_elaboration_chain', 0):.2f} |
| mean_branch_depth | {profile.get('mean_branch_depth', 0):.2f} |
| specificity_gradient | {profile.get('specificity_gradient', 0):.4f} |
| reasoning_density | {profile.get('reasoning_density', 0):.4f} |

## Structure
| Metric | Value |
|--------|-------|
| exploration_exploitation_ratio | {profile.get('exploration_exploitation_ratio', 0):.4f} |
| backtracking_rate | {profile.get('backtracking_rate', 0):.4f} |
| cross_branch_connectivity | {profile.get('cross_branch_connectivity', 0):.4f} |
| convergence_index | {profile.get('convergence_index', 0):.4f} |
| graph_density | {profile.get('graph_density', 0):.4f} |
| revision_depth | {profile.get('revision_depth', 0):.4f} |

## Metacognitive
| Metric | Value |
|--------|-------|
| self_reflection_rate | {profile.get('self_reflection_rate', 0):.4f} |
| critique_to_hypothesis_ratio | {profile.get('critique_to_hypothesis_ratio', 0):.4f} |
| hedging_density | {profile.get('hedging_density', 0):.4f} |
| perspective_taking | {profile.get('perspective_taking', 0):.4f} |

## Efficiency
| Metric | Value |
|--------|-------|
| token_per_idea | {profile.get('token_per_idea', 0):.2f} |
| redundancy_ratio | {profile.get('redundancy_ratio', 0):.4f} |

## Summary
| Metric | Value |
|--------|-------|
| avg_tokens | {profile.get('avg_tokens', 0):.1f} |
| avg_tus | {profile.get('avg_tus', 0):.1f} |

---

![Cognitive Profile Radar Chart](benchmark_radar_{TIMESTAMP}.png)

*Generated by ThinkBench v2*
"""
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved report: {report_path}")

    with open(output_dir / f"benchmark_{TIMESTAMP}.json", "w") as f:
        json.dump(profiles, f, indent=2)


async def run_pipeline(questions_file: str, runs: int, output_dir: str, args) -> None:
    with open(questions_file) as f:
        questions = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(questions)} questions")

    traces_dir = Path(output_dir) / "traces"
    graphs_dir = Path(output_dir) / "graphs"
    profiles_dir = Path(output_dir) / "profiles"
    docs_dir = Path(output_dir).parent / "docs"

    for d in [traces_dir, graphs_dir, profiles_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    await collect_traces(questions, runs, traces_dir)
    extract_graphs(traces_dir, graphs_dir)
    profiles = compute_profiles(graphs_dir, traces_dir)

    profile_file = profiles_dir / f"results_{TIMESTAMP}.json"
    with open(profile_file, "w") as f:
        json.dump(profiles, f, indent=2)

    generate_report(profiles, docs_dir, args)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ThinkBench Full Pipeline v2")
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="data")
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.questions, args.runs, args.output, args))
