#!/usr/bin/env python
"""ThinkBench experiment runner (v2 — non-generative extraction).

Usage:
    python scripts/run_experiment.py --questions data/questions/ethical_dilemmas.jsonl --runs 3 --all
    python scripts/run_experiment.py --questions data/questions/ethical_dilemmas.jsonl --runs 3 --prompt eliciting --all
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def collect_traces(args):
    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    with open(args.questions) as f:
        questions = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(questions)} questions")

    client = LLMClient(api_key=api_key, endpoint=endpoint, model=model, prompt_variant=args.prompt)
    collector = TraceCollector(client=client, output_dir=Path(args.output_dir))
    traces = await collector.collect_batch(questions=questions, model=model, runs=args.runs, questions_per_batch=5)
    logger.info(f"Collected {len(traces)} traces")
    return traces


def extract_graphs(args):
    """Non-generative graph extraction."""
    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = Path(args.output_dir)
    trace_files = list(traces_dir.glob("traces_*.jsonl"))

    if not trace_files:
        logger.error(f"No trace files in {traces_dir}")
        return

    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    for trace_file in trace_files:
        with open(trace_file) as f:
            traces = [json.loads(line) for line in f]
        logger.info(f"Processing {len(traces)} traces from {trace_file.name}")

        for trace in traces:
            stats["total"] += 1
            trace_id = trace["trace_id"]
            output_path = graphs_dir / f"{trace_id}.json"

            if output_path.exists():
                stats["skipped"] += 1
                continue

            try:
                tus, edges, embeddings = segment(trace["raw_cot"])
                if not tus:
                    logger.warning(f"No TUs for {trace_id}")
                    stats["failed"] += 1
                    continue

                classify_nodes(tus, edges, embeddings)
                graph = build_graph(trace, tus, edges)

                with open(output_path, "w") as f:
                    json.dump(graph.model_dump(), f, indent=2)

                stats["success"] += 1
                logger.info(f"  OK: {trace_id[:8]} ({len(graph.nodes)} nodes)")

            except Exception as e:
                logger.error(f"  FAILED: {trace_id[:8]}: {e}")
                stats["failed"] += 1

    logger.info(f"Extraction: {stats}")
    return stats


def compute_profiles_fn(args):
    graphs_dir = Path(args.graphs_dir)
    graph_files = list(graphs_dir.glob("*.json"))
    if not graph_files:
        logger.error(f"No graphs in {graphs_dir}")
        return

    traces_dir = Path(args.output_dir)
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
            profiles.append(compute_profile(graph, model=graph.model, domain=graph.domain))
        except Exception as e:
            logger.error(f"  Error {gf.name}: {e}")

    aggregated = aggregate_profiles(profiles, trace_model, trace_prompt)
    for row in aggregated:
        logger.info(f"Aggregated {row['num_traces']} traces for {row['model']}")

    output_path = Path(args.profiles_dir) / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Saved results to {output_path}")
    return aggregated


async def main():
    parser = argparse.ArgumentParser(description="ThinkBench experiment runner (v2)")
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="data/traces/")
    parser.add_argument("--graphs-dir", type=str, default="data/graphs/")
    parser.add_argument("--profiles-dir", type=str, default="data/profiles/")
    parser.add_argument("--prompt", type=str, default="normal", choices=PROMPT_VARIANTS)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.collect:
        logger.info("PHASE 1: Collecting traces")
        await collect_traces(args)

    if args.all or args.extract:
        logger.info("PHASE 2: Extracting graphs (non-generative)")
        extract_graphs(args)

    if args.all or args.compute:
        logger.info("PHASE 3: Computing profiles")
        compute_profiles_fn(args)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    asyncio.run(main())
