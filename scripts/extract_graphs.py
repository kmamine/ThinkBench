"""Batch extraction script - extract thought graphs from all traces."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thinkbench.collect.models import LLMClient
from thinkbench.extract import Segmenter, Classifier, Linker, build_graph


async def extract_single_trace(
    client: LLMClient,
    segmenter: Segmenter,
    classifier: Classifier,
    linker: Linker,
    trace: dict,
    output_dir: Path,
) -> bool:
    """Extract graph from a single trace."""
    trace_id = trace["trace_id"]
    output_path = output_dir / f"{trace_id}.json"

    if output_path.exists():
        print(f"  Skipping {trace_id} (already exists)")
        return True

    try:
        tus = await segmenter.segment(trace["raw_cot"])
        if not tus:
            print(f"  Warning: No TUs segmented for {trace_id}")
            return False

        classified_tus = await classifier.classify(tus)
        edges = await linker.link(classified_tus)

        graph = build_graph(trace, classified_tus, edges)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(graph.model_dump(), f, indent=2)

        print(
            f"  Extracted {trace_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges"
        )
        return True

    except Exception as e:
        print(f"  Error extracting {trace_id}: {e}")
        return False


async def extract_all(
    traces_path: Path,
    output_dir: Path,
    api_key: str,
    endpoint: str,
    model: str,
    limit: Optional[int] = None,
):
    """Extract graphs from all traces in a file."""
    client = LLMClient(api_key=api_key, endpoint=endpoint, model=model)
    segmenter = Segmenter(client)
    classifier = Classifier(client)
    linker = Linker(client)

    with open(traces_path) as f:
        traces = [json.loads(line) for line in f]

    if limit:
        traces = traces[:limit]

    print(f"Processing {len(traces)} traces from {traces_path.name}")

    success = 0
    for trace in traces:
        if await extract_single_trace(
            client, segmenter, classifier, linker, trace, output_dir
        ):
            success += 1

    print(f"Extracted {success}/{len(traces)} graphs to {output_dir}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch extract thought graphs")
    parser.add_argument(
        "--input", type=str, default="data/traces", help="Traces directory"
    )
    parser.add_argument(
        "--output", type=str, default="data/graphs", help="Output directory"
    )
    parser.add_argument("--api-key", type=str, default=None, help="vLLM API key")
    parser.add_argument("--endpoint", type=str, default=None, help="vLLM endpoint")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--limit", type=int, default=None, help="Limit traces per file")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = args.endpoint or os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = args.model or os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    trace_files = list(input_dir.glob("traces_*.jsonl"))
    if not trace_files:
        print(f"No trace files found in {input_dir}")
        return

    for traces_path in trace_files:
        print(f"\n=== Processing {traces_path.name} ===")
        await extract_all(traces_path, output_dir, api_key, endpoint, model, args.limit)


if __name__ == "__main__":
    asyncio.run(main())
