"""Batch extraction script — extract thought graphs from all traces (v2, non-generative)."""

import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thinkbench.extract.segmenter import segment
from thinkbench.extract.classifier import classify_nodes
from thinkbench.extract.linker import build_graph


def extract_single_trace(trace: dict, output_dir: Path, stats: dict) -> bool:
    trace_id = trace["trace_id"]
    output_path = output_dir / f"{trace_id}.json"

    if output_path.exists():
        print(f"  Skipping {trace_id} (already exists)")
        stats["skipped"] += 1
        stats["total"] += 1
        return True

    try:
        tus, edges = segment(trace["raw_cot"])
        if not tus:
            print(f"  Warning: No TUs for {trace_id}")
            stats["failed"] += 1
            stats["total"] += 1
            return False

        classify_nodes(tus)
        graph = build_graph(trace, tus, edges)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(graph.model_dump(), f, indent=2)

        print(f"  Extracted {trace_id}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        stats["success"] += 1
        stats["total"] += 1
        return True

    except Exception as e:
        print(f"  Failed {trace_id}: {e}")
        stats["failed"] += 1
        stats["total"] += 1
        return False


def extract_all(traces_path: Path, output_dir: Path, limit: Optional[int] = None) -> dict:
    stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    with open(traces_path) as f:
        traces = [json.loads(line) for line in f]
    if limit:
        traces = traces[:limit]

    print(f"Processing {len(traces)} traces from {traces_path.name}")
    for trace in traces:
        extract_single_trace(trace, output_dir, stats)

    print(f"\n=== Summary for {traces_path.name} ===")
    print(f"  Total: {stats['total']} | OK: {stats['success']} | "
          f"Fail: {stats['failed']} | Skip: {stats['skipped']}")
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch extract thought graphs (v2)")
    parser.add_argument("--input", type=str, default="data/traces")
    parser.add_argument("--output", type=str, default="data/graphs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-nli", action="store_true",
                        help="Skip NLI Pass 3 (faster, regex+TextTiling only)")
    args = parser.parse_args()

    if args.no_nli:
        os.environ["THINKBENCH_NO_NLI"] = "1"

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    trace_files = sorted(input_dir.glob("traces_*.jsonl"))

    if not trace_files:
        print(f"No trace files found in {input_dir}")
        return

    totals = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    for traces_path in trace_files:
        print(f"\n=== Processing {traces_path.name} ===")
        result = extract_all(traces_path, output_dir, args.limit)
        for k in totals:
            totals[k] += result[k]

    print(f"\n{'=' * 50}")
    print(f"TOTAL: {totals['total']} | OK: {totals['success']} | "
          f"FAIL: {totals['failed']} | SKIP: {totals['skipped']}")


if __name__ == "__main__":
    main()
