"""Batch compute cognitive profiles from extracted graphs."""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thinkbench.extract.schemas import ThoughtGraph
from thinkbench.metrics import compute_profile


def compute_all_profiles(graphs_dir: Path) -> dict:
    """Compute profiles for all graphs in directory."""
    graph_files = list(graphs_dir.glob("*.json"))

    if not graph_files:
        print(f"No graphs found in {graphs_dir}")
        return {}

    profiles = []
    for graph_file in graph_files:
        try:
            with open(graph_file) as f:
                graph_data = json.load(f)
            graph = ThoughtGraph(**graph_data)
            profile = compute_profile(graph, model=graph.model, domain=graph.domain)
            profile.trace_id = graph.trace_id  # Add trace_id for matching
            profiles.append(profile)
            print(f"  Computed profile for {graph.trace_id}")
        except Exception as e:
            print(f"  Error computing profile for {graph_file.name}: {e}")

    print(f"Computed {len(profiles)} profiles")
    return profiles


def aggregate_by_effort(profiles: list, traces_dir: Path, graphs_dir: Path) -> list:
    """Aggregate profiles by thinking effort level."""
    effort_groups = defaultdict(list)
    trace_tokens = {}
    trace_effort = {}

    trace_files = list(traces_dir.glob("traces_*.jsonl"))

    for tf in trace_files:
        with open(tf) as f:
            for line in f:
                t = json.loads(line)
                trace_id = t["trace_id"]
                trace_tokens[trace_id] = t.get("token_count", 0)
                effort = t.get("thinking_effort", "unknown")
                if "HIGH" in effort:
                    trace_effort[trace_id] = "high"
                elif "MAX" in effort:
                    trace_effort[trace_id] = "max"
                elif "MEDIUM" in effort:
                    trace_effort[trace_id] = "medium"
                elif "LOW" in effort:
                    trace_effort[trace_id] = "low"
                else:
                    trace_effort[trace_id] = "unknown"

    for profile in profiles:
        trace_id = profile.trace_id
        if trace_id and trace_id in trace_effort:
            effort_groups[trace_effort[trace_id]].append(
                (profile, trace_tokens.get(trace_id, 0))
            )
        else:
            print(f"  Warning: No effort mapping for trace {trace_id}")

    aggregated = []
    for effort, items in sorted(effort_groups.items()):
        if not items:
            continue

        n = len(items)
        metrics = defaultdict(list)
        total_tokens = 0

        for profile, tokens in items:
            total_tokens += tokens
            for field, value in profile.model_dump().items():
                if field in ("model", "domain", "trace_id"):
                    continue
                metrics[field].append(value)

        result = {
            "effort": effort,
            "num_traces": n,
            "avg_tokens": total_tokens / n if n > 0 else 0.0,
        }

        for field, values in sorted(metrics.items()):
            result[field] = sum(values) / n if values else 0.0

        aggregated.append(result)
        print(f"Aggregated {n} traces for effort: {effort}")

    return aggregated


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch compute cognitive profiles")
    parser.add_argument(
        "--input", type=str, default="data/graphs", help="Graphs directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/profiles/effort_comparison.json",
        help="Output file",
    )
    parser.add_argument(
        "--traces",
        type=str,
        default="data/traces",
        help="Traces directory (for effort mapping)",
    )
    args = parser.parse_args()

    graphs_dir = Path(args.input)
    output_path = Path(args.output)
    traces_dir = Path(args.traces)

    print("=== Computing profiles ===")
    profiles = compute_all_profiles(graphs_dir)

    if not profiles:
        print("No profiles computed")
        return

    print("\n=== Aggregating by effort ===")
    aggregated = aggregate_by_effort(profiles, traces_dir, graphs_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
