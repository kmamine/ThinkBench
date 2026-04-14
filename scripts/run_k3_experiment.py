"""Run K=3 experiment for LOW, MEDIUM, HIGH efforts."""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thinkbench.collect.models import LLMClient, ThinkingEffort
from thinkbench.collect.collector import TraceCollector


async def main():
    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    questions_path = Path("data/questions/ethical_dilemmas.jsonl")
    with open(questions_path) as f:
        questions = [json.loads(line) for line in f]

    print(f"Loaded {len(questions)} questions")

    efforts = [ThinkingEffort.LOW, ThinkingEffort.MEDIUM, ThinkingEffort.HIGH]
    K = 3
    runs_per_effort = {}

    for effort in efforts:
        print(f"\n=== Collecting K={K} traces for {effort.value.upper()} ===")

        client = LLMClient(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            thinking_effort=effort,
        )

        collector = TraceCollector(client, output_dir=Path("data/traces"))

        traces = []
        for i, q in enumerate(questions[:1]):  # Just first question for now
            for run in range(1, K + 1):
                print(f"  Collecting run {run}/{K}...")
                trace = await collector.collect_single(
                    question=q,
                    model=model,
                    run=run,
                    thinking_effort=effort,
                )
                traces.append(trace)

        # Save traces
        output_file = (
            f"data/traces/traces_{model.replace('/', '-')}_{effort.value}_k{K}.jsonl"
        )
        with open(output_file, "w") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")

        runs_per_effort[effort.value] = len(traces)
        print(f"  Saved {len(traces)} traces to {output_file}")

    print(f"\n=== Summary ===")
    for effort, count in runs_per_effort.items():
        print(f"  {effort}: {count} traces")

    print("\nNext steps:")
    print("  1. python scripts/extract_graphs.py")
    print("  2. python scripts/compute_profiles.py")


if __name__ == "__main__":
    asyncio.run(main())
