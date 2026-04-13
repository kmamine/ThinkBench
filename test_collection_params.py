"""Test script for multi-run collection with different thinking efforts."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkbench.collect.models import LLMClient, ThinkingEffort
from thinkbench.collect.collector import TraceCollector, load_questions


async def run_experiment():
    """Run collection with different K values and thinking efforts."""
    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    # Test parameters
    k_values = [3, 5, 15]
    thinking_efforts = [
        ThinkingEffort.LOW,
        ThinkingEffort.MEDIUM,
        ThinkingEffort.HIGH,
        ThinkingEffort.MAX,
    ]

    question = {
        "id": "TEST_001",
        "domain": "test",
        "text": "What are the ethical implications of AI decision-making in self-driving cars?",
    }

    print(f"Model: {model}")
    print(f"Endpoint: {endpoint}")
    print()

    results_summary = []

    for effort in thinking_efforts:
        print(
            f"=== Thinking Effort: {effort.value.upper()} ({effort.tokens} tokens) ==="
        )

        client = LLMClient(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            thinking_effort=effort,
        )

        collector = TraceCollector(client, Path("data/traces"))

        for k in k_values:
            print(f"  Collecting with K={k}...")

            traces = await collector.collect_batch(
                [question], model, runs=k, thinking_effort=effort
            )

            # Save traces
            output = collector.save_traces(
                traces, model, suffix=f"_k{k}", thinking_effort=effort
            )

            # Get stats
            tokens = [t["token_count"] for t in traces]
            avg_tokens = sum(tokens) / len(tokens) if tokens else 0

            print(f"    K={k}: {len(traces)} traces, avg tokens: {avg_tokens:.0f}")

            results_summary.append(
                {
                    "effort": effort.value,
                    "k": k,
                    "traces": len(traces),
                    "avg_tokens": avg_tokens,
                }
            )

        print()

    print("=== Summary ===")
    print(f"{'Effort':<10} {'K':<5} {'Traces':<8} {'Avg Tokens':<12}")
    print("-" * 40)
    for r in results_summary:
        print(
            f"{r['effort']:<10} {r['k']:<5} {r['traces']:<8} {r['avg_tokens']:<12.0f}"
        )


async def quick_test():
    """Quick single-run test."""
    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    question = {
        "id": "TEST_001",
        "domain": "test",
        "text": "What are the ethical implications of AI decision-making in self-driving cars?",
    }

    print("Quick test: Comparing thinking efforts...")
    print()

    for effort in [ThinkingEffort.LOW, ThinkingEffort.HIGH, ThinkingEffort.MAX]:
        client = LLMClient(
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            thinking_effort=effort,
        )

        collector = TraceCollector(client, Path("data/traces"))

        trace = await collector.collect_single(
            question, model, run=1, thinking_effort=effort
        )

        print(f"{effort.value.upper():>6}: {trace['token_count']} tokens")
        print(f"  Preview: {trace['raw_cot'][:150]}...")
        print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(quick_test())
    else:
        asyncio.run(run_experiment())
