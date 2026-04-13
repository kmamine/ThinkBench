"""Test script for ThinkBench collector with various parameters."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkbench.collect.models import LLMClient
from thinkbench.collect.collector import TraceCollector, load_questions


async def test_collector():
    """Test the collector with configurable parameters."""
    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    print(f"Testing with model: {model}")
    print(f"Endpoint: {endpoint}")

    client = LLMClient(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        temperature=0.7,
        max_tokens=2048,
    )

    question = {
        "id": "TEST_001",
        "domain": "test",
        "text": "What are the ethical implications of AI decision-making in self-driving cars?",
    }

    collector = TraceCollector(client, Path("data/traces"))

    print("\n--- Testing single trace collection ---")
    trace = await collector.collect_single(question, model, run=1)
    print(f"Trace ID: {trace['trace_id']}")
    print(f"Token count: {trace['token_count']}")
    print(f"Raw CoT (first 500 chars): {trace['raw_cot'][:500]}...")

    output = collector.save_traces([trace], model)
    print(f"Saved to: {output}")


async def test_batch():
    """Test batch collection with different temperatures."""
    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    client = LLMClient(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
    )

    questions = [
        {
            "id": "TEST_001",
            "domain": "test",
            "text": "What are the ethical implications of AI decision-making?",
        },
        {
            "id": "TEST_002",
            "domain": "test",
            "text": "How should cities design public transit systems?",
        },
    ]

    collector = TraceCollector(client, Path("data/traces"))

    print("\n--- Testing batch collection ---")
    traces = await collector.collect_batch(questions, model, runs=1)
    print(f"Collected {len(traces)} traces")

    output = collector.save_traces(traces, model)
    print(f"Saved to: {output}")


if __name__ == "__main__":
    asyncio.run(test_collector())
