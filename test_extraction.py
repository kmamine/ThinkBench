"""Test script for full extraction pipeline."""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from thinkbench.collect.models import LLMClient
from thinkbench.extract import Segmenter, Classifier, Linker, build_graph


async def test_extraction():
    """Test the full extraction pipeline."""
    api_key = os.getenv("VLLM_API_KEY", "sofia-token-j7y6qXDOTJ6grLvo")
    endpoint = os.getenv("VLLM_ENDPOINT", "http://10.17.1.57:8978")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")

    print(f"Testing extraction pipeline with model: {model}")
    print(f"Endpoint: {endpoint}")

    client = LLMClient(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
    )

    traces_path = Path("data/traces/traces_Qwen-Qwen3.5-35B-A3B.jsonl")
    if not traces_path.exists():
        print("No traces found. Run test_collector.py first.")
        return

    with open(traces_path) as f:
        trace = json.loads(f.readline())

    print(f"\n--- Loaded trace: {trace['trace_id']}")
    print(f"Question: {trace['question_id']}")
    print(f"Token count: {trace['token_count']}")

    segmenter = Segmenter(client)
    classifier = Classifier(client)
    linker = Linker(client)

    print("\n--- Stage A: Segmentation ---")
    tus = await segmenter.segment(trace["raw_cot"])
    print(f"Segmented into {len(tus)} thought units")
    if tus:
        print(f"First TU: {tus[0].text[:100]}...")

    print("\n--- Stage B: Classification ---")
    classified_tus = await classifier.classify(tus)
    print(f"Classified {len(classified_tus)} thought units")
    type_counts = {}
    for tu in classified_tus:
        if tu.node_type:
            type_counts[tu.node_type.value] = type_counts.get(tu.node_type.value, 0) + 1
    print(f"Type distribution: {type_counts}")

    print("\n--- Stage C: Linking ---")
    edges = await linker.link(classified_tus)
    print(f"Built {len(edges)} edges")
    edge_types = {}
    for e in edges:
        edge_types[e.edge_type.value] = edge_types.get(e.edge_type.value, 0) + 1
    print(f"Edge type distribution: {edge_types}")

    print("\n--- Building ThoughtGraph ---")
    graph = build_graph(trace, classified_tus, edges)
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Trace ID: {graph.trace_id}")
    print(f"Model: {graph.model}")

    output_path = Path("data/graphs/graph_test.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(graph.model_dump(), f, indent=2)
    print(f"\nSaved graph to: {output_path}")


if __name__ == "__main__":
    asyncio.run(test_extraction())
