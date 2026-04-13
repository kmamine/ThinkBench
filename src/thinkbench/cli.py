"""CLI for ThinkBench."""

import os
import asyncio
import logging
from pathlib import Path

import typer

app = typer.Typer(help="ThinkBench - LLM cognitive profiling")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@app.command()
def collect(
    questions: Path = typer.Option(..., help="Path to questions JSONL file"),
    model: str = typer.Option("Qwen/Qwen3.5-4B", help="Model name"),
    runs: int = typer.Option(1, help="Number of runs per question"),
    output: Path = typer.Option(Path("data/traces"), help="Output directory"),
    api_key: str = typer.Option(None, help="API key (or set VLLM_API_KEY)"),
    endpoint: str = typer.Option(None, help="API endpoint (or set VLLM_ENDPOINT)"),
):
    """Collect CoT traces from a model."""
    from .collect import LLMClient, TraceCollector, load_questions

    client = LLMClient(
        api_key=api_key or os.getenv("VLLM_API_KEY"),
        endpoint=endpoint or os.getenv("VLLM_ENDPOINT"),
        model=model,
    )
    collector = TraceCollector(client, output)

    questions_list = load_questions(questions)
    typer.echo(f"Loaded {len(questions_list)} questions")

    traces = asyncio.run(collector.collect_batch(questions_list, model, runs))
    output_path = collector.save_traces(traces, model)

    typer.echo(f"Collected {len(traces)} traces -> {output_path}")


if __name__ == "__main__":
    app()
