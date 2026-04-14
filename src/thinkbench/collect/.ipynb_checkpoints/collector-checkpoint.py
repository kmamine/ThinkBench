"""CoT trace collector for ThinkBench."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging
import os

from .models import LLMClient, ThinkingEffort
from ..extract.schemas import ThoughtUnit, Edge, ThoughtGraph

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Think carefully about this problem. Show your reasoning step by step."


class TraceCollector:
    def __init__(
        self,
        client: LLMClient,
        output_dir: Path,
        system_prompt: str = SYSTEM_PROMPT,
        max_retries: int = 3,
    ):
        self.client = client
        self.output_dir = Path(output_dir)
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def collect_single(
        self,
        question: dict,
        model: str,
        run: int = 1,
        thinking_effort: Optional[ThinkingEffort] = None,
    ) -> dict:
        """Collect a single CoT trace."""
        trace_id = str(uuid.uuid4())
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question["text"]},
        ]

        for attempt in range(self.max_retries):
            try:
                raw_cot = await self.client.chat(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096,
                    thinking_effort=thinking_effort,
                )
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)

        record = {
            "trace_id": trace_id,
            "model": model,
            "question_id": question["id"],
            "domain": question["domain"],
            "run": run,
            "thinking_effort": str(thinking_effort) if thinking_effort else None,
            "raw_cot": raw_cot,
            "token_count": len(raw_cot.split()),
            "collected_at": datetime.utcnow().isoformat() + "Z",
        }

        return record

    async def collect_batch(
        self,
        questions: list[dict],
        model: str,
        runs: int = 1,
        thinking_effort: Optional[ThinkingEffort] = None,
    ) -> list[dict]:
        """Collect multiple traces with parallel execution."""
        tasks = []
        for q in questions:
            for run in range(1, runs + 1):
                tasks.append(self.collect_single(q, model, run, thinking_effort))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    def save_traces(
        self,
        traces: list[dict],
        model: str,
        suffix: str = "",
        thinking_effort: Optional[ThinkingEffort] = None,
    ):
        safe_model = model.replace("/", "-")
        effort_suffix = f"_{thinking_effort.value}" if thinking_effort else ""
        output_path = (
            self.output_dir / f"traces_{safe_model}{effort_suffix}{suffix}.jsonl"
        )
        with open(output_path, "w") as f:
            for trace in traces:
                f.write(json.dumps(trace) + "\n")
        logger.info(f"Saved {len(traces)} traces to {output_path}")
        return output_path


def load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


async def run_collection(
    questions_path: str,
    model: str,
    runs: int = 1,
    output_dir: str = "data/traces",
    thinking_effort: Optional[ThinkingEffort] = None,
    **client_kwargs,
):
    """CLI entry point for trace collection."""
    client = LLMClient(thinking_effort=thinking_effort, **client_kwargs)
    collector = TraceCollector(client, Path(output_dir))

    questions = load_questions(Path(questions_path))
    logger.info(f"Loaded {len(questions)} questions, running {runs} times each")

    traces = await collector.collect_batch(questions, model, runs, thinking_effort)
    output_path = collector.save_traces(traces, model, thinking_effort=thinking_effort)

    return output_path
