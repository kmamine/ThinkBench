"""CoT trace collector for ThinkBench."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

from .models import LLMClient
from ..extract.schemas import ThoughtUnit

logger = logging.getLogger(__name__)


class TraceCollector:
    def __init__(
        self,
        client: LLMClient,
        output_dir: Path,
        max_retries: int = 3,
    ):
        self.client = client
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized TraceCollector with output_dir: {output_dir}")

    async def collect_single(
        self,
        question: dict,
        model: str,
        run: int = 1,
    ) -> dict:
        """Collect a single CoT trace."""
        trace_id = str(uuid.uuid4())
        messages = [
            {"role": "system", "content": self.client.system_prompt},
            {"role": "user", "content": question["text"]},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw_cot = await self.client.chat(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=20000,
                )
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    delay = 2**attempt
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    f"All {self.max_retries} attempts failed for trace {trace_id}"
                )
                raise

        record = {
            "trace_id": trace_id,
            "model": model,
            "prompt_variant": self.client.prompt_variant,
            "question_id": question["id"],
            "domain": question.get("domain", "unknown"),
            "run": run,
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
        questions_per_batch: int = 5,
    ) -> List[dict]:
        """Collect CoT traces for multiple questions with runs.

        Args:
            questions: List of question dicts
            model: Model name
            runs: Number of runs per question
            questions_per_batch: Process this many questions before saving (for resilience)
        """
        all_traces = []

        for q_idx, question in enumerate(questions):
            for run in range(1, runs + 1):
                try:
                    logger.info(
                        f"[{q_idx + 1}/{len(questions)}] Question {question['id']}, Run {run}/{runs}..."
                    )
                    trace = await self.collect_single(
                        question=question,
                        model=model,
                        run=run,
                    )
                    all_traces.append(trace)
                    logger.info(f"  OK: {trace['token_count']} tokens")

                    # Save in batches for resilience
                    if len(all_traces) % questions_per_batch == 0:
                        self._save_traces(all_traces, model)

                except Exception as e:
                    logger.error(f"  FAILED for {question['id']} run {run}: {e}")
                    continue

        if all_traces:
            self._save_traces(all_traces, model)

        return all_traces

    def _save_traces(self, traces: List[dict], model: str):
        """Save collected traces to file."""
        model_slug = model.replace("/", "-")
        prompt_variant = self.client.prompt_variant
        output_file = self.output_dir / f"traces_{model_slug}_{prompt_variant}.jsonl"

        # Load existing traces
        existing = []
        if output_file.exists():
            with open(output_file) as f:
                existing = [json.loads(line) for line in f]

        # Combine and dedupe
        existing_ids = {t["trace_id"] for t in existing}
        new_traces = [t for t in traces if t["trace_id"] not in existing_ids]

        all_traces = existing + new_traces

        # Save
        with open(output_file, "w") as f:
            for t in all_traces:
                f.write(json.dumps(t) + "\n")

        logger.info(
            f"Saved {len(new_traces)} new traces to {output_file} (total: {len(all_traces)})"
        )
