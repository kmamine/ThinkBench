"""Thought graph extraction: Segmenter - CoT to Thought Units."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..collect.models import LLMClient
from .schemas import ThoughtUnit

logger = logging.getLogger(__name__)

SEGMENT_PROMPT = """You are a reasoning trace parser. Given a chain-of-thought reasoning trace, 
segment it into atomic thought units. Each thought unit should express exactly 
one idea, claim, hypothesis, consideration, or reasoning step.

Rules:
1. A thought unit is the smallest span that makes sense alone
2. Split at logical transitions: new ideas, direction changes, elaborations
3. Preserve the original text verbatim — only add boundary markers
4. Linguistic cues for boundaries include: "Wait,", "Actually,", "Alternatively,",
   "But", "Let me reconsider", "On the other hand", "Now,", "So,", "However,",
   "Hmm,", "What if", "Let me think about", paragraph breaks
5. Do NOT split within a single chain of deduction (A→B→C stays together if 
   it's one logical move)

Output valid JSON array only:"""


class Segmenter:
    def __init__(self, client: LLMClient, confidence_threshold: float = 0.7):
        self.client = client
        self.confidence_threshold = confidence_threshold

    async def segment(self, raw_cot: str) -> list[ThoughtUnit]:
        """Segment a CoT trace into thought units."""
        messages = [
            {"role": "system", "content": SEGMENT_PROMPT},
            {"role": "user", "content": raw_cot},
        ]

        try:
            response = await self.client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=8192,
            )
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise

        thought_units = self._parse_response(response)

        if not thought_units:
            logger.warning(
                "JSON extraction failed, falling back to simple segmentation"
            )
            thought_units = self._fallback_segment(raw_cot)

        logger.info(f"Segmented into {len(thought_units)} thought units")
        return thought_units

    def _parse_response(self, response: str) -> list[ThoughtUnit]:
        """Parse JSON from LLM response."""
        import re
        import json

        text = response.strip()

        json_patterns = [
            r"\[(?:\s*\{[\s\S]*?\}\s*,?\s*){2,}\]",  # At least 2 objects
            r"(\[[\s\S]*?\])(?:\s*[\`*])",
        ]

        json_str = None
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                json_str = match.group(1) if match.lastindex else match.group(0)
                break

        if not json_str:
            return []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        thought_units = []
        for item in data:
            if isinstance(item, dict):
                tu = ThoughtUnit(
                    tu_id=item.get("tu_id", len(thought_units)),
                    text=item.get("text", item.get("content", "")),
                    start_char=item.get("start_char", 0),
                    end_char=item.get("end_char", len(item.get("text", ""))),
                )
            elif isinstance(item, str):
                tu = ThoughtUnit(
                    tu_id=len(thought_units),
                    text=item,
                    start_char=0,
                    end_char=len(item),
                )
            else:
                continue
            if tu.text and len(tu.text) > 10:
                thought_units.append(tu)

        return thought_units

    def _fallback_segment(self, text: str) -> list[ThoughtUnit]:
        """Simple fallback segmentation by splitting on headers and transitions."""
        import re

        segments = re.split(r"\n### |\nStep \d+:|\n\d+\.|\n•|\n- ", text)

        thought_units = []
        pos = 0
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if len(segment) < 30:
                continue
            thought_units.append(
                ThoughtUnit(
                    tu_id=i,
                    text=segment[:500],
                    start_char=pos,
                    end_char=pos + len(segment[:500]),
                )
            )
            pos += len(segment)

        if not thought_units:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            pos = 0
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if len(sent) < 40:
                    continue
                thought_units.append(
                    ThoughtUnit(
                        tu_id=i,
                        text=sent,
                        start_char=pos,
                        end_char=pos + len(sent),
                    )
                )
                pos += len(sent)

        return thought_units

    def _fallback_segment(self, text: str) -> list[ThoughtUnit]:
        """Simple fallback segmentation by splitting on common boundaries."""
        import re

        boundaries = re.split(r"\n\s*(?=\d+\.|[A-Z][a-z]+:|[•\-])", text)

        thought_units = []
        pos = 0
        for i, segment in enumerate(boundaries):
            segment = segment.strip()
            if len(segment) < 20:
                continue
            thought_units.append(
                ThoughtUnit(
                    tu_id=i,
                    text=segment,
                    start_char=pos,
                    end_char=pos + len(segment),
                )
            )
            pos += len(segment)

        return thought_units

    async def segment_batch(
        self, raw_cots: list[tuple[str, str]]
    ) -> list[list[ThoughtUnit]]:
        """Segment multiple traces."""
        tasks = []
        for trace_id, raw_cot in raw_cots:
            tasks.append(self.segment(raw_cot))

        results = await self.client.client.chat.completions.create(*tasks)
        return [r for r in results if not isinstance(r, Exception)]


def load_traces(path: Path) -> list[tuple[str, dict]]:
    """Load traces from JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            trace = json.loads(line)
            traces.append((trace["trace_id"], trace))
    return traces
