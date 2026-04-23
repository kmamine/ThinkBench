"""LLM client for ThinkBench - supports vLLM and OpenAI-compatible endpoints."""

import os
from typing import Optional

from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

DEEP_THINKING_SYSTEM_PROMPT = """Think through this problem carefully and thoroughly.
Explore multiple perspectives, consider edge cases, and show your complete reasoning chain.
Think step by step and examine the problem from all angles before reaching a conclusion.
Show all your work and reasoning - do not skip steps."""


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 20000,
        timeout: int = 300,
    ):
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.endpoint = endpoint or os.getenv("VLLM_ENDPOINT", "http://localhost:8978")
        self.model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        base_url = f"{self.endpoint}/v1"
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
        )
        logger.info(
            f"Initialized LLM client: model={self.model}, endpoint={self.endpoint}"
        )
        logger.info(f"System prompt: {DEEP_THINKING_SYSTEM_PROMPT[:50]}...")

    @property
    def system_prompt(self) -> str:
        return DEEP_THINKING_SYSTEM_PROMPT

    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request."""
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        messages = list(messages)

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                messages[0]["content"] + " " + DEEP_THINKING_SYSTEM_PROMPT
            )
        else:
            messages.insert(
                0, {"role": "system", "content": DEEP_THINKING_SYSTEM_PROMPT}
            )

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body=extra_body or {},
            )
            message = resp.choices[0].message
            content = message.content
            reasoning = message.reasoning

            if content:
                return content.strip()
            elif reasoning:
                extracted = self._extract_output_from_reasoning(reasoning)
                if extracted:
                    return extracted
                return reasoning.strip()
            else:
                raise ValueError("Empty response from LLM")
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def _extract_output_from_reasoning(self, reasoning: str) -> Optional[str]:
        """Extract JSON or final answer from reasoning field for reasoning models."""
        import re

        json_patterns = [
            r"\{[^{}]*\}",
            r"\[[\s\S]*\]",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, reasoning)
            if match:
                return match.group(0)

        return None

    async def chat_with_retries(
        self,
        messages: list[dict],
        max_retries: int = 3,
        base_delay: float = 5.0,
        **kwargs,
    ) -> str:
        """Send chat request with exponential backoff retries."""
        import asyncio

        last_error = None
        for attempt in range(max_retries):
            try:
                return await self.chat(messages, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {max_retries} attempts failed: {e}")
        raise last_error
