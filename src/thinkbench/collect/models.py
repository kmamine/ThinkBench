"""LLM client for ThinkBench - supports vLLM and OpenAI-compatible endpoints."""

import os
import asyncio
from enum import Enum
from typing import Optional

from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)


class ThinkingEffort(str, Enum):
    """Thinking effort levels for reasoning models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"

    @property
    def tokens(self) -> int:
        """Return thinking budget in tokens for models that support it."""
        tokens = {
            ThinkingEffort.LOW: 200,
            ThinkingEffort.MEDIUM: 1000,
            ThinkingEffort.HIGH: 4000,
            ThinkingEffort.MAX: 16000,
        }
        return tokens[self]

    @property
    def system_prompt_modifier(self) -> str:
        """Return system prompt modifier for prompt-based effort control."""
        modifiers = {
            ThinkingEffort.LOW: "Think briefly and give a concise answer.",
            ThinkingEffort.MEDIUM: "Think through this problem carefully.",
            ThinkingEffort.HIGH: "Think step by step. Show all your reasoning. Consider multiple angles.",
            ThinkingEffort.MAX: "Think deeply and exhaustively. Explore all possibilities. Consider edge cases. Show comprehensive reasoning.",
        }
        return modifiers[self]

    @property
    def use_thinking_param(self) -> bool:
        """Whether to use thinking_budget parameter (for models that support it)."""
        return False  # Default to False - use prompt-based control


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120,
        thinking_effort: Optional[ThinkingEffort] = None,
    ):
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.endpoint = endpoint or os.getenv("VLLM_ENDPOINT", "http://localhost:8978")
        self.model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-4B")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.thinking_effort = thinking_effort

        base_url = f"{self.endpoint}/v1"
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
        )
        logger.info(
            f"Initialized LLM client: model={self.model}, endpoint={self.endpoint}, thinking_effort={thinking_effort}"
        )

    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[dict] = None,
        thinking_effort: Optional[ThinkingEffort] = None,
    ) -> str:
        """Send a chat completion request."""
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        thinking_effort = thinking_effort or self.thinking_effort

        messages = list(messages)  # Copy to avoid modifying original

        body = extra_body or {}

        if thinking_effort:
            if thinking_effort.use_thinking_param:
                body["thinking_budget"] = thinking_effort.tokens
            else:
                modifier = thinking_effort.system_prompt_modifier
                if messages and messages[0].get("role") == "system":
                    messages[0]["content"] = messages[0]["content"] + " " + modifier
                else:
                    messages.insert(0, {"role": "system", "content": modifier})

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body=body,
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

        # First try to find JSON
        json_patterns = [
            r"\{[^{}]*\}",
            r"\[[\s\S]*\]",
        ]

        for pattern in json_patterns:
            match = re.search(pattern, reasoning)
            if match:
                return match.group(0)

        # If no JSON, look for type codes directly in reasoning
        type_pattern = r"\b(HYP|RFR|ANA|BRS|JUS|SPC|IMP|CON|CRT|CMP|MET|SYN)\b"
        type_match = re.search(type_pattern, reasoning)
        if type_match:
            # Return just the type code so classifier can parse it
            return type_match.group(1)

        return None

    async def chat_async(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[dict] = None,
    ) -> str:
        """Alias for chat (async-compatible)."""
        return await self.chat(messages, temperature, max_tokens, top_p, extra_body)

    def chat_sync(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[dict] = None,
    ) -> str:
        """Synchronous wrapper for chat."""
        return asyncio.run(
            self.chat(messages, temperature, max_tokens, top_p, extra_body)
        )


class BatchLLMClient:
    """Batch client for parallel LLM calls with rate limiting."""

    def __init__(self, client: LLMClient, max_concurrent: int = 5):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def call(self, messages: list[dict], **kwargs) -> str:
        async with self.semaphore:
            return await self.client.chat(messages, **kwargs)

    async def batch(self, requests: list[list[dict]], **kwargs) -> list[str]:
        tasks = [self.call(req, **kwargs) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
