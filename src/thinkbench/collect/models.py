"""LLM client for ThinkBench - supports vLLM and OpenAI-compatible endpoints."""

import os
from typing import Optional

from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPTS: dict[str, str] = {
    "empty": "",
    "pure": "You are a helpful assistant.",
    "normal": "You are a helpful assistant. Think carefully before answering.",
    "eliciting": (
        "You are a careful thinker. Before answering, work through the problem "
        "thoroughly using the following approach:\n\n"
        "- Spread wide first: explore as many distinct angles, framings, and "
        "stakeholder perspectives as you can before going deep on any one\n"
        "- Reframe: try restating the problem from at least two completely different "
        "starting points\n"
        "- Go deep: for each idea you raise, follow it — ask what it implies, what "
        "supports it, what limits it\n"
        "- Challenge yourself: after developing an idea, actively look for what is "
        "wrong with it or what it misses\n"
        "- Connect: look for relationships between the threads you've opened — do any "
        "of them reinforce, contradict, or combine with each other?\n"
        "- Reflect: periodically check whether your reasoning is going in circles or "
        "missing something important\n"
        "- Converge: at the end, bring your threads together into a coherent position "
        "that acknowledges the tensions you found\n\n"
        "Think out loud. Do not skip steps."
    ),
}

PROMPT_VARIANTS = list(SYSTEM_PROMPTS.keys())


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        prompt_variant: str = "normal",
        temperature: float = 0.7,
        max_tokens: int = 20000,
        timeout: int = 300,
    ):
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.endpoint = endpoint or os.getenv("VLLM_ENDPOINT", "http://localhost:8978")
        self.model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
        self.prompt_variant = prompt_variant if prompt_variant in SYSTEM_PROMPTS else "normal"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        base_url = f"{self.endpoint}/v1"
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
        )
        logger.info(f"Initialized LLM client: model={self.model}, endpoint={self.endpoint}")
        logger.info(f"Prompt variant: {self.prompt_variant}")

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS[self.prompt_variant]

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

        system_prompt = SYSTEM_PROMPTS[self.prompt_variant]
        # Strip any existing system turn first
        if messages and messages[0].get("role") == "system":
            messages = messages[1:]
        # Only prepend system message when the variant has a non-empty prompt
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

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
