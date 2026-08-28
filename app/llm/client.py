from typing import Protocol

from openai import AsyncOpenAI


class LLMClient(Protocol):
    async def generate(self, *, system_prompt: str, user_prompt: str) -> str: ...

    async def close(self) -> None: ...


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        if not model.strip():
            raise ValueError("LLM model must not be empty")

        self._model = model
        self._max_tokens = max_tokens

        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url or None,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
            max_tokens=self._max_tokens,
        )

        content = response.choices[0].message.content

        return content.strip() if content else ""

    async def close(self) -> None:
        await self._client.close()
