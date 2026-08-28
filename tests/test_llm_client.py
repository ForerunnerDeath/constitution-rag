from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.client import OpenAICompatibleLLMClient


@pytest.mark.asyncio
async def test_generate_returns_stripped_content() -> None:
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(content="  Конституция устанавливает это правило.  ")
        )
    ]

    sdk_client = MagicMock()
    sdk_client.chat.completions.create = AsyncMock(return_value=response)
    sdk_client.close = AsyncMock()

    with patch(
        "app.llm.client.AsyncOpenAI",
        return_value=sdk_client,
    ):
        client = OpenAICompatibleLLMClient(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            max_tokens=512,
            timeout_seconds=20.0,
        )

        result = await client.generate(
            system_prompt="system rules",
            user_prompt="user question",
        )

    assert result == "Конституция устанавливает это правило."

    sdk_client.chat.completions.create.assert_awaited_once_with(
        model="test-model",
        messages=[
            {
                "role": "system",
                "content": "system rules",
            },
            {
                "role": "user",
                "content": "user question",
            },
        ],
        temperature=0,
        max_tokens=512,
    )


@pytest.mark.asyncio
async def test_close_closes_sdk_client() -> None:
    sdk_client = MagicMock()
    sdk_client.close = AsyncMock()

    with patch(
        "app.llm.client.AsyncOpenAI",
        return_value=sdk_client,
    ):
        client = OpenAICompatibleLLMClient(
            model="test-model",
            api_key="test-key",
            base_url="http://localhost:1234/v1",
            max_tokens=512,
            timeout_seconds=20.0,
        )

        await client.close()

    sdk_client.close.assert_awaited_once()
