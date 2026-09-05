"""Model resolution for the assistant, with optional prompt-caching extras.

The framework's `get_model` already adapts provider differences (OpenAI-compatible,
DeepSeek, Anthropic, ...). This module adds the doc's `extra_body` prompt-caching
extension for OpenAI-compatible endpoints and a single monkeypatchable seam for tests.
"""

import inspect

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from core import get_model, settings
from schema.models import OpenAICompatibleName

_CACHING_SUPPORTED = "extra_body" in inspect.signature(ChatOpenAI).parameters


def resolve_model(config: RunnableConfig | None = None) -> BaseChatModel:
    model_name = (config or {}).get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    if (
        _CACHING_SUPPORTED
        and settings.PROMPT_CACHING_ENABLED
        and settings.COMPATIBLE_BASE_URL
        and settings.COMPATIBLE_MODEL
        and model_name == OpenAICompatibleName.OPENAI_COMPATIBLE
    ):
        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY.get_secret_value()
            if settings.COMPATIBLE_API_KEY
            else "not-set",
            extra_body={"prompt_caching_enabled": True},
        )
    return get_model(model_name)
