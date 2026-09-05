"""Token counting with tiktoken, degrading gracefully when it is unavailable."""

from functools import lru_cache

from langchain_core.messages import BaseMessage

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

_DEFAULT_ENCODING = "cl100k_base"
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=32)
def _encoding_for_model(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)


def count_tokens(text: str, model: str | None = None) -> int:
    if not text:
        return 0
    encoding = _encoding_for_model(model or "gpt-4o")
    if encoding is None:
        return max(1, len(text) // _CHARS_PER_TOKEN)
    try:
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // _CHARS_PER_TOKEN)


def count_messages_tokens(messages: list[BaseMessage], model: str | None = None) -> int:
    total = 0
    for message in messages:
        total += count_tokens(message.type, model)
        content = message.content
        if isinstance(content, str):
            total += count_tokens(content, model)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    total += count_tokens(item, model)
                elif isinstance(item, dict) and item.get("type") == "text":
                    total += count_tokens(str(item.get("text", "")), model)
        total += 4
    return total
