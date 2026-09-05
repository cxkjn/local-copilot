"""Memory management: four-type memory + persistent block rules.

Redis-backed when `REDIS_URL` is configured; otherwise an in-memory fallback keeps
the assistant usable offline and in tests. Memory extraction is awaited in the graph's
terminal node so a newly persisted block rule is enforced before the next turn begins.
"""

import json
import logging
from collections.abc import Awaitable, Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agents.code_assistant.state import AgentMemory, BlockRule, MemoryExtraction, content_to_str
from core import settings

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None

_REDIS_KEY_MEMORY = "agent:memory:{session_id}"
_REDIS_KEY_RULES = "agent:block_rule:{user_id}"

Judge = Callable[[str, list[BlockRule]], Awaitable[BlockRule | None]]


class MemoryStore:
    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None) -> None:
        self._redis_url = redis_url or settings.REDIS_URL
        self._ttl = ttl_seconds if ttl_seconds is not None else settings.MEMORY_TTL_SECONDS
        self._redis: object | None = None
        self._mem: dict[str, AgentMemory] = {}
        self._rules: dict[str, list[BlockRule]] = {}

    async def _client(self):
        if self._redis is None:
            if self._redis_url and aioredis is not None:
                try:
                    client = aioredis.from_url(self._redis_url, decode_responses=True)
                    await client.ping()
                    self._redis = client
                except Exception as e:
                    logger.warning("Redis unavailable (%s); using in-memory fallback", e)
                    self._redis = False
            else:
                self._redis = False
        return self._redis

    async def aget_memory(self, session_id: str) -> AgentMemory | None:
        client = await self._client()
        if client:
            raw = await client.get(_REDIS_KEY_MEMORY.format(session_id=session_id))
            if not raw:
                return None
            try:
                return AgentMemory.model_validate_json(raw)
            except Exception:
                logger.exception("failed to parse memory for session %s", session_id)
                return None
        return self._mem.get(session_id)

    async def aset_memory(self, session_id: str, memory: AgentMemory) -> None:
        client = await self._client()
        if client:
            await client.set(
                _REDIS_KEY_MEMORY.format(session_id=session_id),
                memory.model_dump_json(),
                ex=self._ttl,
            )
        else:
            self._mem[session_id] = memory

    async def aget_block_rules(self, user_id: str) -> list[BlockRule]:
        client = await self._client()
        if client:
            raw = await client.get(_REDIS_KEY_RULES.format(user_id=user_id))
            if not raw:
                return []
            try:
                rules = [BlockRule.model_validate(r) for r in json.loads(raw)]
            except Exception:
                logger.exception("failed to parse block rules for user %s", user_id)
                return []
            return [r for r in rules if r.enable]
        return [r for r in self._rules.get(user_id, []) if r.enable]

    async def aadd_block_rule(self, user_id: str, rule: BlockRule) -> None:
        client = await self._client()
        if client:
            key = _REDIS_KEY_RULES.format(user_id=user_id)
            raw = await client.get(key)
            existing = json.loads(raw) if raw else []
            existing.append(rule.model_dump())
            await client.set(key, json.dumps(existing))
        else:
            self._rules.setdefault(user_id, []).append(rule)

    async def check_block_rules(
        self, user_id: str, text: str, *, judge: Judge | None = None
    ) -> BlockRule | None:
        rules = await self.aget_block_rules(user_id)
        logger.warning("MEM-DEBUG check user=%r text=%r rules=%d judge=%s", user_id, text, len(rules), judge is not None)
        if not rules or not text:
            return None
        if judge is not None:
            try:
                return await judge(text, rules)
            except Exception:
                logger.exception("block rule semantic judge failed; falling back to keyword match")
        candidates = [
            r
            for r in rules
            if (r.trigger_pattern and r.trigger_pattern in text)
            or any(k and k in text for k in r.keywords)
        ]
        return candidates[0] if candidates else None

    async def aclose(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None


_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(redis_url=settings.REDIS_URL)
    return _store


_MEMORY_EXTRACT_PROMPT = """你是记忆抽取器。阅读完整会话历史，输出结构化 JSON：
- preference: 用户的使用偏好、回复风格要求（无则空字符串）
- feedback: 用户对助手回复的评价与意见（无则空字符串）
- knowledge: 用户提供的事实与背景知识（无则空字符串）
- reference: 对话中涉及的参考资料、链接（无则空字符串）
- block_intent: 用户表达的"以后拒绝/不要再"类持久拒绝意图的原文语义（无则空字符串）
- block_keywords: block_intent 的核心主题短词（2-4 字的名词/动词，例如"笑话"而非"别再讲笑话"；是用户后续请求中最可能出现的匹配单元，无则空列表）
严格只输出 JSON。"""


def _build_transcript(messages: Sequence[BaseMessage]) -> str:
    roles = {"human": "用户", "ai": "助手", "tool": "工具"}
    lines = []
    for msg in messages:
        text = content_to_str(msg.content).strip()
        if text:
            lines.append(f"{roles.get(msg.type, msg.type)}: {text}")
    return "\n".join(lines)


async def background_extract_memory(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    session_id: str,
    user_id: str,
    store: MemoryStore | None = None,
) -> None:
    store = store or get_memory_store()
    try:
        runnable = model.with_structured_output(MemoryExtraction, method="json_mode").with_config(tags=["skip_stream"])
        transcript = _build_transcript(messages)
        extraction: MemoryExtraction = await runnable.ainvoke(
            [SystemMessage(content=_MEMORY_EXTRACT_PROMPT), HumanMessage(content=transcript)]
        )
        logger.warning("MEM-DEBUG extract block_intent=%r block_keywords=%r", extraction.block_intent, extraction.block_keywords)
        memory = AgentMemory(
            preference=extraction.preference.strip(),
            feedback=extraction.feedback.strip(),
            knowledge=extraction.knowledge.strip(),
            reference=extraction.reference.strip(),
        )
        await store.aset_memory(session_id, memory)
        intent = extraction.block_intent.strip()
        if intent:
            await store.aadd_block_rule(
                user_id,
                BlockRule(
                    trigger_pattern=intent,
                    semantic_desc=f"用户指令持久拒绝: {intent}",
                    keywords=[k.strip() for k in extraction.block_keywords if k.strip()],
                ),
            )
            logger.info("persisted block rule for user %s: %s", user_id, intent)
    except Exception:
        logger.exception("background memory extraction failed for session %s", session_id)
