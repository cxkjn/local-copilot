"""Context compression node: semantic summarization of old history past the token limit.

Keeps the most recent rounds verbatim so the active interaction is never affected; only
older history is summarized (progressive compression). The summary is stored in the
`conversation_summary` state channel and injected as a SystemMessage at inference time;
removed history is dropped with `RemoveMessage`. If summarization fails, the oldest
messages are dropped instead of failing the conversation.
"""

import logging

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import RemoveMessage

from agents.code_assistant import model as model_module
from agents.code_assistant.state import AgentState, content_to_str
from agents.code_assistant.tokenizer import count_messages_tokens
from core import settings

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """你是会话摘要器。把以下历史对话压缩成一段简洁的中文语义摘要，必须保留：
1. 用户提出的关键事实、偏好与明确要求；
2. 已经达成的决策与结论；
3. 尚未解决或待办的事项；
4. 涉及的重要参考资料。
只输出摘要正文，不要任何多余说明。"""


async def summarize_history(
    model,
    history: list[BaseMessage],
    config: RunnableConfig,
    previous_summary: str = "",
) -> str:
    prompt: list[BaseMessage] = [SystemMessage(content=_SUMMARY_PROMPT)]
    if previous_summary:
        prompt.append(
            SystemMessage(
                content=f"已有历史摘要（在此基础上增量更新，不要重复叙述）:\n{previous_summary}"
            )
        )
    prompt.extend(history)
    prompt.append(HumanMessage(content="请输出上述历史对话的语义摘要。"))
    response = await model.ainvoke(prompt, config)
    return content_to_str(response.content)


async def context_compress_node(state: AgentState, config: RunnableConfig) -> dict:
    messages = state.get("messages") or []
    if len(messages) < 2:
        return {}
    model_name = config.get("configurable", {}).get("model", settings.DEFAULT_MODEL)
    if count_messages_tokens(messages, model_name) <= settings.MAX_TOKEN_LIMIT:
        return {}

    keep = max(2, settings.CONTEXT_KEEP_RECENT_ROUNDS * 2)
    history = messages[:-keep]
    if not history:
        return {}

    previous = state.get("conversation_summary") or ""
    m = model_module.resolve_model(config)
    try:
        summary = await summarize_history(m, history, config, previous_summary=previous)
    except Exception as e:
        logger.warning("context summarization failed (%s); dropping oldest messages", e)
        summary = previous

    removals = [RemoveMessage(id=msg.id) for msg in history if msg.id]
    updates: dict = {"conversation_summary": summary}
    if len(removals) == len(history):
        updates["messages"] = removals
    else:
        logger.warning("some history messages lack ids; keeping them in state")
    return updates
