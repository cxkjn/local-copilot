"""Plan-mode: LLM task decomposition plus parallel sub-agent execution.

Each sub-task runs in its own sub-agent loop over the shared workspace; results are
aggregated back into the conversation for the main agent to finalize.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agents.code_assistant import model as model_module
from agents.code_assistant.executor import execute_tool
from agents.code_assistant.security import FiveLayerSecurity
from agents.code_assistant.state import AgentState, CheckResult, PlanOutput, content_to_str
from core import settings

logger = logging.getLogger(__name__)

_PLAN_PROMPT = """你是一个任务规划器。把用户请求拆解为若干相互独立的子任务。
要求：
- 每个子任务必须自包含、可独立执行，不依赖其他子任务的结果；
- 子任务数量不超过 {max_subtasks} 个；
- 输出 JSON: {{"subtasks": ["子任务1", "子任务2", ...], "reasoning": "拆解思路"}}。"""

_SUBAGENT_PROMPT = """你是一个专注的子Agent，正在执行子任务：
{task}

你可以调用工具完成该任务。完成后用简洁的中文总结执行结果。"""


def _should_plan(last_human: str, config: RunnableConfig) -> bool:
    if config.get("configurable", {}).get("plan_mode"):
        return True
    return len(last_human) >= settings.PLAN_MODE_COMPLEXITY_THRESHOLD


def _last_human_message(state: AgentState) -> str | None:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return content_to_str(message.content)
    return None


async def plan_split_node(state: AgentState, config: RunnableConfig) -> dict:
    if state.get("planning_done"):
        return {}
    updates: dict = {"planning_done": True}
    last_human = _last_human_message(state)
    if last_human is None or not _should_plan(last_human, config):
        return updates

    m = model_module.resolve_model(config)
    runnable = m.with_structured_output(PlanOutput, method="json_mode").with_config(tags=["skip_stream"])
    try:
        response: PlanOutput = await runnable.ainvoke(
            [
                SystemMessage(
                    content=_PLAN_PROMPT.format(max_subtasks=settings.PLAN_MODE_MAX_SUBTASKS)
                ),
                HumanMessage(content=last_human),
            ],
            config,
        )
    except Exception:
        logger.exception("plan split failed; continuing without decomposition")
        return updates
    subtasks = [s.strip() for s in response.subtasks if s and s.strip()]
    if subtasks:
        updates["subtasks"] = subtasks[: settings.PLAN_MODE_MAX_SUBTASKS]
    return updates


async def run_subtask(
    task: str,
    index: int,
    state: AgentState,
    config: RunnableConfig,
    tools: Sequence[BaseTool],
    model: BaseChatModel,
    security: FiveLayerSecurity,
) -> dict[str, str]:
    tool_map = {t.name: t for t in tools}
    messages = [
        SystemMessage(content=_SUBAGENT_PROMPT.format(task=task)),
        HumanMessage(content=task),
    ]
    bound = model.bind_tools(tools)
    output = ""
    for _ in range(settings.PLAN_MODE_MAX_SUB_STEPS):
        response = await bound.ainvoke(messages, config)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            output = content_to_str(response.content)
            break
        for tc in tool_calls:
            decision = await security.check_tool_call(
                user_id=state.get("user_id", ""),
                tool_name=tc["name"],
                tool_args=tc.get("args") or {},
            )
            if decision.result is CheckResult.REJECT:
                if decision.requires_approval:
                    content = f"高危工具 {tc['name']} 在子任务中跳过（需人工确认）"
                else:
                    content = f"安全网关拦截: {decision.reason}"
                messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
                continue
            tool = tool_map.get(tc["name"])
            if tool is None:
                messages.append(
                    ToolMessage(content=f"未知工具: {tc['name']}", tool_call_id=tc["id"])
                )
                continue
            result = await execute_tool(tool, tc.get("args") or {})
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    return {"task": task, "output": output}


def make_subtask_exec_node(tools: Sequence[BaseTool]):
    async def subtask_exec_node(state: AgentState, config: RunnableConfig) -> dict:
        subtasks = state.get("subtasks") or []
        if not subtasks:
            return {}
        m = model_module.resolve_model(config)
        security = FiveLayerSecurity()
        results = await asyncio.gather(
            *[
                run_subtask(task, index, state, config, tools, m, security)
                for index, task in enumerate(subtasks)
            ]
        )
        lines = [
            f"子任务{index + 1}: {r['task']}\n结果: {r['output'] or '（无输出）'}"
            for index, r in enumerate(results)
        ]
        summary = "子Agent并行执行完成，汇总如下：\n" + "\n\n".join(lines)
        return {
            "subtasks": [],
            "plan_results": list(results),
            "messages": [AIMessage(content=summary)],
        }

    return subtask_exec_node


def route_after_plan(state: AgentState) -> Literal["subtask_exec", "llm_infer"]:
    return "subtask_exec" if state.get("subtasks") else "llm_infer"
