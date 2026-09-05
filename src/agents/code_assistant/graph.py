"""Main graph orchestration for the code assistant.

Flow: block_rule_check -> context_compress -> plan_split -> (subtask_exec) -> llm_infer
      -> tool_exec -> context_compress (loop) | memory_harvest -> END
"""

from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from agents.code_assistant import model as model_module
from agents.code_assistant.context_compress import context_compress_node
from agents.code_assistant.executor import execute_tool
from agents.code_assistant.memory import background_extract_memory, get_memory_store
from agents.code_assistant.plan import (
    make_subtask_exec_node,
    plan_split_node,
    route_after_plan,
)
from agents.code_assistant.security import FiveLayerSecurity
from agents.code_assistant.state import (
    AgentState,
    BlockRule,
    BlockVerdict,
    CheckResult,
    SecurityDecision,
    content_to_str,
)
from core import settings

current_date = datetime.now().strftime("%B %d, %Y")


def build_instructions(tools) -> str:
    tool_names = ", ".join(t.name for t in tools) or "（无）"
    return f"""
你是 local-copilot，一个本地代码助手，工作目录（PROJECT_ROOT）就是你正在维护的代码仓库。今日日期 {current_date}。

可用工具: {tool_names}

规则:
- 改代码前先用 ReadFile / ListDirectory / SearchFiles 看清现状，再动手；
- 写文件用 WriteFile（覆盖）或 EditFile（精确替换），不要重复阅读刚写入的内容；
- 文件路径都是相对工作目录的，工具会把访问限制在工作目录内，不要尝试越界；
- 需要跑命令、执行测试或查看 git 状态时用 RunCommand；
- 说明改动时讲清楚改了什么、为什么，尽量精简；
- 需要事实、外部资料或 API 用法时，先用 WebSearch 再回答；
- 中文回复，Markdown 排版；
- 遵守用户已经表达的持久偏好与拒绝规则。
"""


async def block_rule_check_node(state: AgentState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    updates: dict = {
        "user_id": state.get("user_id") or configurable.get("user_id") or "",
        "session_id": state.get("session_id") or configurable.get("thread_id") or "",
        "is_broken": False,
        "planning_done": False,
    }
    last = (state.get("messages") or [None])[-1]
    if not isinstance(last, HumanMessage):
        return updates
    text = content_to_str(last.content)
    if not text:
        return updates
    store = get_memory_store()
    judge = _semantic_judge(config) if settings.BLOCK_RULE_SEMANTIC_CHECK else None
    rule = await store.check_block_rules(updates["user_id"], text, judge=judge)
    if rule is not None:
        updates["is_broken"] = True
        updates["messages"] = [AIMessage(content=f"该请求已被持久规则拦截：{rule.semantic_desc}")]
    return updates


def _semantic_judge(config: RunnableConfig):
    async def judge(text: str, rules: list[BlockRule]) -> BlockRule | None:
        m = model_module.resolve_model(config)
        runnable = m.with_structured_output(BlockVerdict, method="json_mode").with_config(tags=["skip_stream"])
        prompt = SystemMessage(
            content=(
                "判断用户输入是否命中以下持久拒绝规则（语义匹配，不要求字面一致）：\n"
                f"{[r.model_dump() for r in rules]}\n"
                f"用户输入: {text}\n"
                '只输出 JSON: {"blocked": true/false, "rule_index": 命中规则在列表中的下标（从 0 开始）或 null}'
            )
        )
        verdict: BlockVerdict = await runnable.ainvoke([prompt], config)
        if verdict.blocked and verdict.rule_index is not None:
            index = int(verdict.rule_index)
            if 0 <= index < len(rules):
                return rules[index]
        return None

    return judge


def route_after_block_check(state: AgentState) -> Literal["context_compress", "end"]:
    return "end" if state.get("is_broken") else "context_compress"


def make_llm_infer_node(system_prompt: str, tools):
    async def llm_infer_node(state: AgentState, config: RunnableConfig) -> dict:
        m = model_module.resolve_model(config)
        bound = m.bind_tools(tools) if tools else m
        prompt = SystemMessage(content=system_prompt)
        summary = state.get("conversation_summary")
        if summary:
            prompt = SystemMessage(content=f"{system_prompt}\n\n历史会话摘要：\n{summary}")
        response = await bound.ainvoke([prompt, *state["messages"]], config)
        if state.get("remaining_steps", 100) < 2 and getattr(response, "tool_calls", None):
            return {
                "messages": [
                    AIMessage(id=response.id, content="抱歉，处理该请求需要的步骤过多，已停止。")
                ]
            }
        return {"messages": [response]}

    return llm_infer_node


def route_after_llm(state: AgentState) -> Literal["tool_exec", "memory_harvest"]:
    last = (state.get("messages") or [None])[-1]
    if state.get("is_broken"):
        return "memory_harvest"
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_exec"
    return "memory_harvest"


def make_tool_exec_node(tools):
    tool_map = {t.name: t for t in tools}

    async def tool_exec_node(state: AgentState, config: RunnableConfig) -> dict:
        last = (state.get("messages") or [None])[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}
        calls = last.tool_calls
        track = list(state.get("tool_call_track") or [])
        seen_ids = {entry["tool_call_id"] for entry in track}
        security = FiveLayerSecurity()
        user_id = state.get("user_id", "")

        decisions: dict[str, SecurityDecision] = {}
        pending_approval = False
        for tc in calls:
            if tc.get("id") in seen_ids:
                decisions[tc["id"]] = SecurityDecision(
                    result=CheckResult.REJECT, reason=f"重复的工具调用 ID: {tc['id']}"
                )
                continue
            decisions[tc["id"]] = await security.check_tool_call(
                user_id=user_id, tool_name=tc["name"], tool_args=tc.get("args") or {}
            )
            if decisions[tc["id"]].requires_approval:
                pending_approval = True

        approvals: dict[str, str] = {}
        if pending_approval:
            names = [c["name"] for c in calls if decisions[c["id"]].requires_approval]
            verdict = interrupt(f"以下高危工具调用需要人工确认: {names}。回复 yes 确认，no 拒绝。")
            approvals = {c["id"]: verdict for c in calls}

        results: list = []
        track_updates: list[dict] = []
        is_broken = bool(state.get("is_broken"))
        for tc in calls:
            decision = decisions[tc["id"]]
            approved = approvals.get(tc["id"])
            if decision.result is CheckResult.REJECT and not (
                decision.requires_approval and approved == "yes"
            ):
                results.append(
                    ToolMessage(content=f"安全网关拦截: {decision.reason}", tool_call_id=tc["id"])
                )
                track_updates.append(
                    {"tool_call_id": tc["id"], "name": tc["name"], "status": "rejected"}
                )
                is_broken = True
                continue
            tool = tool_map.get(tc["name"])
            if tool is None:
                results.append(
                    ToolMessage(content=f"未知工具: {tc['name']}", tool_call_id=tc["id"])
                )
                track_updates.append(
                    {"tool_call_id": tc["id"], "name": tc["name"], "status": "rejected"}
                )
                is_broken = True
                continue
            output = await execute_tool(tool, tc.get("args") or {})
            results.append(
                ToolMessage(content=output, tool_call_id=tc["id"])
            )
            track_updates.append(
                {"tool_call_id": tc["id"], "name": tc["name"], "status": "done"}
            )

        return {
            "messages": results,
            "tool_call_track": track + track_updates,
            "is_broken": is_broken,
        }

    return tool_exec_node


def route_after_tool(state: AgentState) -> Literal["context_compress", "memory_harvest"]:
    return "memory_harvest" if state.get("is_broken") else "context_compress"


async def memory_harvest_node(state: AgentState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    if not configurable.get("harvest_memory", True):
        return {}
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    if not user_id or not session_id or not state.get("messages"):
        return {}
    m = model_module.resolve_model(config)
    await background_extract_memory(m, state["messages"], session_id, user_id)
    return {}


def build_graph(tools, *, system_prompt: str | None = None) -> CompiledStateGraph:
    prompt = system_prompt or build_instructions(tools)
    graph = StateGraph(AgentState)
    graph.add_node("block_rule_check", block_rule_check_node)
    graph.add_node("context_compress", context_compress_node)
    graph.add_node("plan_split", plan_split_node)
    graph.add_node("subtask_exec", make_subtask_exec_node(tools))
    graph.add_node("llm_infer", make_llm_infer_node(prompt, tools))
    graph.add_node("tool_exec", make_tool_exec_node(tools))
    graph.add_node("memory_harvest", memory_harvest_node)

    graph.set_entry_point("block_rule_check")
    graph.add_conditional_edges(
        "block_rule_check",
        route_after_block_check,
        {"context_compress": "context_compress", "end": END},
    )
    graph.add_edge("context_compress", "plan_split")
    graph.add_conditional_edges(
        "plan_split", route_after_plan, {"subtask_exec": "subtask_exec", "llm_infer": "llm_infer"}
    )
    graph.add_edge("subtask_exec", "llm_infer")
    graph.add_conditional_edges(
        "llm_infer", route_after_llm, {"tool_exec": "tool_exec", "memory_harvest": "memory_harvest"}
    )
    graph.add_conditional_edges(
        "tool_exec",
        route_after_tool,
        {"context_compress": "context_compress", "memory_harvest": "memory_harvest"},
    )
    graph.add_edge("memory_harvest", END)
    return graph.compile()
