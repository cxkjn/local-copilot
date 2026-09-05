import shutil
import uuid
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agents.code_assistant import model as model_module
from agents.code_assistant.context_compress import context_compress_node
from agents.code_assistant.executor import execute_tool
from agents.code_assistant.graph import build_graph
from agents.code_assistant.memory import (
    MemoryStore,
    background_extract_memory,
    get_memory_store,
)
from agents.code_assistant.plan import _should_plan, run_subtask
from agents.code_assistant.sandbox import PathEscapesWorkspace, resolve_workspace_path
from agents.code_assistant.security import FiveLayerSecurity
from agents.code_assistant.state import (
    AgentMemory,
    BlockRule,
    BlockVerdict,
    CheckResult,
    MemoryExtraction,
)
from agents.code_assistant.tokenizer import count_messages_tokens, count_tokens
from agents.tools import calculator
from core import settings

_TMP_ROOT = Path(__file__).resolve().parent.parent.parent / ".pytest-tmp"


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.fixture
def tmp_path():
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TMP_ROOT / f"ca-test-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_memory_store():
    import agents.code_assistant.memory as memory_module

    memory_module._store = None
    yield
    memory_module._store = None


def make_config(**kwargs):
    configurable = {
        "thread_id": kwargs.pop("thread_id", "s1"),
        "user_id": kwargs.pop("user_id", "u1"),
        "harvest_memory": False,
    }
    configurable.update(kwargs)
    return {"configurable": configurable}


def patch_model(monkeypatch, messages):
    fake = ScriptedModel(messages=iter(messages))
    monkeypatch.setattr(model_module, "resolve_model", lambda config=None: fake)
    return fake


# ---------- data models ----------


def test_state_models():
    assert CheckResult.PASS.value == "pass"
    memory = AgentMemory(preference="简洁", feedback="不错", knowledge="", reference="")
    assert memory.preference == "简洁"
    rule = BlockRule(trigger_pattern="不要讲黄色笑话", semantic_desc="拒绝黄色笑话")
    assert rule.enable is True
    verdict = BlockVerdict(blocked=True, rule_index=0)
    assert verdict.blocked and verdict.rule_index == 0
    extraction = MemoryExtraction(block_intent="拒绝推销", block_keywords=["推销"])
    assert extraction.block_keywords == ["推销"]


# ---------- tokenizer ----------


def test_tokenizer():
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
    messages = [HumanMessage(content="hello"), AIMessage(content="world")]
    assert count_messages_tokens(messages) > 0


# ---------- security gateway ----------


@pytest.mark.asyncio
async def test_security_layers():
    sec = FiveLayerSecurity()
    decision = await sec.check_tool_call(user_id="u", tool_name="Calculator", tool_args={})
    assert decision.result is CheckResult.PASS

    sec = FiveLayerSecurity(tool_whitelist={"WebSearch"})
    decision = await sec.check_tool_call(user_id="u", tool_name="Calculator", tool_args={})
    assert decision.result is CheckResult.REJECT and "白名单" in decision.reason

    sec = FiveLayerSecurity(content_blocklist={"sudo"})
    decision = await sec.check_tool_call(
        user_id="u", tool_name="ShellExec", tool_args={"command": "sudo rm -rf /"}
    )
    assert decision.result is CheckResult.REJECT and "不安全内容" in decision.reason

    decision = await sec.check_tool_call(user_id="u", tool_name="WriteFile", tool_args={})
    assert decision.result is CheckResult.REJECT and decision.requires_approval

    async def permission_policy(user_id, tool_name):
        return user_id == "admin"

    sec = FiveLayerSecurity(permission_policy=permission_policy)
    decision = await sec.check_tool_call(user_id="guest", tool_name="Calculator", tool_args={})
    assert decision.result is CheckResult.REJECT and "无权" in decision.reason


# ---------- workspace path confinement ----------


def test_resolve_workspace_path(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PROJECT_ROOT", str(tmp_path))
    assert resolve_workspace_path("a/b.txt") == (tmp_path / "a/b.txt").resolve()
    with pytest.raises(PathEscapesWorkspace):
        resolve_workspace_path("../escape.txt")


# ---------- executor ----------


@pytest.mark.asyncio
async def test_executor_path_containment(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PROJECT_ROOT", str(tmp_path))

    @tool
    def WriteFile(path: str, content: str) -> str:
        """Write content to a file."""
        Path(path).write_text(content, encoding="utf-8")
        return "ok"

    result = await execute_tool(WriteFile, {"path": "notes.txt", "content": "hi"})
    assert result == "ok"
    assert (tmp_path / "notes.txt").exists()

    result = await execute_tool(WriteFile, {"path": "../../escape.txt", "content": "x"})
    assert "路径越界拦截" in result


# ---------- memory store ----------


@pytest.mark.asyncio
async def test_memory_store_in_memory():
    store = MemoryStore(redis_url=None)
    await store.aset_memory("s1", AgentMemory(preference="简洁"))
    memory = await store.aget_memory("s1")
    assert memory.preference == "简洁"

    rule = BlockRule(
        trigger_pattern="不要讲黄色笑话", semantic_desc="拒绝黄色笑话", keywords=["黄色笑话"]
    )
    await store.aadd_block_rule("u1", rule)
    assert len(await store.aget_block_rules("u1")) == 1

    hit = await store.check_block_rules("u1", "以后别讲黄色笑话了")
    assert hit is rule
    assert await store.check_block_rules("u1", "今天天气不错") is None

    async def judge(text, candidates):
        return None

    assert await store.check_block_rules("u1", "以后别讲黄色笑话了", judge=judge) is None


class StubStructuredModel:
    def __init__(self, value):
        self._value = value

    def with_structured_output(self, schema, **kwargs):
        return self

    def with_config(self, **kwargs):
        return self

    async def ainvoke(self, *args, **kwargs):
        return self._value


@pytest.mark.asyncio
async def test_background_extract_memory():
    store = MemoryStore(redis_url=None)
    model = StubStructuredModel(
        MemoryExtraction(
            preference="中文回复", block_intent="不要讲黄色笑话", block_keywords=["黄色笑话"]
        )
    )
    await background_extract_memory(model, [HumanMessage(content="hi")], "s1", "u1", store)
    memory = await store.aget_memory("s1")
    assert memory.preference == "中文回复"
    rules = await store.aget_block_rules("u1")
    assert rules and "黄色笑话" in rules[0].trigger_pattern

    await background_extract_memory(
        StubStructuredModel(MemoryExtraction()), [HumanMessage(content="hi")], "s2", "u2", store
    )
    assert await store.aget_block_rules("u2") == []


# ---------- context compression ----------


@pytest.mark.asyncio
async def test_context_compress_node(monkeypatch):
    import agents.code_assistant.context_compress as cc

    async def fake_summarizer(model, history, config, previous_summary=""):
        return "历史摘要"

    monkeypatch.setattr(cc, "count_messages_tokens", lambda messages, model=None: 999999)
    monkeypatch.setattr(cc, "summarize_history", fake_summarizer)
    monkeypatch.setattr(settings, "CONTEXT_KEEP_RECENT_ROUNDS", 1)

    messages = [
        HumanMessage(content="q1", id="m1"),
        AIMessage(content="a1", id="m2"),
        HumanMessage(content="q2", id="m3"),
        AIMessage(content="a2", id="m4"),
    ]
    state = {"messages": messages, "user_id": "u1", "session_id": "s1"}
    result = await context_compress_node(state, make_config())
    assert result["conversation_summary"] == "历史摘要"
    assert [m.id for m in result["messages"]] == ["m1", "m2"]


@pytest.mark.asyncio
async def test_context_compress_node_under_limit():
    messages = [HumanMessage(content="q1"), AIMessage(content="a1")]
    result = await context_compress_node({"messages": messages}, make_config())
    assert result == {}


# ---------- plan mode ----------


def test_should_plan():
    assert _should_plan("x" * 600, {"configurable": {}}) is True
    assert _should_plan("short", {"configurable": {}}) is False
    assert _should_plan("short", {"configurable": {"plan_mode": True}}) is True


@pytest.mark.asyncio
async def test_run_subtask():
    fake = ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "Calculator",
                            "args": {"expression": "2*21"},
                            "id": "sub-call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="答案是 42。"),
            ]
        )
    )
    result = await run_subtask(
        "计算 2*21",
        0,
        {"user_id": "u1", "messages": []},
        {"configurable": {}},
        [calculator],
        fake,
        FiveLayerSecurity(),
    )
    assert result["output"] == "答案是 42。"


@pytest.mark.asyncio
async def test_plan_split_node(monkeypatch):
    from agents.code_assistant.plan import plan_split_node
    from agents.code_assistant.state import PlanOutput

    def resolve(config=None):
        return StubStructuredModel(PlanOutput(subtasks=["子任务A", "子任务B"]))

    monkeypatch.setattr(model_module, "resolve_model", resolve)
    state = {"messages": [HumanMessage(content="x" * 600)], "user_id": "u1", "session_id": "s1"}
    result = await plan_split_node(state, {"configurable": {}})
    assert result["subtasks"] == ["子任务A", "子任务B"]
    assert result["planning_done"] is True
    second = await plan_split_node({**state, "planning_done": True}, {"configurable": {}})
    assert second == {}


@pytest.mark.asyncio
async def test_plan_split_node_short_input_no_plan(monkeypatch):
    from agents.code_assistant.plan import plan_split_node

    monkeypatch.setattr(model_module, "resolve_model", lambda config=None: object())
    state = {"messages": [HumanMessage(content="hi")], "user_id": "u1", "session_id": "s1"}
    result = await plan_split_node(state, {"configurable": {}})
    assert result == {"planning_done": True}


# ---------- graph end-to-end ----------


@pytest.mark.asyncio
async def test_graph_simple_flow(monkeypatch):
    patch_model(monkeypatch, ["你好，我是智能助手。"])
    graph = build_graph([calculator])
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="你好")], "user_id": "u1", "session_id": "s1"},
        config=make_config(),
    )
    assert isinstance(result["messages"][-1], AIMessage)
    assert "智能助手" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_graph_tool_loop(monkeypatch):
    patch_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Calculator",
                        "args": {"expression": "1+1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="计算结果是 2。"),
        ],
    )
    graph = build_graph([calculator])
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="1+1=?")], "user_id": "u1", "session_id": "s1"},
        config=make_config(),
    )
    contents = [m.content for m in result["messages"]]
    assert "2" in contents
    assert contents[-1] == "计算结果是 2。"


@pytest.mark.asyncio
async def test_graph_high_risk_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "PROJECT_ROOT", str(tmp_path))

    @tool
    def WriteFile(path: str, content: str) -> str:
        """Write content to a file."""
        Path(path).write_text(content, encoding="utf-8")
        return "written"

    patch_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "WriteFile",
                        "args": {"path": "notes.txt", "content": "hi"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="文件已写入。"),
        ],
    )
    graph = build_graph([WriteFile])
    graph.checkpointer = MemorySaver()
    config = make_config()
    await graph.ainvoke(
        {"messages": [HumanMessage(content="write a file")], "user_id": "u1", "session_id": "s1"},
        config=config,
    )
    state = await graph.aget_state(config)
    assert state.tasks and any(t.interrupts for t in state.tasks)

    result = await graph.ainvoke(Command(resume="yes"), config=config)
    contents = [m.content for m in result["messages"]]
    assert "written" in contents
    assert contents[-1] == "文件已写入。"


@pytest.mark.asyncio
async def test_graph_declined_approval(monkeypatch):
    @tool
    def DeleteFile(path: str) -> str:
        """Delete a file."""
        return "deleted"

    patch_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "DeleteFile",
                        "args": {"path": "x.txt"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="不应执行"),
        ],
    )
    graph = build_graph([DeleteFile])
    graph.checkpointer = MemorySaver()
    config = make_config()
    await graph.ainvoke(
        {"messages": [HumanMessage(content="delete a file")], "user_id": "u1", "session_id": "s1"},
        config=config,
    )
    result = await graph.ainvoke(Command(resume="no"), config=config)
    contents = [m.content for m in result["messages"]]
    assert any("安全网关拦截" in c for c in contents)
    assert "不应执行" not in contents


@pytest.mark.asyncio
async def test_graph_duplicate_tool_call_id(monkeypatch):
    patch_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Calculator",
                        "args": {"expression": "1+1"},
                        "id": "dup-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
    )
    graph = build_graph([calculator])
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="calc")],
            "user_id": "u1",
            "session_id": "s1",
            "tool_call_track": [{"tool_call_id": "dup-1", "name": "Calculator", "status": "done"}],
        },
        config=make_config(),
    )
    contents = [m.content for m in result["messages"]]
    assert any("重复的工具调用 ID" in c for c in contents)


@pytest.mark.asyncio
async def test_graph_block_rule_intercept(monkeypatch):
    monkeypatch.setattr(settings, "BLOCK_RULE_SEMANTIC_CHECK", False)
    store = get_memory_store()
    await store.aadd_block_rule(
        "u-block",
        BlockRule(
            trigger_pattern="不要讲黄色笑话",
            semantic_desc="用户拒绝黄色笑话",
            keywords=["黄色笑话"],
        ),
    )
    patch_model(monkeypatch, ["不应到达这里"])
    graph = build_graph([calculator])
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="以后不要讲黄色笑话了")],
            "user_id": "u-block",
            "session_id": "s1",
        },
        config=make_config(user_id="u-block"),
    )
    last = result["messages"][-1]
    assert isinstance(last, AIMessage) and "拦截" in last.content


@pytest.mark.asyncio
async def test_graph_semantic_block_judge(monkeypatch):
    def resolve(config=None):
        return StubStructuredModel(BlockVerdict(blocked=True, rule_index=0))

    monkeypatch.setattr(model_module, "resolve_model", resolve)
    store = get_memory_store()
    await store.aadd_block_rule(
        "u-judge",
        BlockRule(trigger_pattern="拒绝推销电话", semantic_desc="用户拒绝推销", keywords=["推销"]),
    )
    graph = build_graph([calculator])
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="别再给我打推销电话了")],
            "user_id": "u-judge",
            "session_id": "s1",
        },
        config=make_config(user_id="u-judge"),
    )
    assert "拦截" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_graph_context_compress(monkeypatch):
    import agents.code_assistant.context_compress as cc

    async def fake_summarizer(model, history, config, previous_summary=""):
        return "历史摘要"

    monkeypatch.setattr(cc, "count_messages_tokens", lambda messages, model=None: 999999)
    monkeypatch.setattr(cc, "summarize_history", fake_summarizer)
    monkeypatch.setattr(settings, "CONTEXT_KEEP_RECENT_ROUNDS", 1)
    patch_model(monkeypatch, ["最终回答"])

    graph = build_graph([])
    history = [
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
    ]
    result = await graph.ainvoke(
        {"messages": history, "user_id": "u1", "session_id": "s1"}, config=make_config()
    )
    assert result["conversation_summary"] == "历史摘要"
    assert [m.content for m in result["messages"]] == ["q2", "a2", "最终回答"]


# ---------- registry ----------


def test_agent_registered():
    from agents.agents import agents

    assert "code-assistant" in agents
