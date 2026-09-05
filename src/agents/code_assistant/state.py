"""Core data structures for the code assistant."""

from enum import StrEnum
from typing import Any

from langgraph.graph import MessagesState
from langgraph.managed import RemainingSteps
from pydantic import BaseModel, Field


class CheckResult(StrEnum):
    PASS = "pass"
    REJECT = "reject"


class SecurityDecision(BaseModel):
    result: CheckResult
    reason: str = ""
    requires_approval: bool = False


class BlockRule(BaseModel):
    trigger_pattern: str = Field(description="LLM 提取的触发语义")
    semantic_desc: str = Field(description="规则描述")
    keywords: list[str] = Field(default_factory=list, description="快速预筛关键词")
    enable: bool = True


class AgentMemory(BaseModel):
    preference: str = ""
    feedback: str = ""
    knowledge: str = ""
    reference: str = ""


class MemoryExtraction(BaseModel):
    preference: str = ""
    feedback: str = ""
    knowledge: str = ""
    reference: str = ""
    block_intent: str = ""
    block_keywords: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    subtasks: list[str] = Field(default_factory=list)
    reasoning: str = ""


class BlockVerdict(BaseModel):
    blocked: bool = False
    rule_index: int | None = None


class AgentState(MessagesState, total=False):
    session_id: str
    user_id: str
    tool_call_track: list[dict[str, Any]]
    is_broken: bool
    subtasks: list[str]
    plan_results: list[dict[str, str]]
    planning_done: bool
    conversation_summary: str
    remaining_steps: RemainingSteps


def content_to_str(content: str | list[Any]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)
