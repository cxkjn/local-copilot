"""Base tools for the code assistant (MCP tools are loaded lazily on top)."""

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool

from agents.code_assistant.code_tools import CODE_TOOLS
from agents.tools import calculator

BASE_TOOLS: list[BaseTool] = [
    calculator,
    DuckDuckGoSearchResults(name="WebSearch"),
    *CODE_TOOLS,
]
