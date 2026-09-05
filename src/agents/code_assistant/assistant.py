"""Code assistant entry point: lazy-loads MCP tools, then builds the graph."""

import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.code_assistant.graph import build_graph
from agents.code_assistant.mcp_tools import load_mcp_tools
from agents.code_assistant.tools import BASE_TOOLS
from agents.lazy_agent import LazyLoadingAgent

logger = logging.getLogger(__name__)


class CodeAssistant(LazyLoadingAgent):
    """The code assistant, with async MCP tool loading."""

    def __init__(self) -> None:
        super().__init__()
        self._mcp_tools: list[BaseTool] = []
        self._mcp_client: MultiServerMCPClient | None = None

    async def load(self) -> None:
        self._mcp_tools, self._mcp_client = await load_mcp_tools()
        self._graph = build_graph(BASE_TOOLS + self._mcp_tools)
        self._loaded = True


code_assistant = CodeAssistant()
