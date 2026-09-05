"""MCP tool discovery for the assistant.

`MCP_SERVERS_JSON` configures one or more MCP servers (streamable-http or stdio),
e.g. {"github": {"transport": "streamable_http", "url": "https://...", "headers": {...}}}
or {"filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}}.
The returned client must be kept alive by the caller so tool sessions stay open.
"""

import json
import logging

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from core import settings

logger = logging.getLogger(__name__)


def parse_mcp_connections() -> dict:
    raw = settings.MCP_SERVERS_JSON
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("MCP_SERVERS_JSON is not valid JSON; MCP tools disabled")
        return {}
    if not isinstance(value, dict) or not value:
        logger.warning("MCP_SERVERS_JSON must be a non-empty JSON object; MCP tools disabled")
        return {}
    return value


async def load_mcp_tools() -> tuple[list[BaseTool], MultiServerMCPClient | None]:
    connections = parse_mcp_connections()
    if not connections:
        return [], None
    client = MultiServerMCPClient(connections)
    try:
        tools = await client.get_tools()
        logger.info("loaded %d MCP tools from servers: %s", len(tools), ", ".join(connections))
        return tools, client
    except Exception:
        logger.exception("failed to load MCP tools")
        return [], None
