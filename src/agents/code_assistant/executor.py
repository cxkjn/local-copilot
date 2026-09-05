"""Tool execution wrapper for the code assistant.

Path-style arguments are confined to the workspace before invocation (defense-in-depth
on top of the built-in tools' own confinement), and failures are normalized into
tool-friendly strings so a bad tool call degrades gracefully instead of crashing the
conversation.
"""

import logging

from langchain_core.tools import BaseTool

from agents.code_assistant.sandbox import PathEscapesWorkspace, resolve_workspace_path

logger = logging.getLogger(__name__)

_PATH_ARG_HINTS = (
    "path",
    "file",
    "dir",
    "directory",
    "cwd",
    "filename",
    "source",
    "destination",
    "src",
    "dst",
    "target",
)
_PATH_TOOL_NAMES = {
    "ReadFile",
    "WriteFile",
    "EditFile",
    "DeleteFile",
    "ListDirectory",
    "MoveFile",
    "CopyFile",
    "SearchFiles",
    "ShellExec",
    "RunCommand",
}


def _is_remote(value: str) -> bool:
    return value.startswith(("http://", "https://", "ftp://", "s3://", "gs://"))


def _resolve_path_args(tool: BaseTool, args: dict) -> dict:
    if tool.name not in _PATH_TOOL_NAMES:
        return dict(args)
    resolved = dict(args)
    for key, value in args.items():
        if (
            isinstance(value, str)
            and not _is_remote(value)
            and any(hint in key.lower() for hint in _PATH_ARG_HINTS)
        ):
            resolved[key] = str(resolve_workspace_path(value))
    return resolved


async def execute_tool(tool: BaseTool, args: dict) -> str:
    try:
        safe_args = _resolve_path_args(tool, args)
        result = await tool.ainvoke(safe_args)
    except PathEscapesWorkspace as e:
        return f"路径越界拦截: {e}"
    except Exception as e:
        logger.warning("tool %s failed: %s", tool.name, e)
        return f"工具执行失败: {type(e).__name__}: {e}"
    return str(result)
