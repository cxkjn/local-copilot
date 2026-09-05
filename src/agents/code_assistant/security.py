"""Security gateway enforced before every tool call.

Layers, short-circuiting on the first REJECT:
1. user permission      - does the user have the right to use this tool
2. content safety       - tool arguments screened against the content blocklist
3. tool whitelist       - unknown / non-whitelisted tools are refused
4. high-risk approval   - destructive tools require human confirmation (HITL)
5. workspace confinement - file paths are confined to PROJECT_ROOT by the code tools
   and the executor (always on, not a runtime check here)
"""

import json
import logging
from collections.abc import Awaitable, Callable

from agents.code_assistant.state import CheckResult, SecurityDecision
from core import settings

logger = logging.getLogger(__name__)

PermissionChecker = Callable[[str, str], Awaitable[bool]]


class FiveLayerSecurity:
    def __init__(
        self,
        *,
        tool_whitelist: set[str] | None = None,
        high_risk_tools: set[str] | None = None,
        content_blocklist: set[str] | None = None,
        permission_policy: PermissionChecker | None = None,
    ) -> None:
        self.tool_whitelist = (
            set(tool_whitelist) if tool_whitelist is not None else set(settings.TOOL_WHITELIST)
        )
        self.high_risk_tools = (
            set(high_risk_tools) if high_risk_tools is not None else set(settings.HIGH_RISK_TOOLS)
        )
        self.content_blocklist = (
            set(content_blocklist)
            if content_blocklist is not None
            else set(settings.CONTENT_BLOCKLIST)
        )
        self.permission_policy = permission_policy

    async def check_tool_call(
        self,
        *,
        user_id: str,
        tool_name: str,
        tool_args: dict,
    ) -> SecurityDecision:
        if self.permission_policy is not None:
            allowed = await self.permission_policy(user_id, tool_name)
            if not allowed:
                return SecurityDecision(
                    result=CheckResult.REJECT,
                    reason=f"用户 {user_id or '(anonymous)'} 无权使用工具 {tool_name}",
                )
        if reason := self._check_content_safety(tool_args):
            return SecurityDecision(result=CheckResult.REJECT, reason=reason)
        if self.tool_whitelist and tool_name not in self.tool_whitelist:
            return SecurityDecision(
                result=CheckResult.REJECT, reason=f"工具 {tool_name} 不在白名单内"
            )
        if tool_name in self.high_risk_tools:
            return SecurityDecision(
                result=CheckResult.REJECT,
                reason=f"高危工具 {tool_name} 需要人工确认",
                requires_approval=True,
            )
        return SecurityDecision(result=CheckResult.PASS, reason="ok")

    def _check_content_safety(self, tool_args: dict) -> str | None:
        if not self.content_blocklist or not tool_args:
            return None
        blob = json.dumps(tool_args, ensure_ascii=False).lower()
        for word in self.content_blocklist:
            if word.lower() in blob:
                return f"工具参数包含不安全内容: {word}"
        return None
