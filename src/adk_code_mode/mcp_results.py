# SPDX-FileCopyrightText: 2025-present Agenta <hello@buildagenta.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Return an MCP tool's ``structuredContent`` to code instead of the whole result.

ADK's ``McpTool`` returns the full ``CallToolResult`` (``content``,
``structuredContent``, ``isError``). When the tool declares an ``outputSchema``,
the MCP spec requires ``structuredContent`` to match it, and ``content`` is meant to
be a serialised copy for older clients. So code gets ``structuredContent`` alone,
and the stub's return type is built from the ``outputSchema``.
"""

from __future__ import annotations

from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_tool import McpTool

from adk_code_mode.tool_result_artifacts import ToolResultArtifactTool


def output_schema(tool: BaseTool) -> dict[str, Any] | None:
    """Return the ``outputSchema`` of an MCP tool, or ``None`` for any other tool."""
    inner = tool._wrapped if isinstance(tool, ToolResultArtifactTool) else tool
    return inner.raw_mcp_tool.outputSchema if isinstance(inner, McpTool) else None


def structured_result(tool: BaseTool, result: Any) -> Any:
    """Return ``structuredContent`` when ``tool`` is an MCP tool with an ``outputSchema``."""
    if output_schema(tool) and isinstance(result, dict) and "structuredContent" in result:
        return result["structuredContent"]
    return result


__all__ = ["output_schema", "structured_result"]
