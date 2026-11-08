# SPDX-FileCopyrightText: 2025-present Agenta <hello@buildagenta.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Describe the MCP result envelope returned by ADK."""

from __future__ import annotations

from typing import Any

from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_tool import McpTool


def response_schema(tool: BaseTool) -> dict[str, Any] | None:
    """Describe an MCP result and type its structured content from the tool's output schema."""
    if not isinstance(tool, McpTool) or not (output := tool.raw_mcp_tool.outputSchema):
        return None
    return {
        **{key: output[key] for key in ("$defs", "definitions") if key in output},
        "type": "object",
        "properties": {
            "content": {"type": "array", "items": {}},
            "isError": {"type": "boolean"},
            "structuredContent": {"anyOf": [output, {"type": "null"}]},
        },
        "required": ["content", "isError"],
    }


__all__ = ["response_schema"]
