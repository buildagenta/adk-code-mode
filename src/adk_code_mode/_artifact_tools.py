# SPDX-FileCopyrightText: 2025-present Agenta <hello@buildagenta.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Built-in ``save_artifact`` / ``load_artifact`` / ``list_artifacts`` tools.

These are regular ``FunctionTool`` instances injected at the front of
``ExecuteCodeTool``'s configured tools (set ``include_artifact_tools=False`` to opt out).
They appear in the rendered catalog as top-level tools; the model writes
``from tools import save_artifact, load_artifact, list_artifacts``.

The wire format is JSON. Binary content is base64 over the wire — the model
encodes it before calling ``save_artifact`` and decodes it after
``load_artifact``. ``load_artifact`` returns the content itself: parsed JSON,
a ``str`` for other text, or ``{"mime_type", "bytes"}`` for binary.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_TYPES = {"application/json", "application/xml", "application/javascript"}


def _is_text_mime(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    mime_type = _base_mime(mime_type)
    if mime_type.startswith(_TEXT_MIME_PREFIXES):
        return True
    if mime_type in _TEXT_MIME_TYPES:
        return True
    return mime_type.endswith("+json") or mime_type.endswith("+xml")


def _is_json_mime(mime_type: str | None) -> bool:
    base = _base_mime(mime_type or "")
    return base == "application/json" or base.endswith("+json")


def _base_mime(mime_type: str) -> str:
    return mime_type.split(";", 1)[0].strip().lower()


def _charset(mime_type: str | None) -> str:
    for param in (mime_type or "").split(";")[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset":
            return value.strip().strip('"')
    return "utf-8"


async def save_artifact(
    *,
    filename: str,
    content: str,
    mime_type: str | None = None,
    custom_metadata: dict[str, Any] | None = None,
    tool_context: ToolContext,
) -> int:
    """Save an artifact to the session. Returns the new version number.

    For text or JSON, pass ``content`` as a string and set ``mime_type``:

        save_artifact(
            filename="report.json",
            content=json.dumps({"status": "ok"}),
            mime_type="application/json",
        )

    For binary, base64-encode the bytes and set the binary mime type:

        save_artifact(
            filename="image.png",
            content=base64.b64encode(image_bytes).decode("ascii"),
            mime_type="image/png",
        )

    ``custom_metadata`` is custom metadata to associate with the artifact.
    ``load_artifact`` gives back the parsed JSON, the text, or
    ``{"mime_type", "bytes"}`` with the same base64 string.
    """
    if _is_text_mime(mime_type):
        part = genai_types.Part(
            inline_data=genai_types.Blob(
                data=content.encode(_charset(mime_type)),
                mime_type=mime_type or "text/plain",
            )
        )
    else:
        part = genai_types.Part(
            inline_data=genai_types.Blob(
                data=base64.b64decode(content),
                mime_type=mime_type or "application/octet-stream",
            )
        )
    return await tool_context.save_artifact(filename, part, custom_metadata=custom_metadata)


async def load_artifact(
    *,
    filename: str,
    version: int | None = None,
    tool_context: ToolContext,
) -> Any:
    """Load an artifact by filename and return its content. Returns ``None`` if not found.

    The return value depends on the artifact's mime type:

    - JSON (``application/json``, ``*+json``): the parsed value — for a saved
      tool result, exactly what the original call returned.
    - Other text (``text/*``, XML, JavaScript): the text as a ``str``.
    - Binary: ``{"mime_type": <str>, "bytes": <base64 str>}``.

    Example:

        rows = load_artifact(filename="search_call-1.json")  # list/dict, already parsed
        notes = load_artifact(filename="notes.txt")  # str
        image = load_artifact(filename="chart.png")
        png = base64.b64decode(image["bytes"])
    """
    part = await tool_context.load_artifact(filename, version=version)
    if part is None:
        return None
    inline = part.inline_data
    if inline is None:
        return part.text or ""
    mime_type = inline.mime_type
    data = inline.data or b""
    if isinstance(data, str):
        data = data.encode("utf-8")
    if not _is_text_mime(mime_type):
        return {"mime_type": mime_type, "bytes": base64.b64encode(data).decode("ascii")}
    text = data.decode(_charset(mime_type))
    if _is_json_mime(mime_type):
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


async def list_artifacts(*, tool_context: ToolContext) -> list[str]:
    """Return the filenames of artifacts visible to this session.

    Example:

        for filename in list_artifacts():
            print(filename)
    """
    return await tool_context.list_artifacts()


ARTIFACT_TOOLS: tuple[FunctionTool, ...] = (
    FunctionTool(save_artifact),
    FunctionTool(load_artifact),
    FunctionTool(list_artifacts),
)


__all__ = ["ARTIFACT_TOOLS", "list_artifacts", "load_artifact", "save_artifact"]
