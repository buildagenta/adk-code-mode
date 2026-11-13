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
``load_artifact``. By default loaded artifacts return a flat ``{"kind",
"data", "mime_type"}`` dict so the model dispatches on ``kind`` without
importing host-side types. ``ExecuteCodeTool(load_artifact_format="parsed")``
swaps in a ``load_artifact`` that returns the content itself: parsed JSON, a
``str`` for other text, or ``{"mime_type", "bytes"}`` for binary.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Literal

from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

_TEXT_MIME_PREFIXES = ("text/",)
_TEXT_MIME_TYPES = {"application/json", "application/xml", "application/javascript"}

LoadArtifactFormat = Literal["envelope", "parsed"]
"""What ``load_artifact`` returns.

``"envelope"`` (the default) is ``{"kind", "data", "mime_type"}``. ``"parsed"``
is the content itself: parsed JSON, a ``str`` for other text, or
``{"mime_type", "bytes"}`` for binary.
"""


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
) -> dict[str, Any] | None:
    """Load an artifact by filename. Returns ``None`` if not found.

    Returned dict shape:

        {
            "kind": "text" | "bytes",
            "data": <str>,                # text: the string; bytes: base64-encoded
            "mime_type": <str | None>,
        }

    Example:

        result = load_artifact(filename="report.json")
        if result is None:
            return
        if result["kind"] == "text" and result["mime_type"] == "application/json":
            payload = json.loads(result["data"])
        elif result["kind"] == "bytes":
            blob = base64.b64decode(result["data"])
    """
    part = await tool_context.load_artifact(filename, version=version)
    if part is None:
        return None
    inline = part.inline_data
    if inline is not None:
        mime_type = inline.mime_type
        data = _inline_bytes(inline)
        if _is_text_mime(mime_type):
            return {
                "kind": "text",
                "data": data.decode(_charset(mime_type)),
                "mime_type": mime_type,
            }
        return {
            "kind": "bytes",
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": mime_type,
        }
    if part.text is not None:
        return {"kind": "text", "data": part.text, "mime_type": None}
    return {"kind": "text", "data": "", "mime_type": None}


async def load_artifact_parsed(
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
    data = _inline_bytes(inline)
    if not _is_text_mime(mime_type):
        return {"mime_type": mime_type, "bytes": base64.b64encode(data).decode("ascii")}
    text = data.decode(_charset(mime_type))
    if _is_json_mime(mime_type):
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


# Code imports it as ``load_artifact`` whichever format the host picked.
load_artifact_parsed.__name__ = load_artifact_parsed.__qualname__ = "load_artifact"


def _inline_bytes(inline: genai_types.Blob) -> bytes:
    data = inline.data or b""
    return data.encode("utf-8") if isinstance(data, str) else data


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

_PARSED_ARTIFACT_TOOLS: tuple[FunctionTool, ...] = (
    ARTIFACT_TOOLS[0],
    FunctionTool(load_artifact_parsed),
    ARTIFACT_TOOLS[2],
)


def artifact_tools(load_format: LoadArtifactFormat = "envelope") -> tuple[FunctionTool, ...]:
    """Return the artifact tools, with ``load_artifact`` returning ``load_format``."""
    if load_format == "envelope":
        return ARTIFACT_TOOLS
    if load_format == "parsed":
        return _PARSED_ARTIFACT_TOOLS
    raise ValueError(f"load_artifact_format must be 'envelope' or 'parsed', not {load_format!r}")


__all__ = [
    "ARTIFACT_TOOLS",
    "LoadArtifactFormat",
    "artifact_tools",
    "list_artifacts",
    "load_artifact",
    "load_artifact_parsed",
    "save_artifact",
]
