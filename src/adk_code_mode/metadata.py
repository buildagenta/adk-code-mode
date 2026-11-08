# SPDX-FileCopyrightText: 2025-present Agenta <hello@buildagenta.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Assemble the ``<code-mode>`` block appended to the model's system instruction.

The block is the sandbox's *inventory* — which Python, which preinstalled
packages, which importable functions. The invariant contract of calling
``execute_code`` lives in the tool's description instead, so the two never
restate each other.

The Python version and package list are reported by the sandbox itself in its
``ReadyFrame``, which only arrives once a container has booted — i.e. after the
first ``execute_code`` call. They are cached here, keyed by the backend's
``identity`` (its URL or image) rather than on ``ExecuteCodeTool``, so one warm
process serves every agent pointing at the same image and the value survives a
tool being rebuilt. Recording is monotonic and rendering is sorted: the block
must be byte-stable once known, because it sits in front of the whole prompt
and any churn re-writes the model provider's cached prefix.

Tags are indented; their text content is not. The ``<tools-package>`` body is
Python source, where leading whitespace is meaningful.
"""

from __future__ import annotations

from typing import Any

from adk_code_mode.runtime.protocol import ReadyFrame
from adk_code_mode.tools.catalog import Detail, render_catalog
from adk_code_mode.tools.namespacing import NamespacedTool

_OPEN = "<code-mode>"
_CLOSE = "</code-mode>"
_INDENT = "  "

HOW_TO_USE = (
    "This section describes the sandbox that `execute_code` runs in: the Python "
    "version available, the third-party packages preinstalled in it, and the "
    "functions you can import from the `tools` package."
)

# Every tier without signatures needs these: the full tier shows them inline.
READ_BEFORE_CALLING = (
    "Before calling a function you haven't used, print its stub to see its "
    "signature, docstring and the fields of its return type: "
    "`/tools/<module>/<fn>.py`, or `/tools/<fn>.py` for a function imported from "
    "`tools`. Print every stub you need in one `execute_code` call."
)

DISCOVER_TOOLS = (
    f"List `/tools/` and its module directories to find the functions available. "
    f"{READ_BEFORE_CALLING}"
)

_TOOLS_TOO_LARGE = "The full catalog is too large to include here."

# identity -> (python version, {import name: {distribution name: version}})
_ENVIRONMENTS: dict[str, tuple[str, dict[str, dict[str, str]]]] = {}


def backend_identity(backend: Any) -> str:
    """Return the cache key for a backend's sandbox image.

    Built-in backends expose ``identity`` (URL or image tag). Third-party
    backends without one collapse to their class name, which is still stable
    within a process.
    """
    return getattr(backend, "identity", None) or type(backend).__name__


def record(identity: str, frame: ReadyFrame) -> None:
    """Cache the environment a sandbox reported on connect.

    A frame carrying neither field is ignored rather than stored: sandboxes
    older than this field, and any future reconnect that fails to report, must
    not blank out an environment we already know.
    """
    if not frame.python_version and not frame.packages:
        return
    packages = {
        import_name: dict(distributions) for import_name, distributions in frame.packages.items()
    }
    _ENVIRONMENTS[identity] = (frame.python_version, packages)


def reset() -> None:
    """Drop every cached environment. For tests."""
    _ENVIRONMENTS.clear()


def _tag(name: str, content: str) -> str:
    # The catalog's sections end in a newline; strip so every tag closes flush
    # against its content rather than after a stray blank line.
    body = content.strip("\n")
    return f"{_INDENT}<{name}>\n{body}\n{_INDENT}</{name}>"


def _inline_tag(name: str, value: str) -> str:
    return f"{_INDENT}<{name}>{value}</{name}>"


def _wrap(children: list[str]) -> str:
    return "\n".join([_OPEN, *children, _CLOSE])


def render(*, identity: str, namespaced: list[NamespacedTool], max_chars: int) -> str:
    """Render the ``<code-mode>`` block, degrading until it fits ``max_chars``.

    ``max_chars`` is measured against the final string, tags included. Tiers,
    in order: the full catalog; import lines only (no types, signatures or
    docstrings), with ``READ_BEFORE_CALLING`` added to the how-to-use text; a
    pointer, with ``DISCOVER_TOOLS`` added instead, telling the model to read
    ``/tools/`` itself. The
    last tier is unconditional — an oversized tool surface must never cost the
    model the Python version and package list, which are small and which it
    cannot discover any other way.

    Tags whose content is unknown (no ``ReadyFrame`` yet) or empty (a base
    image with no extra packages, a tool with no tools) are omitted entirely
    rather than rendered blank, which would read as "none available".
    """
    python_version, packages = _ENVIRONMENTS.get(identity, ("", {}))

    environment: list[str] = []
    if python_version:
        environment.append(_inline_tag("python-version", python_version))
    if packages:
        lines = []
        for import_name, distributions in sorted(
            packages.items(),
            key=lambda item: (item[0].casefold(), item[0]),
        ):
            providers = ", ".join(
                f"{name} {version}"
                for name, version in sorted(
                    distributions.items(),
                    key=lambda item: (item[0].casefold(), item[0]),
                )
            )
            if providers:
                lines.append(f"{import_name}: {providers}")
        if lines:
            environment.append(_tag("installed-packages", "\n".join(lines)))

    how_to_use = _tag("how-to-use", HOW_TO_USE)
    if not namespaced:
        return _wrap([how_to_use, *environment])

    tiers: list[tuple[Detail, str]] = [
        ("full", how_to_use),
        ("names", _tag("how-to-use", f"{HOW_TO_USE} {READ_BEFORE_CALLING}")),
    ]
    for detail, tier_how_to_use in tiers:
        catalog = _tag("tools-package", render_catalog(namespaced, detail=detail))
        block = _wrap([tier_how_to_use, *environment, catalog])
        if len(block) <= max_chars:
            return block
    discover = _tag("how-to-use", f"{HOW_TO_USE} {DISCOVER_TOOLS}")
    return _wrap([discover, *environment, _tag("tools-package", _TOOLS_TOO_LARGE)])


__all__ = [
    "DISCOVER_TOOLS",
    "HOW_TO_USE",
    "READ_BEFORE_CALLING",
    "backend_identity",
    "record",
    "render",
    "reset",
]
