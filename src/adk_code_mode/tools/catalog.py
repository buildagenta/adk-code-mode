# SPDX-FileCopyrightText: 2025-present Agenta <hello@buildagenta.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Render the tool catalog string injected into the model's system prompt.

The catalog is grouped by module: ``# tools.<namespace>`` sections (sorted),
then ``# tools`` for any top-level tools. Each section opens with an import
line, then the ``TypedDict`` classes its functions return, then ``.pyi``-style
function definitions (signature + docstring + ``...`` body).
``adk_code_mode.metadata`` wraps the result in a ``<tools-package>`` tag inside
the ``<code-mode>`` block; this module returns just the inner content.
"""

from __future__ import annotations

from typing import Literal

from adk_code_mode.tools.namespacing import NamespacedTool
from adk_code_mode.tools.stubs import RenderedTool, render_tool

Detail = Literal["full", "names"]


def _render_entry(rt: RenderedTool) -> str:
    """One ``.pyi``-style entry: signature + docstring + ``...`` placeholder."""
    return f"{rt.signature_for('catalog')}\n    {rt.docstring}\n    ...\n"


def _render_module_section(module: str, tools: list[NamespacedTool], detail: Detail) -> str:
    rendered = [render_tool(t) for t in sorted(tools, key=lambda t: t.attribute)]
    import_line = f"from {module} import {', '.join(rt.attribute for rt in rendered)}"
    # At ``names`` detail the import line, which already lists every function
    # in the module, is the whole section.
    if detail == "names":
        return f"# {module}\n\n{import_line}\n"
    types = dict.fromkeys(source for rt in rendered for source in rt.types)
    blocks = [*(f"{source}\n" for source in types), *(_render_entry(rt) for rt in rendered)]
    return f"# {module}\n\n{import_line}\n\n" + "\n".join(blocks)


def render_catalog(namespaced: list[NamespacedTool], *, detail: Detail = "full") -> str:
    """Render the tool catalog (without the ``<tools-package>`` wrapper).

    Namespaced sections come first in namespace-sorted order, then top-level
    tools. Tools within a section are sorted by attribute name for
    determinism. A section defines the ``TypedDict`` return types of all its
    tools once, ahead of their signatures.

    ``detail="names"`` drops every type, signature and docstring, leaving only
    the per-module import lines. It is the fallback when the full catalog
    doesn't fit the system-instruction budget.
    """
    grouped: dict[str | None, list[NamespacedTool]] = {}
    for nt in namespaced:
        grouped.setdefault(nt.namespace, []).append(nt)

    sections = [
        _render_module_section(f"tools.{ns}", grouped[ns], detail)
        for ns in sorted(n for n in grouped if n is not None)
    ]
    if None in grouped:
        sections.append(_render_module_section("tools", grouped[None], detail))
    return "\n".join(sections)


__all__ = [
    "Detail",
    "render_catalog",
]
