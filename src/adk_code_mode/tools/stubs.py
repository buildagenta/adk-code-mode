# SPDX-FileCopyrightText: 2025-present Agenta <hello@buildagenta.com>
#
# SPDX-License-Identifier: Apache-2.0
"""Generate Python stub modules for ADK tools.

Each stub is a real ``.py`` file the sandbox writes into ``/tools/``. The
function body is a single delegation into the host via the RPC client, so the
generated file is small and robust. Parameter types come from the tool's
JSON Schema (``BaseTool._get_declaration().parameters_json_schema``); the return
type comes from ``response_json_schema`` (an MCP tool's ``outputSchema``), with
objects rendered as ``TypedDict`` classes shared by the tools of one module.

``render_tool`` returns a structured ``RenderedTool``; ``render_tool_source``
turns one into the on-disk ``.py`` stub. The catalog renderer (in
``adk_code_mode.tools.catalog``) consumes the same ``RenderedTool`` to produce
the ``.pyi``-style block injected into the model's system prompt.
"""

from __future__ import annotations

import itertools
import json
import keyword
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Literal

from adk_code_mode.mcp_results import output_schema
from adk_code_mode.tool_result_artifacts import ToolResultArtifactTool
from adk_code_mode.tools.namespacing import NamespacedTool, PythonNameCollisionError

_PRIMITIVES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}

# Gemini `types.Schema` renames JSON Schema's keywords; map them back. A value of
# `None` drops the key (it carries nothing a caller can act on).
_KEY_ALIASES: dict[str, str | None] = {
    "max_length": "maxLength",
    "min_length": "minLength",
    "max_items": "maxItems",
    "min_items": "minItems",
    "max_properties": "maxProperties",
    "min_properties": "minProperties",
    "any_of": "anyOf",
    "additional_properties": "additionalProperties",
    "property_ordering": None,
}

# Validation and annotation keywords worth showing a caller, in the order they
# read best. These never reach the signature — no Python type expresses
# `maxLength` — so the docstring is their only route to the model.
#
# Deliberately not surfaced: `not`, `if`/`then`/`else`, `dependentSchemas`,
# `propertyNames`, `unevaluated*` and `contains` (as a schema). They express
# conditional structure that reads as noise in a one-line summary, and none of
# them appear in the OpenAPI documents this renders in practice. `$schema`,
# `$id`, `$anchor`, `$comment` and `$vocabulary` describe the document rather
# than the value.
_CONSTRAINTS = (
    "format",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minContains",
    "maxContains",
    "minProperties",
    "maxProperties",
    "contentEncoding",
    "contentMediaType",
    "const",
    "example",
)

# Boolean annotations that change whether a caller should send a value at all.
_FLAGS = (
    ("deprecated", "deprecated"),
    ("readOnly", "read-only"),
    ("writeOnly", "write-only"),
)

# Guards against self-referential schemas, which recurse forever otherwise. Not a
# size budget: the agent can always read the full stub with `help()`.
_MAX_SCHEMA_DEPTH = 8

Target = Literal["stub", "catalog"]


def _schema_dict_from_declaration(declaration: Any) -> dict[str, Any]:
    """Return a JSON-Schema-shaped dict for a tool declaration.

    ADK tools expose their schema one of two ways: ``parameters_json_schema``
    (RestApiTool, MCP tools, anything built from a real JSON Schema) or
    ``parameters`` (FunctionTool, which builds a Gemini ``types.Schema`` from
    the function signature). We normalise both to a plain dict.
    """
    if declaration is None:
        return {}
    pjs = getattr(declaration, "parameters_json_schema", None)
    if pjs:
        return dict(pjs)
    params = getattr(declaration, "parameters", None)
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        dumped = params.model_dump(mode="json", exclude_none=True)
        return dumped if isinstance(dumped, dict) else {}
    if isinstance(params, dict):
        return dict(params)
    return {}


def _effective_schema(schema: Any) -> dict[str, Any]:
    """Normalise a schema fragment to plain JSON Schema and flatten ``allOf``.

    Two shapes reach us. With ADK's ``JSON_SCHEMA_FOR_FUNC_DECL`` feature on,
    ``parameters_json_schema`` carries untouched JSON Schema. With it off (the
    default today) ADK converts through a Gemini ``types.Schema``, which
    upper-cases ``type`` and snake-cases the validation keywords — and drops the
    body of an ``allOf``-wrapped ``$ref`` entirely, which no amount of work here
    can recover.

    Merging ``allOf`` matters because ``allOf: [$ref] + nullable: true`` is how
    OpenAPI 3.0 spells "nullable reference": without the merge the wrapper looks
    like an empty schema and the target's description and properties are lost.
    """
    if not isinstance(schema, dict):
        return {}

    normalised: dict[str, Any] = {}
    for key, value in schema.items():
        canonical = _KEY_ALIASES.get(key, key)
        if canonical is None or value is None:
            continue
        normalised.setdefault(canonical, value)

    ty = normalised.get("type")
    if isinstance(ty, str):
        normalised["type"] = ty.lower()
    elif isinstance(ty, list):
        normalised["type"] = [t.lower() if isinstance(t, str) else t for t in ty]

    variants = normalised.pop("allOf", None)
    if isinstance(variants, list):
        merged: dict[str, Any] = {}
        for part in variants:
            merged = _merge_schema(merged, _effective_schema(part))
        # Wrapper keys (e.g. `nullable`) win; `properties` / `required` still merge.
        normalised = _merge_schema(merged, normalised)

    # A one-variant union expresses no choice, so fold it in rather than lose
    # the description and properties hanging off the single branch.
    for key in ("anyOf", "oneOf"):
        branches = normalised.get(key)
        if isinstance(branches, list) and len(branches) == 1:
            folded = _effective_schema(branches[0])
            normalised.pop(key)
            normalised = _merge_schema(folded, normalised)

    return normalised


def _merge_schema(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Combine two schemas the way ``allOf`` means: properties accumulate.

    A shallow ``update`` replaces the whole ``properties`` dict, so OpenAPI 3.0
    ``allOf: [$ref, {properties: extra}]`` would keep only ``extra``.
    """
    out = dict(base)
    extra_props = extra.get("properties")
    if isinstance(extra_props, dict):
        props = dict(out.get("properties") or {})
        for name, schema in extra_props.items():
            existing = props.get(name)
            if isinstance(existing, dict) and isinstance(schema, dict):
                props[name] = _merge_schema(existing, schema)
            else:
                props[name] = schema
        out["properties"] = props
    extra_required = extra.get("required")
    if isinstance(extra_required, list):
        required = [name for name in (out.get("required") or []) if isinstance(name, str)]
        for name in extra_required:
            if isinstance(name, str) and name not in required:
                required.append(name)
        out["required"] = required
    for key, value in extra.items():
        if key in ("properties", "required"):
            continue
        out[key] = value
    return out


def _schema_to_type(schema: Any) -> str:
    """Convert a JSON-Schema fragment to a Python type expression (PEP 604).

    Best-effort. Anything we can't resolve cleanly falls back to ``Any``.
    """
    schema = _effective_schema(schema)
    if not schema:
        return "Any"

    # Compositions.
    if "anyOf" in schema or "oneOf" in schema:
        variants = schema.get("anyOf") or schema.get("oneOf") or []
        parts = [_schema_to_type(v) for v in variants]
        parts = list(dict.fromkeys(parts))
        if not parts:
            return "Any"
        if len(parts) == 1:
            return parts[0]
        return " | ".join(parts)

    if "enum" in schema and schema["enum"]:
        lits: list[str] = []
        for value in schema["enum"]:
            if isinstance(value, str):
                lits.append(repr(value))
            elif isinstance(value, bool):
                lits.append("True" if value else "False")
            elif isinstance(value, (int, float)):
                lits.append(repr(value))
            elif value is None:
                lits.append("None")
            else:
                return "Any"
        return "Literal[" + ", ".join(lits) + "]"

    ty = schema.get("type")
    if isinstance(ty, str):
        ty = ty.lower()
    nullable = bool(schema.get("nullable"))
    if isinstance(ty, list):
        parts = [
            _schema_to_type(
                {
                    "type": t.lower() if isinstance(t, str) else t,
                    **{k: v for k, v in schema.items() if k != "type"},
                }
            )
            for t in ty
        ]
        parts = list(dict.fromkeys(parts))
        result = parts[0] if len(parts) == 1 else " | ".join(parts)
    elif ty == "array":
        items = schema.get("items")
        prefix_items = schema.get("prefixItems")
        if prefix_items:
            inner = ", ".join(_schema_to_type(p) for p in prefix_items)
            result = f"tuple[{inner}]"
        elif isinstance(items, dict):
            result = f"list[{_schema_to_type(items)}]"
        else:
            result = "list[Any]"
    elif ty == "object":
        # An open map (`additionalProperties` as a schema, no fixed properties)
        # types precisely; anything with declared keys stays `dict[str, Any]`
        # and gets its shape from the docstring instead.
        extra = schema.get("additionalProperties")
        if isinstance(extra, dict) and not schema.get("properties"):
            result = f"dict[str, {_schema_to_type(extra)}]"
        else:
            result = "dict[str, Any]"
    elif isinstance(ty, str) and ty in _PRIMITIVES:
        result = _PRIMITIVES[ty]
    else:
        result = "Any"

    if nullable and result != "Any":
        result = f"{result} | None"
    return result


def _python_default(schema: dict[str, Any]) -> str | None:
    """Return a Python source expression for the schema's default, or None."""
    if "default" not in schema:
        return None
    try:
        return repr(schema["default"])
    except Exception:
        return None


def _format_scalar(value: Any) -> str:
    """Render a schema value compactly. Gemini widens integers to floats."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str):
        return value
    return str(value)


def _constraint_suffix(schema: dict[str, Any], *, include_default: bool = False) -> str:
    """Render a schema's validation keywords as ``(minimum=1, maximum=7)``.

    Array constraints living on ``items`` are folded in under an ``items:``
    prefix, so ``daysOfWeek`` shows its 1-7 bound rather than hiding it a level
    down where the type expression has already collapsed to ``list[int]``.
    """
    parts = [f"{key}={_format_scalar(schema[key])}" for key in _CONSTRAINTS if key in schema]
    examples = schema.get("examples")
    if isinstance(examples, list) and examples:
        parts.append("examples=" + ", ".join(_format_scalar(v) for v in examples[:2]))
    if include_default and "default" in schema:
        parts.append(f"default={_format_scalar(schema['default'])}")
    parts.extend(label for key, label in _FLAGS if schema.get(key) is True)
    if schema.get("additionalProperties") is False:
        parts.append("no other keys")
    dependent = schema.get("dependentRequired")
    if isinstance(dependent, dict):
        parts.extend(
            f"{name} requires {', '.join(needed)}" for name, needed in dependent.items() if needed
        )
    pattern_properties = schema.get("patternProperties")
    if isinstance(pattern_properties, dict) and pattern_properties:
        parts.append("key patterns: " + ", ".join(pattern_properties))

    items = schema.get("items")
    if isinstance(items, dict):
        item_schema = _effective_schema(items)
        item_parts = [
            f"{key}={_format_scalar(item_schema[key])}"
            for key in _CONSTRAINTS
            if key in item_schema
        ]
        if item_parts:
            parts.append("items: " + ", ".join(item_parts))

    return f" ({', '.join(parts)})" if parts else ""


def _prose(schema: dict[str, Any]) -> str:
    """The schema's human text: ``description``, else ``title``.

    Plenty of real schemas carry only a ``title``; dropping it would leave the
    field with a type and nothing else.
    """
    for key in ("description", "title"):
        text = str(schema.get(key, "")).strip()
        if text:
            return text
    return ""


def _const_or_single_enum(schema: dict[str, Any]) -> str | None:
    """A branch tag value: ``const`` or a one-element ``enum``."""
    if "const" in schema:
        return _format_scalar(schema["const"])
    enum = schema.get("enum")
    if isinstance(enum, list) and len(enum) == 1:
        return _format_scalar(enum[0])
    return None


def _union_branch_tag(branch: dict[str, Any], *, discriminator: str | None) -> str | None:
    """Label a union branch without implying a nested object key.

    OpenAPI discriminator unions (``type: openapi | mcp``) often lose
    ``discriminator`` once ADK inlines ``$ref`` into ``items``, leaving only a
    single-value ``type`` enum on each object. Tag as ``(type=openapi)`` rather
    than a fake ``openapi:`` key the model might nest under.
    """
    props = branch.get("properties")
    if not isinstance(props, dict):
        return None
    names: list[str] = []
    if discriminator:
        names.append(discriminator)
    if "type" not in names:
        names.append("type")
    for name in names:
        raw = props.get(name)
        if not isinstance(raw, dict):
            continue
        value = _const_or_single_enum(_effective_schema(raw))
        if value is not None:
            return f"{name}={value}"
    return None


def _describe_union(
    schema: dict[str, Any], variants: list[Any], *, indent: str, depth: int
) -> list[str]:
    """Expand object ``oneOf``/``anyOf`` branches into the docstring.

    The signature collapses every object variant to ``dict[str, Any]``, so
    without this the keys on a discriminator union never reach the model.
    Primitive-only unions stay in the type expression and are skipped here.
    """
    disc = schema.get("discriminator")
    discriminator = None
    if isinstance(disc, dict):
        name = disc.get("propertyName")
        if isinstance(name, str) and name:
            discriminator = name

    parent_required = schema.get("required")
    lines: list[str] = []
    for raw in variants:
        branch = _effective_schema(raw)
        if isinstance(parent_required, list):
            branch = _merge_schema(branch, {"required": parent_required})
        nested = _describe_fields(branch, indent=indent, depth=depth + 1)
        if not nested:
            continue
        if lines:
            lines.append(f"{indent}-- or --")
        tag = _union_branch_tag(branch, discriminator=discriminator)
        description = _prose(branch).splitlines()
        suffix = _constraint_suffix(branch)
        if tag:
            head = f"{indent}({tag})"
            if description:
                head += f" - {description[0].strip()}"
            lines.append(head + suffix)
        elif description:
            lines.append(f"{indent}{description[0].strip()}{suffix}")
        elif suffix:
            lines.append(f"{indent}{suffix.strip()}")
        for extra in description[1:]:
            lines.append(f"{indent}    {extra.strip()}")
        lines.extend(nested)
    return lines


def _describe_properties(schema: dict[str, Any], *, indent: str, depth: int) -> list[str]:
    """One line per object property, plus nested expansion."""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return []
    required = {name for name in (schema.get("required") or []) if isinstance(name, str)}
    lines: list[str] = []
    for name, raw in properties.items():
        field = _effective_schema(raw)
        head = f"{indent}{name}: {_schema_to_type(field)}"
        if name in required:
            head += " (required)"
        description = _prose(field).splitlines()
        if description:
            head += f" - {description[0].strip()}"
        lines.append(head + _constraint_suffix(field, include_default=True))
        for extra in description[1:]:
            lines.append(f"{indent}    {extra.strip()}")
        lines.extend(_describe_fields(field, indent=indent + "    ", depth=depth + 1))
    return lines


def _describe_prefix_items(schema: dict[str, Any], *, indent: str, depth: int) -> list[str]:
    """Expand object (or union) slots of a ``prefixItems`` tuple.

    Primitive slots are already in the ``tuple[...]`` type expression.
    """
    prefix_items = schema.get("prefixItems")
    if not isinstance(prefix_items, list) or not prefix_items:
        return []
    lines: list[str] = []
    for index, raw in enumerate(prefix_items):
        field = _effective_schema(raw)
        nested = _describe_fields(field, indent=indent + "    ", depth=depth + 1)
        if not nested:
            continue
        head = f"{indent}[{index}]: {_schema_to_type(field)}"
        description = _prose(field).splitlines()
        if description:
            head += f" - {description[0].strip()}"
        lines.append(head + _constraint_suffix(field, include_default=True))
        for extra in description[1:]:
            lines.append(f"{indent}    {extra.strip()}")
        lines.extend(nested)
    return lines


def _describe_fields(schema: dict[str, Any], *, indent: str, depth: int = 0) -> list[str]:
    """Lines describing an object's properties, recursing into nested schemas.

    The signature can only ever say ``dict[str, Any]`` for an object, so without
    this the key names, their types and their meanings never reach the model.
    Multi-branch ``oneOf``/``anyOf`` is walked the same way: a single-branch
    union is folded in ``_effective_schema``, but two object variants have no
    ``properties`` at the union itself. Shared parent ``properties`` are listed
    first, then the branches.
    """
    if depth >= _MAX_SCHEMA_DEPTH:
        return []

    schema = _effective_schema(schema)

    variants = schema.get("anyOf") or schema.get("oneOf")
    lines = _describe_properties(schema, indent=indent, depth=depth)
    if isinstance(variants, list) and len(variants) > 1:
        lines.extend(_describe_union(schema, variants, indent=indent, depth=depth))
    if lines:
        return lines

    prefix_lines = _describe_prefix_items(schema, indent=indent, depth=depth)
    if prefix_lines:
        return prefix_lines

    items = schema.get("items")
    if isinstance(items, dict):
        return _describe_fields(items, indent=indent, depth=depth + 1)
    return lines


def _format_docstring(
    *, description: str, param_docs: list[tuple[str, dict[str, Any], str | None]]
) -> str:
    lines: list[str] = []
    summary = (description or "").strip() or "Call the tool."
    lines.extend(textwrap.wrap(summary, width=88) or [summary])
    if param_docs:
        lines.append("")
        lines.append("Args:")
        for name, schema, default_repr in param_docs:
            desc = _prose(schema)
            first, *rest = (desc or "").splitlines() or [""]
            suffix = f" (default: {default_repr})" if default_repr is not None else ""
            head = (first + _constraint_suffix(schema) + suffix).strip()
            lines.append(f"    {name}: {head}" if head else f"    {name}:")
            for extra in rest:
                lines.append(f"        {extra}")
            lines.extend(_describe_fields(schema, indent="        "))
    body = "\n    ".join(lines).replace("\\", "\\\\").replace('"""', '""\\"')
    return f'"""{body}\n    """'


# Names a generated class must not take: the stub's own imports.
_TYPING_NAMES = frozenset({"Any", "Literal", "NotRequired", "TYPE_CHECKING", "TypedDict"})

_REF_PREFIXES = ("#/$defs/", "#/definitions/")

_Fields = tuple[tuple[str, str], ...]


def _pascal(raw: str) -> str:
    """``first_name`` / ``firstName`` / ``first-name`` -> ``FirstName``."""
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[^0-9a-zA-Z]+", raw))


def _is_object(schema: dict[str, Any]) -> bool:
    properties = schema.get("properties")
    return (
        schema.get("type") in ("object", None) and isinstance(properties, dict) and bool(properties)
    )


def _typed_dict_source(name: str, fields: _Fields, optional: bool) -> str:
    """Class syntax, or the functional syntax when a key isn't an identifier."""
    total = ", total=False" if optional else ""
    if all(key.isidentifier() and not keyword.iskeyword(key) for key, _ in fields):
        body = "".join(f"\n    {key}: {ty}" for key, ty in fields)
        return f"class {name}(TypedDict{total}):{body}"
    items = ", ".join(f"{key!r}: {ty!r}" for key, ty in fields)
    return f"{name} = TypedDict({name!r}, {{{items}}}{total})"


def _referenced_defs(schema: Any, defs: dict[str, Any]) -> dict[str, Any]:
    """The ``defs`` entries ``schema`` refers to, transitively."""
    found: dict[str, Any] = {}

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(_REF_PREFIXES):
                name = ref.split("/")[-1]
                if name in defs and name not in found:
                    found[name] = defs[name]
                    visit(defs[name])
            for value in node.values():
                visit(value)

    visit(schema)
    return found


class _ModuleTypes:
    """The ``TypedDict`` return types of one module's tools, shared by shape.

    Identical shapes get one class, so the catalog defines a shape that many
    tools return once. A class is named after its schema (a ``$defs`` key or
    ``title``), else the property holding it; when a different shape already
    has that name, the parent's name is prefixed, then a number appended. An
    untitled top-level object is the tool's own ``<Tool>Response``, never shared.
    """

    def __init__(self, reserved: set[str]) -> None:
        self.sources: dict[str, str] = {}
        self._by_shape: dict[str, str] = {}
        self._reserved = _TYPING_NAMES | reserved

    def uses(self, type_expr: str) -> tuple[str, ...]:
        """Sources of the classes ``type_expr`` refers to, transitively, parents first."""
        names: dict[str, None] = {}

        def visit(text: str) -> None:
            for word in re.findall(r"\w+", text):
                if word in self.sources and word not in names:
                    names[word] = None
                    visit(self.sources[word])

        visit(type_expr)
        return tuple(self.sources[name] for name in names)

    def type_of(
        self,
        schema: Any,
        *,
        short: str,
        parent: str,
        defs: dict[str, Any],
        depth: int = 0,
        own: bool = False,
    ) -> str:
        """Convert a response schema to a type expression, defining its classes.

        ``short`` names an object found here; ``parent`` is the enclosing
        class's preferred name, used to qualify ``short`` on a clash. ``own``
        marks the tool's top level, whose untitled object isn't shared.
        """
        schema = _effective_schema(schema)
        if depth >= _MAX_SCHEMA_DEPTH:
            return "Any"
        ty = schema.get("type")
        ref = schema.get("$ref")
        variants = schema.get("anyOf") or schema.get("oneOf")
        extra = schema.get("additionalProperties")
        title = schema.get("title")

        def child(sub: Any, short: str, own: bool = False) -> str:
            return self.type_of(
                sub, short=short, parent=parent, defs=defs, depth=depth + 1, own=own
            )

        if isinstance(ref, str) and ref.startswith(_REF_PREFIXES) and ref.split("/")[-1] in defs:
            name = ref.split("/")[-1]
            parts = [self._ref(name, defs[name], parent=parent, defs=defs, depth=depth)]
        elif "enum" in schema:
            return _schema_to_type(schema)
        elif isinstance(variants, list) and variants:
            parts = [child(v, short, own) for v in variants]
        elif isinstance(ty, list):
            rest = {k: v for k, v in schema.items() if k not in ("type", "nullable")}
            parts = [child({**rest, "type": t}, short, own) for t in ty]
        elif _is_object(schema):
            candidates = [_pascal(title)] if isinstance(title, str) and title else []
            names = [*candidates, short, parent + short]
            share = bool(candidates) or not own
            parts = [self._object(schema, names, defs, depth, share=share)]
        elif ty == "object" and isinstance(extra, dict) and extra:
            parts = [f"dict[str, {child(extra, f'{short}Value')}]"]
        elif ty == "array" and (isinstance(schema.get("items"), dict) or schema.get("prefixItems")):
            # JSON arrays reach code as lists, so `prefixItems` slots join the item union.
            items = schema.get("items")
            slots = [
                *(schema.get("prefixItems") or []),
                *([items] if isinstance(items, dict) else []),
            ]
            elements = list(dict.fromkeys(child(slot, f"{short}Item") for slot in slots))
            parts = [f"list[{'Any' if 'Any' in elements else ' | '.join(elements)}]"]
        else:
            return _schema_to_type(schema)
        unique = list(dict.fromkeys(parts))
        if "Any" in unique:
            return "Any"
        if schema.get("nullable") and "None" not in unique:
            unique.append("None")
        return " | ".join(unique)

    def _claim(self, candidates: list[str]) -> str:
        valid = [
            name
            for name in candidates
            if name.isidentifier() and not keyword.iskeyword(name) and name not in self._reserved
        ] or ["Type"]
        numbered = (f"{valid[0]}{n}" for n in itertools.count(2))
        name = next(name for name in itertools.chain(valid, numbered) if name not in self.sources)
        self.sources[name] = ""  # Taken; the source follows once the fields are rendered.
        return name

    def _fields(
        self, schema: dict[str, Any], *, parent: str, defs: dict[str, Any], depth: int
    ) -> tuple[_Fields, bool]:
        """Each property's type, and whether every key is optional (none required)."""
        required = {key for key in schema.get("required") or [] if isinstance(key, str)}
        fields = []
        for key, raw in schema["properties"].items():
            ty = self.type_of(raw, short=_pascal(key), parent=parent, defs=defs, depth=depth + 1)
            fields.append((key, f"NotRequired[{ty}]" if required and key not in required else ty))
        return tuple(fields), not required

    def _object(
        self,
        schema: dict[str, Any],
        candidates: list[str],
        defs: dict[str, Any],
        depth: int,
        *,
        share: bool,
    ) -> str:
        fields, optional = self._fields(schema, parent=candidates[0], defs=defs, depth=depth)
        shape = repr((fields, optional))
        if share and shape in self._by_shape:
            return self._by_shape[shape]
        name = self._claim(candidates)
        self.sources[name] = _typed_dict_source(name, fields, optional)
        if share:
            self._by_shape[shape] = name
        return name

    def _ref(self, name: str, target: Any, *, parent: str, defs: dict[str, Any], depth: int) -> str:
        """A ``$defs`` entry, named after its key. Claimed before its fields, so it may recurse."""
        target = _effective_schema(target)
        if not _is_object(target):
            return self.type_of(
                target, short=_pascal(name), parent=parent, defs=defs, depth=depth + 1
            )
        key = [target, _referenced_defs(target, defs)]
        shape = "$ref " + json.dumps(key, sort_keys=True, default=str)
        if shape not in self._by_shape:
            cls = self._by_shape[shape] = self._claim([_pascal(name), parent + _pascal(name)])
            fields, optional = self._fields(target, parent=cls, defs=defs, depth=depth)
            self.sources[cls] = _typed_dict_source(cls, fields, optional)
            self._by_shape.setdefault(repr((fields, optional)), cls)
        return self._by_shape[shape]


def _sanitise_param(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw).strip("_")
    if not cleaned:
        cleaned = "arg"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    if keyword.iskeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


@dataclass(frozen=True)
class RenderedParam:
    """One parameter as it appears in a generated tool function."""

    py_name: str
    raw_name: str
    type_expr: str
    is_required: bool
    schema_default: str | None
    """``repr()`` of the schema's default value if present; otherwise ``None``."""


@dataclass(frozen=True)
class RenderedTool:
    """Structured tool ready for stub or catalog rendering.

    A tool with exactly one required parameter takes it positionally or by
    keyword; every other parameter is keyword-only.

    Argument forwarding rules (Option A):

    - Required, no schema default: ``name: T``; always forwarded.
    - Required, schema default: ``name: T = <default>``; always forwarded.
    - Optional (with or without schema default): ``name: T | None = _MISSING``
      in the on-disk stub (forwarded only if the caller passed a value, so
      the host-side tool's own default behaviour applies on omission); the
      catalog renders this as ``name: T | None = ...`` since the sentinel is
      an implementation detail. Schema default (when present) is surfaced in
      the docstring rather than baked into the signature, since Python erases
      "argument was not passed" once a real default lands in the parameter
      slot.
    """

    attribute: str
    dotted_path: str
    namespace: str | None
    docstring: str
    params: tuple[RenderedParam, ...]
    return_type: str
    types: tuple[str, ...]
    """Source of each ``TypedDict`` the return type uses, transitively, parents first."""

    def signature_for(self, target: Target) -> str:
        """Return the ``def name(...) -> T:`` line for the requested target."""
        params = _render_params(self.params, target=target)
        return f"def {self.attribute}({params}) -> {self.return_type}:"

    @property
    def needs_missing_sentinel(self) -> bool:
        return any(not p.is_required for p in self.params)


def _render_params(params: tuple[RenderedParam, ...], *, target: Target) -> str:
    """Render the parameter list; a sole required parameter may be positional."""
    pieces: list[str] = []
    optional_default = "_MISSING" if target == "stub" else "..."
    positional = sum(p.is_required for p in params) == 1
    for i, p in enumerate(params):
        if i == int(positional):
            pieces.append("*")
        if p.is_required:
            if p.schema_default is None:
                pieces.append(f"{p.py_name}: {p.type_expr}")
            else:
                pieces.append(f"{p.py_name}: {p.type_expr} = {p.schema_default}")
        else:
            pieces.append(f"{p.py_name}: {_ensure_optional_type(p.type_expr)} = {optional_default}")
    return ", ".join(pieces)


def _ensure_optional_type(type_expr: str) -> str:
    """Return ``type_expr`` augmented with ``| None`` if not already present.

    Avoids ``str | None | None`` for params whose schema was already
    ``nullable: true`` (which ``_schema_to_type`` rendered as ``str | None``).
    """
    parts = [p.strip() for p in type_expr.split("|")]
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        if part and part not in seen:
            seen.add(part)
            deduped.append(part)
    if "None" not in seen:
        deduped.append("None")
    return " | ".join(deduped)


@dataclass(frozen=True)
class StubFile:
    path: str
    source: str


def render_tools(tools: list[NamespacedTool]) -> list[RenderedTool]:
    """Build a ``RenderedTool`` for each tool, in order.

    Captures everything stub and catalog rendering need: signature parts,
    return type and its ``TypedDict``s, formatted docstring, dotted path,
    namespace. Tools of one module share their ``TypedDict``s; names are
    claimed in attribute order, so stubs and the catalog agree on them.
    """
    modules = {
        ns: _ModuleTypes({t.attribute for t in tools if t.namespace == ns})
        for ns in {t.namespace for t in tools}
    }
    rendered = {
        id(nt): _render_tool(nt, modules[nt.namespace])
        for nt in sorted(tools, key=lambda t: t.attribute)
    }
    return [rendered[id(nt)] for nt in tools]


def render_tool(nt: NamespacedTool) -> RenderedTool:
    """Build a ``RenderedTool`` for a single tool; see ``render_tools``."""
    return render_tools([nt])[0]


def _render_tool(nt: NamespacedTool, types: _ModuleTypes) -> RenderedTool:
    declaration = nt.resolved.tool._get_declaration()
    schema: dict[str, Any] = _schema_dict_from_declaration(declaration)

    props: dict[str, Any] = dict(schema.get("properties") or {})
    required = set(schema.get("required") or [])
    ordered = sorted(props.items(), key=lambda kv: (kv[0] not in required, kv[0]))

    params: list[RenderedParam] = []
    param_docs: list[tuple[str, dict[str, Any], str | None]] = []

    seen_param_names: dict[str, str] = {}

    for raw_name, subschema in ordered:
        py_name = _sanitise_param(raw_name)
        existing_raw_name = seen_param_names.get(py_name)
        if existing_raw_name is not None:
            raise PythonNameCollisionError(
                f"Tool {nt.tool_name!r} has parameter names {existing_raw_name!r} and "
                f"{raw_name!r} that both map to Python parameter {py_name!r}. Rename one "
                "of the tool parameters so the generated stub signature is unambiguous."
            )
        seen_param_names[py_name] = raw_name
        subschema = _effective_schema(subschema)
        type_expr = _schema_to_type(subschema)
        default = _python_default(subschema)
        is_required = raw_name in required
        params.append(
            RenderedParam(
                py_name=py_name,
                raw_name=raw_name,
                type_expr=type_expr,
                is_required=is_required,
                schema_default=default,
            )
        )
        doc_default = default if (not is_required and default is not None) else None
        param_docs.append((py_name, subschema, doc_default))

    description = ""
    if declaration is not None and declaration.description:
        description = declaration.description
    elif nt.resolved.tool.description:
        description = nt.resolved.tool.description
    docstring = _format_docstring(description=description, param_docs=param_docs)

    tool = nt.resolved.tool
    inner = tool._wrapped if isinstance(tool, ToolResultArtifactTool) else tool
    response = output_schema(inner) or getattr(declaration, "response_json_schema", None)
    return_type = "Any"
    if isinstance(response, dict) and response:
        defs = {**(response.get("definitions") or {}), **(response.get("$defs") or {})}
        short = _pascal(nt.attribute) + "Response"
        return_type = types.type_of(response, short=short, parent="", defs=defs, own=True)

    return RenderedTool(
        attribute=nt.attribute,
        dotted_path=nt.dotted_path,
        namespace=nt.namespace,
        docstring=docstring,
        params=tuple(params),
        return_type=return_type,
        types=types.uses(return_type),
    )


def render_tool_source(rt: RenderedTool) -> str:
    """Render the on-disk Python stub source for a tool."""
    sig = rt.signature_for("stub")
    # `NotRequired` is 3.11+; annotations are never evaluated, so a sandbox on
    # 3.10 only needs the import kept out of runtime.
    not_required = any("NotRequired[" in t for t in rt.types)
    lines: list[str] = [
        "# Generated by adk-code-mode. Do not edit.",
        "from __future__ import annotations",
        "",
        f"from typing import {'TYPE_CHECKING, ' if not_required else ''}Any, Literal, TypedDict",
        "",
        "from adk_code_mode_sandbox._rpc_client import call as _call",
        "",
    ]
    if not_required:
        lines.extend(["if TYPE_CHECKING:", "    from typing import NotRequired", ""])
    lines.append("")
    if rt.needs_missing_sentinel:
        lines.extend(["_MISSING = object()", "", ""])
    for source in rt.types:
        lines.extend([source, "", ""])
    lines.append(sig)
    lines.append(f"    {rt.docstring}")

    unconditional = [p for p in rt.params if p.is_required]
    optional = [p for p in rt.params if not p.is_required]
    if unconditional:
        lines.append("    _args: dict[str, Any] = {")
        for p in unconditional:
            lines.append(f"        {p.raw_name!r}: {p.py_name},")
        lines.append("    }")
    else:
        lines.append("    _args: dict[str, Any] = {}")
    for p in optional:
        lines.append(f"    if {p.py_name} is not _MISSING:")
        lines.append(f"        _args[{p.raw_name!r}] = {p.py_name}")
    lines.append(f"    return _call({rt.dotted_path!r}, _args)")
    lines.append("")
    return "\n".join(lines)


def render_namespace_init(tools: list[NamespacedTool]) -> str:
    """Render ``__init__.py`` for a namespace package.

    Re-exports each tool in the namespace so the model can write
    ``from tools.<namespace> import <name>``.
    """
    sorted_tools = sorted(tools, key=lambda t: t.attribute)
    imports = "\n".join(f"from .{t.attribute} import {t.attribute}" for t in sorted_tools)
    all_list = ", ".join(repr(t.attribute) for t in sorted_tools)
    header = "# Generated by adk-code-mode. Do not edit.\n"
    return f"{header}from __future__ import annotations\n\n{imports}\n\n__all__ = [{all_list}]\n"


_ROOT_MARKER = "# Generated by adk-code-mode. Do not edit.\n"


_ERRORS = ("HTTPStatusError", "ToolError")


def render_root_init(top_level: list[NamespacedTool]) -> str:
    """Render the generated ``tools`` package ``__init__.py``.

    Re-exports the errors a tool call raises and any top-level (non-namespaced)
    tools, so the model can write ``from tools import <name>``. Namespaced tools
    are *not* re-exported here — the model uses ``from tools.<namespace> import
    <name>``, which only loads that namespace's stubs.
    """
    sorted_tools = sorted(top_level, key=lambda t: t.attribute)
    imports = "\n".join(
        [
            f"from adk_code_mode_sandbox._rpc_client import {', '.join(_ERRORS)}",
            *(f"from .{t.attribute} import {t.attribute}" for t in sorted_tools),
        ]
    )
    all_list = ", ".join(repr(name) for name in [*_ERRORS, *(t.attribute for t in sorted_tools)])
    return (
        f"{_ROOT_MARKER}from __future__ import annotations\n\n{imports}\n\n__all__ = [{all_list}]\n"
    )


def render_tree(namespaced: list[NamespacedTool]) -> list[StubFile]:
    """Render the full stub tree for a list of tools.

    Files are returned with POSIX paths rooted at the generated ``tools``
    package directory. The runtime mounts that package directory at ``/tools``
    and adds its parent to ``sys.path`` so ``import tools`` resolves directly
    to ``/tools/__init__.py``. The root ``__init__.py`` re-exports top-level tools (so
    ``from tools import <name>`` works) but does **not** re-export
    namespaced tools — the model receives a tool catalog up-front and writes
    ``from tools.<namespace> import <name>`` directly, so eagerly re-exporting
    every stub from the root would be wasteful at large surfaces.
    """
    per_tool = sorted(namespaced, key=lambda t: t.dotted_path)
    files: list[StubFile] = []

    grouped: dict[str | None, list[NamespacedTool]] = {}
    for nt in per_tool:
        grouped.setdefault(nt.namespace, []).append(nt)

    for nt, rendered in zip(per_tool, render_tools(per_tool)):
        sub = "" if nt.namespace is None else f"{nt.namespace}/"
        files.append(StubFile(path=f"{sub}{nt.attribute}.py", source=render_tool_source(rendered)))

    for ns, tools in grouped.items():
        if ns is None:
            continue
        files.append(
            StubFile(
                path=f"{ns}/__init__.py",
                source=render_namespace_init(tools),
            )
        )

    top_level = grouped.get(None, [])
    files.append(StubFile(path="__init__.py", source=render_root_init(top_level)))
    return sorted(files, key=lambda f: f.path)


__all__ = [
    "RenderedParam",
    "RenderedTool",
    "StubFile",
    "render_namespace_init",
    "render_root_init",
    "render_tool",
    "render_tool_source",
    "render_tools",
    "render_tree",
]
