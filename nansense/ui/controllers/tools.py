"""The Tools page's form fields and result layout, independent of widgets.

Each custom tool's argument form is built from the same JSON schema MCP
describes it with (`ToolSpec.json_schema`), so the page and `nansense-cli`
offer the same arguments with the same defaults.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

FieldKind = Literal["int", "float", "bool", "select", "text", "list"]
_KINDS: dict[str, FieldKind] = {
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "string": "text",
    "array": "list",
}


@dataclass(frozen=True)
class ToolField:
    """One argument's widget: its kind, default and constraints.

    `nullable` (an `X | None` argument) lets an empty widget mean `None`; a
    `list` field takes a JSON array or comma-separated values.
    """

    key: str
    label: str
    kind: FieldKind
    default: object = None
    required: bool = False
    nullable: bool = False
    options: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""


def tool_fields(schema: Mapping[str, Any]) -> list[ToolField]:
    """The form fields for a tool's argument `schema`, in argument order."""
    required = set(schema.get("required", ()))
    fields: list[ToolField] = []
    for key, prop in schema.get("properties", {}).items():
        variants = prop.get("anyOf", [prop])
        nullable = any(v.get("type") == "null" for v in variants)
        main = next((v for v in variants if v.get("type") != "null"), prop)
        enum = main.get("enum", prop.get("enum"))
        kind: FieldKind = "select" if enum else _KINDS.get(main.get("type", ""), "text")
        fields.append(
            ToolField(
                key=key,
                label=prop.get("title", key),
                kind=kind,
                default=prop.get("default"),
                required=key in required,
                nullable=nullable,
                options=tuple(enum or ()),
                minimum=main.get("minimum", main.get("exclusiveMinimum")),
                maximum=main.get("maximum", main.get("exclusiveMaximum")),
                description=prop.get("description", ""),
            )
        )
    return fields


def collect_arguments(
    fields: list[ToolField], values: Mapping[str, object]
) -> dict[str, object]:
    """Widget values as call arguments: empty optional fields are left to
    their defaults (or `None` when nullable), list fields are parsed."""
    arguments: dict[str, object] = {}
    for field in fields:
        value = values.get(field.key)
        if value is None or value == "":
            if field.nullable:
                arguments[field.key] = None
            elif field.required:
                raise ValueError(f"Enter a value for {field.label}.")
            continue
        if field.kind == "list" and isinstance(value, str):
            text = value.strip()
            value = (
                json.loads(text)
                if text.startswith("[")
                else [item.strip() for item in text.split(",") if item.strip()]
            )
        arguments[field.key] = value
    return arguments


TableRows = tuple[str, list[str], list[dict[str, object]]]


@dataclass(frozen=True)
class ResultLayout:
    """A result split for display: `tables` (title, columns, rows) and the
    rest of the JSON, if any, shown as code."""

    tables: tuple[TableRows, ...]
    rest: object = None


def _as_table(title: str, value: Any) -> TableRows | None:
    """`value` as a titled table when it is a returned `nansense.Table`."""
    if not isinstance(value, dict):
        return None
    columns = value.get("columns")
    rows = value.get("rows")
    if not (isinstance(columns, list) and isinstance(rows, list)):
        return None
    if not all(isinstance(row, dict) for row in rows):
        return None
    return title, [str(c) for c in columns], [
        {str(k): v for k, v in row.items()} for row in rows
    ]


def result_layout(value: Any) -> ResultLayout:
    """Show tables (a returned `nansense.Table`, at the top level or one level
    down a dict) as tables, and everything else as JSON."""
    table = _as_table("", value)
    if table is not None:
        return ResultLayout((table,))
    if not isinstance(value, dict):
        return ResultLayout((), value)
    tables: list[TableRows] = []
    rest: dict[str, object] = {}
    for key, item in value.items():
        table = _as_table(str(key), item)
        if table is None:
            rest[str(key)] = item
        else:
            tables.append(table)
    return ResultLayout(tuple(tables), rest or None)
