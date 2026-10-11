"""The `/tools` page: the training script's own tools (`Session.tool`).

The human counterpart of calling a custom tool from `nansense-cli`: one card
per tool, a form built from its argument schema, a Run button, and the
result — tables as tables, the rest as JSON.
"""

from __future__ import annotations

import json

from nicegui import ui

from nansense.session import Session
from nansense.tools import ToolCall, ToolSpec
from nansense.ui.common import _page_scaffold, _StatusChip, _StatusPill
from nansense.ui.controllers.tools import (
    ToolField,
    collect_arguments,
    result_layout,
    tool_fields,
)
from nansense.ui.top_bar import (
    _add_error_banner,
    _add_repo_logo,
    _add_step_controls,
    _back_button,
    _build_step_until_custom_dialog,
    _top_bar_row,
)


def add_tools_button(session: Session) -> None:
    """The top-bar link to `/tools`, shown only when tools are registered."""
    if not session.tool_specs:
        return
    ui.button(icon="handyman", color="slate-500").props(
        'dense size=md href="/tools"'
    ).tooltip("Custom tools")


def call_chip(call: ToolCall, *, training_running: bool) -> _StatusChip:
    """The status pill for one call."""
    if call.stage == "queued":
        text = (
            "Waiting for training to pause"
            if training_running
            else "Queued"
        )
        return _StatusChip("waiting", None, text)
    if call.stage == "running":
        return _StatusChip("running", None, "Running…")
    if call.stage == "absent":
        return _StatusChip("idle", "block", "Result no longer available. Run again.")
    if call.error is not None:
        return _StatusChip("failed", "error", "Failed")
    return _StatusChip("done", "check_circle", "Done")


def _build_tools_page(session: Session) -> None:
    _page_scaffold("Tools")
    step_until_custom = _build_step_until_custom_dialog(session)
    with ui.column().classes("w-full h-screen no-wrap gap-0"):
        with _top_bar_row():
            _back_button()
            _add_step_controls(session, step_until_custom)
            _add_repo_logo().classes("ml-auto")
        _add_error_banner(session)
        with ui.column().classes("grow min-h-0 w-full overflow-auto p-4 gap-4 bg-slate-200"):
            specs = session.tool_specs
            if not specs:
                ui.label("No tools are available in this session.").classes(
                    "text-slate-600"
                )
            for spec in specs.values():
                _ToolCard(session, spec)


class _ToolCard:
    """One tool: its description, argument form, Run button and result."""

    def __init__(self, session: Session, spec: ToolSpec) -> None:
        self.session = session
        self.spec = spec
        self.fields = tool_fields(spec.json_schema())
        self.widgets: dict[str, ui.element] = {}
        self.seq: int | None = None
        self.shown: ToolCall | None = None
        with ui.card().classes("w-full p-4 gap-2"):
            ui.label(spec.name).classes("font-mono text-base font-bold")
            if spec.description:
                ui.label(_reflowed(spec.description)).classes(
                    "text-sm text-slate-600 whitespace-pre-line"
                )
            with ui.row().classes("items-end gap-3 w-full"):
                for field in self.fields:
                    self.widgets[field.key] = _widget(field)
                ui.button("Run", icon="play_arrow", on_click=self.run, color="yellow-8").props(
                    "dense size=md"
                ).tooltip(
                    "Starts on a worker thread without waiting for a pause"
                    if not spec.on_training_thread
                    else "Runs on the training thread while training is paused"
                )
            self.pill = _StatusPill(_StatusChip("idle", "handyman", "Not run yet"))
            self.pill.set_visibility(False)
            self.error = ui.label("").classes("text-sm text-red-600 whitespace-pre-wrap")
            self.result = ui.column().classes("w-full gap-2")
        ui.timer(0.25, self.tick)

    def run(self) -> None:
        values = {key: getattr(w, "value", None) for key, w in self.widgets.items()}
        try:
            arguments = collect_arguments(self.fields, values)
            self.seq = self.session.request_tool(self.spec.name, arguments)
        except ValueError as error:
            self.error.text = str(error)
            return
        self.error.text = ""
        self.shown = None
        self.result.clear()
        self.pill.set_visibility(True)

    def tick(self) -> None:
        if self.seq is None:
            return
        call = self.session.tool_call(self.seq)
        self.pill.show(call_chip(call, training_running=self.session.is_running))
        if call.stage != "done" or call is self.shown:
            return
        self.shown = call
        self.error.text = call.error or ""
        if call.error is None:
            self._render(call.value)

    def _render(self, value: object) -> None:
        layout = result_layout(value)
        with self.result:
            for title, columns, rows in layout.tables:
                if title:
                    ui.label(title).classes("font-mono text-sm font-bold")
                keys = [f"c{i}" for i in range(len(columns))]
                ui.table(
                    columns=[
                        {"name": k, "label": c, "field": k, "sortable": True}
                        for k, c in zip(keys, columns, strict=True)
                    ],
                    rows=[{k: row.get(c) for k, c in zip(keys, columns, strict=True)} for row in rows],
                    pagination=25,
                ).props("dense flat bordered").classes("w-full")
            if layout.rest is not None:
                ui.code(json.dumps(layout.rest, indent=2), language="json").classes(
                    "w-full text-xs"
                )


def _reflowed(text: str) -> str:
    """A docstring's hard-wrapped lines joined, keeping its paragraphs."""
    return "\n\n".join(" ".join(p.split()) for p in text.split("\n\n"))


def _widget(field: ToolField) -> ui.element:
    default = field.default
    if field.kind == "bool":
        widget: ui.element = ui.switch(field.label, value=bool(default)).props("dense")
    elif field.kind == "select":
        widget = (
            ui.select(list(field.options), label=field.label, value=default)
            .props("dense outlined")
            .classes("min-w-32")
        )
    elif field.kind in ("int", "float"):
        widget = (
            ui.number(
                label=field.label,
                value=default if isinstance(default, (int, float)) else None,
                min=field.minimum,
                max=field.maximum,
                step=1 if field.kind == "int" else None,
                format="%d" if field.kind == "int" else None,
            )
            .props("dense outlined")
            .classes("w-40")
        )
    else:
        text = (
            ", ".join(str(item) for item in default)
            if isinstance(default, list)
            else "" if default is None else str(default)
        )
        widget = (
            ui.input(label=field.label, value=text)
            .props("dense outlined")
            .classes("w-56")
        )
    if field.description:
        widget.tooltip(field.description)
    return widget
