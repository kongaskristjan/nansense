"""The Tools page's form fields, argument collection and result layout."""

from __future__ import annotations

from typing import Annotated, Literal

import pydantic
import pytest

import nansense
from nansense import ToolContext
from nansense.tools import ToolCall
from nansense.ui.common import _CHIP_TONES
from nansense.ui.controllers.tools import collect_arguments, result_layout, tool_fields
from nansense.ui.tools_page import call_chip
from tests.nansense.helpers import TinyNet


def _fields() -> list:
    session = nansense.start(TinyNet(), epochs=1, phases={"train": 1})

    @session.tool()
    def probe(
        ctx: ToolContext,
        count: Annotated[int, pydantic.Field(ge=1, description="How many")] = 3,
        order: Literal["a", "b"] = "a",
        names: list[str] | None = None,
        scale: float | None = None,
        flag: bool = False,
    ) -> None:
        pass

    return tool_fields(session.tool_specs["probe"].json_schema())


def test_fields_follow_the_argument_schema() -> None:
    fields = {f.key: f for f in _fields()}
    assert [f.kind for f in fields.values()] == ["int", "select", "list", "float", "bool"]
    assert fields["count"].minimum == 1 and fields["count"].description == "How many"
    assert fields["order"].options == ("a", "b")
    assert fields["scale"].nullable and fields["scale"].default is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"count": 2, "names": "x, y"}, {"count": 2, "names": ["x", "y"], "scale": None}),
        ({"names": '["x"]', "scale": ""}, {"names": ["x"], "scale": None}),
    ],
)
def test_empty_optional_fields_fall_back_and_lists_parse(
    values: dict[str, object], expected: dict[str, object]
) -> None:
    arguments = collect_arguments(_fields(), values)
    assert {k: v for k, v in arguments.items() if k in expected} == expected


def test_tables_are_pulled_out_of_a_result() -> None:
    table = {"columns": ["a"], "rows": [{"a": 1}]}
    assert result_layout(table).tables == (("", ["a"], [{"a": 1}]),)
    layout = result_layout({"matrix": table, "accuracy": 0.5})
    assert layout.tables == (("matrix", ["a"], [{"a": 1}]),)
    assert layout.rest == {"accuracy": 0.5}
    assert result_layout([1, 2]).rest == [1, 2]


@pytest.mark.parametrize("stage", ["queued", "running", "done", "absent"])
@pytest.mark.parametrize("error", [None, "boom"])
def test_every_call_chip_has_a_defined_tone(stage: str, error: str | None) -> None:
    chip = call_chip(ToolCall(1, "t", stage, error=error), training_running=True)
    assert chip.tone in _CHIP_TONES
