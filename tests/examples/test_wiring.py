"""Source-level checks on how the examples wire NaNsense.

The examples are the reference wiring, so the invariants here are about the
`nansense.start(...)` call itself rather than about training: every example
declares its schedule up front, which is what makes the UI's per-phase totals
and boundary stops exact from the first batch instead of the second epoch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _example_mains() -> list[Path]:
    """Every example entrypoint, so a newly added example is covered without
    the test having to list it."""
    return sorted(_EXAMPLES_DIR.glob("*/main.py"))


def _start_calls(source: str) -> list[ast.Call]:
    """Every `nansense.start(...)` call in `source`."""
    return [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "start"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "nansense"
    ]


def test_examples_are_discovered() -> None:
    """Guards the glob: an empty list would make the checks below vacuous."""
    assert len(_example_mains()) >= 5


@pytest.mark.parametrize("main_py", _example_mains(), ids=lambda p: p.parent.name)
def test_start_declares_the_phase_schedule(main_py: Path) -> None:
    """Each `nansense.start(...)` passes `phases=`.

    The Lightning example has no `start()` call of its own — `NansenseCallback`
    declares the schedule from the trainer's dataloaders — so it passes with an
    empty call list.
    """
    for call in _start_calls(main_py.read_text()):
        keywords = {kw.arg for kw in call.keywords}
        assert "phases" in keywords, (
            f"{main_py.relative_to(_EXAMPLES_DIR.parent)}: nansense.start() on line "
            f"{call.lineno} does not declare `phases=`; the UI would then have no "
            "per-phase totals until the second epoch"
        )
