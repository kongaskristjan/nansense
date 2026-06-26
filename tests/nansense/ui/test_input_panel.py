"""Tests for pure helpers in `nansense.ui.input_panel`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from nicegui.elements.interactive_image import InteractiveImage
from nicegui.elements.label import Label
from nicegui.elements.row import Row
from nicegui.elements.switch import Switch

from nansense.session import Session
from nansense.input_config import InputTransform
from nansense.ui.input_panel import InputPanel, normalized_color

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# (input_name, sample, index) -> values, matching `probe.PerturbationMap`.
PerturbMap = dict[tuple[str, int, tuple[int, ...]], tuple[float, ...]]


@pytest.mark.parametrize(
    "hex_color, channels, mean, std, expected",
    [
        # RGB without stats: plain [0, 1] channel values.
        ("#ff0000", 3, None, None, (1.0, 0.0, 0.0)),
        ("#000000", 3, None, None, (0.0, 0.0, 0.0)),
        # Leading/trailing whitespace tolerated.
        (" #ffffff ", 3, None, None, (1.0, 1.0, 1.0)),
        # Grayscale: mean of the RGB components.
        ("#ff0000", 1, None, None, (1.0 / 3.0,)),
        ("#ffffff", 1, (0.5,), (0.5,), ((1.0 - 0.5) / 0.5,)),
    ],
)
def test_normalized_color_converts_display_colors(
    hex_color: str,
    channels: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
    expected: tuple[float, ...],
) -> None:
    values = normalized_color(hex_color, channels, mean, std)
    assert values == pytest.approx(expected)


def test_normalized_color_back_transforms_with_stats() -> None:
    values = normalized_color("#ffffff", 3, CIFAR_MEAN, CIFAR_STD)
    assert values is not None
    expected = tuple((1.0 - m) / s for m, s in zip(CIFAR_MEAN, CIFAR_STD))
    assert values == pytest.approx(expected)


@pytest.mark.parametrize(
    "transform, tensor, expected",
    [
        (None, torch.rand(1, 3, 4, 4), ("color", 3)),  # RGB image -> color picker
        (None, torch.rand(1, 1, 4, 4), ("color", 1)),  # grayscale -> color picker
        (None, torch.rand(1, 5, 4, 4), ("channels", 5)),  # non-RGB -> channel fields
        (None, torch.rand(2, 4), ("scalar", 1)),  # flat input -> one value field
        (None, None, ("none", 0)),  # nothing shown yet
        (lambda x: x[:, :3], torch.rand(1, 3, 4, 4), ("channels", 3)),  # transform wins
    ],
)
def test_desired_perturb_control(
    transform: object, tensor: torch.Tensor | None, expected: tuple[str, int]
) -> None:
    panel = InputPanel.__new__(InputPanel)
    panel._input_transform = cast("InputTransform | None", transform)
    assert panel._desired_perturb_control(tensor) == expected


def test_perturb_values_from_color_picker() -> None:
    panel = InputPanel.__new__(InputPanel)
    panel._perturb_kind = "color"
    panel._color = "#ff0000"
    panel._input_mean = None
    panel._input_std = None
    assert panel._perturb_values(3) == pytest.approx((1.0, 0.0, 0.0))


def test_perturb_values_from_channel_fields() -> None:
    panel = InputPanel.__new__(InputPanel)
    panel._perturb_kind = "channels"
    panel._value_inputs = [
        cast(Any, SimpleNamespace(value=1.5)),
        cast(Any, SimpleNamespace(value=None)),  # blank reads as 0
        cast(Any, SimpleNamespace(value=-2.0)),
    ]
    assert panel._perturb_values(3) == pytest.approx((1.5, 0.0, -2.0))
    assert panel._perturb_values(2) is None  # channel-count mismatch -> no write


@pytest.mark.parametrize(
    "hex_color, channels, mean, std",
    [
        ("#12345", 3, None, None),  # truncated hex
        ("#zzzzzz", 3, None, None),  # not hex digits
        ("#ffffff", 2, None, None),  # unsupported channel count
        ("#ffffff", 3, (0.5, 0.5), (0.5, 0.5)),  # stats length mismatch
    ],
)
def test_normalized_color_rejects_bad_input(
    hex_color: str,
    channels: int,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
) -> None:
    assert normalized_color(hex_color, channels, mean, std) is None


@pytest.mark.parametrize(
    "perturbations, expected",
    [
        ({}, False),
        ({("x", 0, (1, 2)): (0.5, 0.5, 0.5)}, True),
    ],
)
def test_compare_active_iff_perturbed(
    perturbations: PerturbMap,
    expected: bool,
) -> None:
    """`compare` is not a toggle: it derives from the perturbation map."""
    panel = InputPanel.__new__(InputPanel)
    panel._session = cast(
        Session, SimpleNamespace(perturbations=perturbations)
    )
    assert panel.compare is expected


class _FakeSwitch:
    def __init__(self, value: bool = False) -> None:
        self.value = value

    def set_value(self, value: bool) -> None:
        self.value = value


class _FakeText:
    def __init__(self) -> None:
        self.text = ""


class _FakeVisible:
    def __init__(self, visible: bool = False) -> None:
        self.visible = visible

    def set_visibility(self, visible: bool) -> None:
        self.visible = visible


class _FakeImage:
    """Tracks only whether the crosshair class is present."""

    def __init__(self, cursor: bool = False) -> None:
        self.cursor = cursor

    def classes(self, add: str | None = None, remove: str | None = None) -> None:
        if add and "cursor-crosshair" in add:
            self.cursor = True
        if remove and "cursor-crosshair" in remove:
            self.cursor = False


class _FakeSession:
    """The slice of `Session` `refresh_status` / `_on_clear` touch."""

    def __init__(self, perturbations: PerturbMap) -> None:
        self.is_pinned = False
        self.perturbations = perturbations
        self.pinned_position = None
        self.probe_error = ""
        self.clears = 0

    def clear_perturbations(self) -> None:
        self.clears += 1
        self.perturbations = {}


class _Wired:
    """A bare `InputPanel` wired with fakes, keeping handles for asserting."""

    def __init__(
        self, *, perturbations: PerturbMap, armed: bool, switch_value: bool
    ) -> None:
        self.switch = _FakeSwitch(switch_value)
        # The build keeps the cursor matching the switch, so start them aligned.
        self.image = _FakeImage(cursor=switch_value)
        self.session = _FakeSession(dict(perturbations))
        panel = InputPanel.__new__(InputPanel)
        panel._session = cast(Session, self.session)
        panel._syncing_pin = False
        panel._syncing_perturb = False
        panel._perturb_armed = armed
        panel._pin_switch = cast(Switch, _FakeSwitch(False))
        panel._perturb_switch = cast(Switch, self.switch)
        panel._image = cast(InteractiveImage, self.image)
        panel._pinned_caption = cast(Label, _FakeText())
        panel._perturb_caption = cast(Label, _FakeText())
        panel._clear_row = cast(Row, _FakeVisible(bool(perturbations)))
        panel._compare_caption = cast(Label, _FakeVisible(bool(perturbations)))
        panel._error_label = cast(Label, _FakeText())
        panel._on_change = lambda: None
        # No input selected -> `_sync_perturb_control` (called by refresh_status)
        # sees no input and stays a no-op, leaving the switch-sync under test.
        panel._selected_input = None
        panel._input_transform = None
        panel._perturb_kind = "none"
        panel._perturb_n = 0
        self.panel = panel


@pytest.mark.parametrize(
    "perturbations, armed, switch_value, expected",
    [
        # A perturbation made elsewhere (another tab, or one surviving a
        # rebuild after navigating back from /stats) turns the switch on.
        ({("x", 0, (1, 2)): (0.5, 0.5, 0.5)}, False, False, True),
        # An external clear turns it back off in a tab that didn't arm it.
        ({}, False, True, False),
        # A tab armed locally (toggled on, nothing clicked yet) stays on.
        ({}, True, True, True),
        # Armed and perturbed: already consistent, stays on.
        ({("x", 0, (1, 2)): (0.5, 0.5, 0.5)}, True, True, True),
    ],
)
def test_refresh_status_syncs_perturb_switch(
    perturbations: PerturbMap,
    armed: bool,
    switch_value: bool,
    expected: bool,
) -> None:
    """The switch mirrors `armed or perturbations`; the cursor follows it."""
    wired = _Wired(
        perturbations=perturbations, armed=armed, switch_value=switch_value
    )
    wired.panel.refresh_status()
    assert wired.switch.value is expected
    assert wired.image.cursor is expected


def test_on_perturb_change_ignores_programmatic_sync() -> None:
    """A guarded (mirrored) write must not re-arm or clear perturbations."""
    wired = _Wired(perturbations={}, armed=False, switch_value=True)
    wired.panel._syncing_perturb = True
    wired.panel._on_perturb_change(SimpleNamespace(value=True))
    assert wired.panel._perturb_armed is False
    assert wired.session.clears == 0


def test_on_perturb_change_arms_then_disarms() -> None:
    """A real toggle updates the local armed intent, cursor, and edits."""
    wired = _Wired(perturbations={}, armed=False, switch_value=False)

    wired.panel._on_perturb_change(SimpleNamespace(value=True))
    assert wired.panel._perturb_armed is True
    assert wired.image.cursor is True
    assert wired.session.clears == 0  # arming keeps any edits

    wired.panel._on_perturb_change(SimpleNamespace(value=False))
    assert wired.panel._perturb_armed is False
    assert wired.image.cursor is False
    assert wired.session.clears == 1  # disarming discards them
