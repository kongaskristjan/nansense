"""Tests for axis-role options/defaults and the off-loop render path in
nansense.ui.weights_page."""

from __future__ import annotations

import asyncio

import pytest
import torch

from nansense.session import Session
from nansense.ui.weights_page import (
    _NO_GRADIENT_HTML,
    _compute_snapshot_renders,
    _default_roles,
    _PanelRender,
    _role_options,
    _WeightPanel,
)
from tests.nansense.helpers import _make_snapshot, make_session


@pytest.mark.parametrize(
    "ndim, expected",
    [
        (1, ["x", "index"]),
        (2, ["x", "y", "index"]),
        (3, ["x", "y", "tile", "index"]),
        (4, ["x", "y", "tile", "index"]),
    ],
)
def test_role_options_scale_with_rank(ndim: int, expected: list[str]) -> None:
    assert list(_role_options(ndim)) == expected


@pytest.mark.parametrize(
    "ndim, roles",
    [
        (1, ["x"]),
        (2, ["y", "x"]),
        (3, ["tile", "y", "x"]),
        (4, ["index", "tile", "y", "x"]),
    ],
)
def test_default_roles_match_default_dims(ndim: int, roles: list[str]) -> None:
    assert _default_roles(ndim) == roles


def _first_weight_panel(session: Session) -> _WeightPanel:
    """A panel for the first parameter the session's layers own."""
    layer = next(layer for layer, names in session.layer_weights.items() if names)
    name = session.layer_weights[layer][0]
    shape = tuple(dict(session.model.named_parameters())[name].shape)
    return _WeightPanel(name=name, shape=shape, session=session)


def test_compute_render_produces_strip_html() -> None:
    """The pure render step emits weight + gradient strip markup off the loop."""
    session, _ = make_session()
    panel = _first_weight_panel(session)
    tensor = torch.randn(*panel._shape)
    render = panel.compute_render(
        {panel.name: tensor},
        {panel.name: torch.randn(*panel._shape)},
        optimizer_state={},
        optimizer_hyperparams={},
    )
    assert render.error is None
    assert render.weight_html
    assert render.grad_html != _NO_GRADIENT_HTML


def test_compute_render_no_gradient_placeholder() -> None:
    """A weight with no captured gradient renders the placeholder note."""
    session, _ = make_session()
    panel = _first_weight_panel(session)
    render = panel.compute_render(
        {panel.name: torch.randn(*panel._shape)},
        {},
        optimizer_state={},
        optimizer_hyperparams={},
    )
    assert render.error is None
    assert render.grad_html == _NO_GRADIENT_HTML


def test_compute_render_missing_weight_is_error() -> None:
    session, _ = make_session()
    panel = _first_weight_panel(session)
    render = panel.compute_render(
        {}, {}, optimizer_state={}, optimizer_hyperparams={}
    )
    assert render.error == "no weights captured yet"
    assert not render.weight_html


def test_compute_snapshot_renders_runs_off_the_event_loop() -> None:
    """The strip render from a published snapshot is the unit `tick` hands to
    `asyncio.to_thread`; running it through a thread must produce the rendered
    content the loop would. The Refresh button no longer renders here — it asks
    the training thread to publish a snapshot that this same path then renders.

    This mirrors main_page's `tick`, which offloads `_compute_frame` so the
    CPU-heavy rendering never blocks NiceGUI's websocket keepalive.
    """
    session, _ = make_session()
    panel = _first_weight_panel(session)
    snap = _make_snapshot(
        "train", 0, 0, weights={panel.name: torch.randn(*panel._shape)}
    )

    async def run() -> list[_PanelRender]:
        return await asyncio.to_thread(_compute_snapshot_renders, [panel], snap)

    renders = asyncio.run(run())
    assert len(renders) == 1
    assert renders[0].error is None
    assert renders[0].weight_html


def test_weight_graphs_href_carries_view_scroll_and_watch() -> None:
    # A real `href` (not an `on_click` navigate) keeps the button
    # middle-clickable; `watch=1` moves the "start watching" side effect to
    # the stats page so the link still lands on data in the watched scope.
    from nansense.ui.weights_page import _weight_graphs_href

    assert (
        _weight_graphs_href("fc1")
        == "/stats?layer=fc1&view=graphs&scroll=weights&watch=1"
    )
    assert _weight_graphs_href("odd layer").startswith("/stats?layer=odd%20layer&")
