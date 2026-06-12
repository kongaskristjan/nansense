"""Tests for the render cache and frame computation in nansense.ui.main_page."""

from __future__ import annotations

import pytest
import torch

from nansense.probe import ProbeResult
from nansense.ui.common import _strip_html
from nansense.ui.main_page import (
    _PROBE_NO_GRADIENTS_HTML,
    _RenderCache,
    _compute_frame,
    _display_batch_size,
    _input_img_src,
)
from nansense.ui.render import StripRender, image_mime, render_image, render_strip
from tests.nansense.helpers import _frame_snapshot, _make_snapshot


def test_input_img_src_is_a_data_uri() -> None:
    png = render_image(torch.rand(1, 3, 16, 16), sample_idx=0)
    assert png is not None
    src = _input_img_src(png)
    assert src.startswith(f"data:{image_mime()};base64,")
    assert _input_img_src(None) == ""


def test_render_cache_renders_once_per_key() -> None:
    cache = _RenderCache()
    snap = _frame_snapshot()
    calls = 0

    def render() -> str:
        nonlocal calls
        calls += 1
        return "html"

    assert cache.get_or_render(snap, ("a", "act", 0), render) == "html"
    assert cache.get_or_render(snap, ("a", "act", 0), render) == "html"
    assert calls == 1
    cache.get_or_render(snap, ("a", "act", 1), render)
    assert calls == 2  # a different sample is a different entry


def test_render_cache_resets_on_new_snapshot() -> None:
    cache = _RenderCache()
    calls = 0

    def render() -> str:
        nonlocal calls
        calls += 1
        return "html"

    cache.get_or_render(_frame_snapshot(), ("a", "act", 0), render)
    cache.get_or_render(_frame_snapshot(), ("a", "act", 0), render)
    assert calls == 2  # a new snapshot object invalidates the old entries


def test_compute_frame_renders_strips_and_input() -> None:
    snap = _frame_snapshot()
    rendered, input_html = _compute_frame(
        ["x", "conv", "missing"],
        snap,
        None,
        0,
        input_name="x",
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    act, grad = rendered["conv"]
    assert "<img" in act and "<img" in grad
    assert rendered["x"][1] == ""  # the input has no gradient captured
    assert rendered["missing"] == ("", "")
    assert input_html.startswith("data:")


def test_compute_frame_reuses_cache_within_a_snapshot() -> None:
    cache = _RenderCache()
    snap = _frame_snapshot()

    def frame(sample_idx: int) -> tuple[dict[str, tuple[str, str]], str]:
        return _compute_frame(
            ["conv"],
            snap,
            None,
            sample_idx,
            input_name="x",
            input_mean=None,
            input_std=None,
            cache=cache,
        )

    first, input_first = frame(0)
    again, input_again = frame(0)
    # Cache hits return the exact same strings, not re-rendered copies.
    assert again["conv"][0] is first["conv"][0]
    assert input_again is input_first
    other_sample, _ = frame(1)
    assert other_sample["conv"][0] is not first["conv"][0]


def _frame_probe() -> ProbeResult:
    return ProbeResult(
        input=torch.rand(2, 3, 4, 4),
        activations={"x": torch.rand(2, 3, 4, 4), "conv": torch.rand(2, 2, 4, 4)},
        mode="eval",
    )


def test_compute_frame_prefers_probe_over_snapshot() -> None:
    probe = _frame_probe()
    rendered, input_html = _compute_frame(
        ["x", "conv", "missing"],
        _frame_snapshot(),
        probe,
        0,
        input_name="x",
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    act, grad = rendered["conv"]
    assert "<img" in act
    # Probes are forward-only: every gradient strip is the placeholder note.
    assert grad == _PROBE_NO_GRADIENTS_HTML
    assert rendered["missing"][0] == ""
    assert input_html.startswith("data:")


def _frame_probe_perturbed() -> ProbeResult:
    base = torch.rand(2, 3, 4, 4)
    perturbed = base.clone()
    perturbed[0, :, 1, 1] = 5.0
    return ProbeResult(
        input=base,
        activations={"x": base, "conv": torch.rand(2, 2, 4, 4)},
        mode="eval",
        perturbed_input=perturbed,
        perturbed_activations={"x": perturbed, "conv": torch.rand(2, 2, 4, 4)},
    )


@pytest.mark.parametrize("compare", [False, True])
def test_compute_probe_frame_renders_perturbed_or_diff(compare: bool) -> None:
    probe = _frame_probe_perturbed()
    rendered, input_src = _compute_frame(
        ["x", "conv"],
        None,
        probe,
        0,
        compare=compare,
        input_name="x",
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    assert "<img" in rendered["x"][0]
    assert "<img" in rendered["conv"][0]
    assert rendered["x"][1] == _PROBE_NO_GRADIENTS_HTML
    assert input_src.startswith("data:")


def test_compute_probe_frame_diff_differs_from_perturbed_view() -> None:
    probe = _frame_probe_perturbed()
    cache = _RenderCache()

    def frame(compare: bool) -> dict[str, tuple[str, str]]:
        rendered, _ = _compute_frame(
            ["x"],
            None,
            probe,
            0,
            compare=compare,
            input_name="x",
            input_mean=None,
            input_std=None,
            cache=cache,
        )
        return rendered

    # The diff view (perturbed − original: zero except one pixel) renders
    # different pixels than the perturbed-activations view.
    assert frame(True)["x"][0] != frame(False)["x"][0]


def test_compute_probe_frame_diff_without_perturbations_renders_zeros() -> None:
    """Compare mode on a perturbation-free probe still shows the diff view:
    an all-zero diff (a white strip), not a fallback to the base view."""
    base = torch.rand(2, 2, 4, 4)
    probe = ProbeResult(
        input=torch.rand(2, 3, 4, 4),
        activations={"conv": base},
        mode="eval",
    )
    rendered, _ = _compute_frame(
        ["conv"],
        None,
        probe,
        0,
        compare=True,
        input_name="x",
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    expected = _strip_html(render_strip(torch.zeros_like(base), 0))
    assert rendered["conv"][0] == expected


def test_compute_snapshot_frame_compare_renders_zero_diff() -> None:
    """Compare mode with no probe at all: activation strips show the all-zero
    diff while gradient strips keep their normal view."""
    snap = _frame_snapshot()
    rendered, _ = _compute_frame(
        ["conv"],
        snap,
        None,
        0,
        compare=True,
        input_name="x",
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    act_expected = _strip_html(
        render_strip(torch.zeros_like(snap.activations["conv"]), 0)
    )
    grad_expected = _strip_html(
        render_strip(snap.activation_gradients["conv"], 0)
    )
    assert rendered["conv"][0] == act_expected
    assert rendered["conv"][1] == grad_expected


def test_display_batch_size_prefers_probe() -> None:
    snap = _frame_snapshot()  # batch size 2
    probe = ProbeResult(
        input=torch.rand(5, 3, 4, 4), activations={}, mode="eval"
    )
    assert _display_batch_size(snap, probe) == 5
    assert _display_batch_size(snap, None) == 2
    assert _display_batch_size(None, None) is None


def test_compute_frame_renders_more_layers_than_pool_workers() -> None:
    # Exercise the render pool's queueing: more layers than max_workers.
    names = [f"l{i}" for i in range(20)]
    snap = _make_snapshot(
        "train", 0, 0, activations={name: torch.rand(1, 2, 4, 4) for name in names}
    )
    rendered, _ = _compute_frame(
        names,
        snap,
        None,
        0,
        input_name=None,
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    assert set(rendered) == set(names)
    assert all("<img" in rendered[name][0] for name in names)


def test_compute_frame_empty_layer_does_not_drop_the_frame() -> None:
    # A layer with an empty activation must not abort the others: the good
    # layers still produce strips, the empty one renders as a hidden (blank)
    # strip — one bad layer can't drop the whole frame for the snapshot.
    snap = _make_snapshot(
        "train",
        0,
        0,
        activations={
            "good": torch.rand(2, 2, 4, 4),
            "empty": torch.zeros(2, 0, 4, 4),
            "also_good": torch.rand(2, 3, 4, 4),
        },
    )
    rendered, _ = _compute_frame(
        ["good", "empty", "also_good"],
        snap,
        None,
        0,
        input_name=None,
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    assert "<img" in rendered["good"][0]
    assert "<img" in rendered["also_good"][0]
    assert rendered["empty"] == ("", "")


def test_compute_frame_raising_layer_does_not_drop_the_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Defense in depth: even a layer whose render *raises* (a residual bug
    # `render_strip`'s guards don't catch) must yield blank strips rather than
    # abort the fan-out and drop every other layer's frame. The "bad" tensor
    # is tagged by identity so only its render blows up.
    import nansense.ui.main_page as main_page

    bad = torch.rand(2, 7, 4, 4)
    real_render_strip = main_page.render_strip

    def flaky_render_strip(
        tensor: torch.Tensor | None,
        sample_idx: int,
        *,
        input_hw: tuple[int, int] | None = None,
    ) -> StripRender | None:
        if tensor is bad:
            raise RuntimeError("boom")
        return real_render_strip(tensor, sample_idx, input_hw=input_hw)

    monkeypatch.setattr(main_page, "render_strip", flaky_render_strip)
    snap = _make_snapshot(
        "train",
        0,
        0,
        activations={
            "good": torch.rand(2, 2, 4, 4),
            "bad": bad,
            "also_good": torch.rand(2, 3, 4, 4),
        },
    )
    rendered, _ = _compute_frame(
        ["good", "bad", "also_good"],
        snap,
        None,
        0,
        input_name=None,
        input_mean=None,
        input_std=None,
        cache=_RenderCache(),
    )
    assert "<img" in rendered["good"][0]
    assert "<img" in rendered["also_good"][0]
    assert rendered["bad"] == ("", "")
