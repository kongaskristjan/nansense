"""Tests for the numerical-error debugger (nansense.debugger) and its
integration into the session's batch lifecycle."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense import debugger
from nansense.debugger import (
    DebugError,
    DebugSettings,
    LayerReport,
    dtype_band,
    merged,
    run_checks,
    without_category,
)
from nansense.schedule import BatchPosition
from nansense.session import Mode
from tests.nansense.helpers import make_position, run_in_thread, train_step

_POS: BatchPosition = make_position("train", 0, 3)


def _check(
    settings: DebugSettings | None = None,
    *,
    activations: dict[str, Tensor] | None = None,
    activation_grads: dict[str, Tensor] | None = None,
    weight_grads: dict[str, Tensor] | None = None,
    layer_weights: dict[str, list[str]] | None = None,
) -> DebugError | None:
    """`run_checks` with empty-dict defaults for the unsupplied tensor maps."""
    return run_checks(
        settings if settings is not None else DebugSettings(),
        position=_POS,
        activations=activations or {},
        activation_grads=activation_grads or {},
        weight_grads=weight_grads or {},
        layer_weights=layer_weights or {},
    )


# --- NaN / Inf ------------------------------------------------------------


def test_single_nan_in_activation_trips() -> None:
    error = _check(
        activations={"l": torch.tensor([1.0, float("nan"), 2.0])},
        layer_weights={"l": []},
    )
    assert error is not None
    assert error.reasons == ("nan",)
    assert [r.layer for r in error.layers] == ["l"]
    assert error.layers[0].nan == 1 / 3


def test_inf_in_weight_grad_mapped_via_layer_weights() -> None:
    error = _check(
        activations={"conv": torch.zeros(2)},
        weight_grads={"conv.weight": torch.tensor([1.0, float("inf")])},
        layer_weights={"conv": ["conv.weight"]},
    )
    assert error is not None
    assert error.reasons == ("inf",)
    # 1 inf out of 2 (activation) + 2 (weight grad) = 4 scanned elements.
    assert error.layers[0].inf == 1 / 4


def test_clean_tensors_return_none() -> None:
    assert (
        _check(
            activations={"l": torch.randn(10)},
            activation_grads={"l": torch.randn(10)},
            layer_weights={"l": []},
        )
        is None
    )


def test_nan_inf_check_off_ignores_nan() -> None:
    error = _check(
        DebugSettings(check_nan_inf=False),
        activations={"l": torch.tensor([float("nan")])},
        activation_grads={"l": torch.randn(5)},
        layer_weights={"l": []},
    )
    assert error is None


# --- Underflow / overflow (dtype-aware bands, summed-|value| metric) -------


def test_subnormal_fp16_gradients_trip_underflow() -> None:
    # 1e-5 < finfo(float16).tiny (~6.1e-5) and > 0 -> subnormal band.
    grad = torch.full((100,), 1e-5, dtype=torch.float16)
    error = _check(activation_grads={"l": grad}, layer_weights={"l": []})
    assert error is not None
    assert error.reasons == ("underflow",)
    assert error.layers[0].underflow == 1.0


def test_subnormal_fp32_gradients_trip_underflow() -> None:
    grad = torch.full((50,), 1e-40, dtype=torch.float32)
    error = _check(activation_grads={"l": grad}, layer_weights={"l": []})
    assert error is not None
    assert error.reasons == ("underflow",)


def test_saturating_fp16_gradients_trip_overflow() -> None:
    grad = torch.full((100,), float(torch.finfo(torch.float16).max), dtype=torch.float16)
    error = _check(activation_grads={"l": grad}, layer_weights={"l": []})
    assert error is not None
    assert error.reasons == ("overflow",)
    assert error.layers[0].overflow == 1.0


def test_underflow_metric_is_summed_absolute_value() -> None:
    """Sum-of-|value|: a handful of subnormals next to normal-magnitude values
    contribute almost nothing to the sum, so underflow stays well under the
    threshold even at 50% by count — the chosen (intentionally conservative)
    semantics."""
    grad = torch.cat(
        [
            torch.full((50,), 1e-5, dtype=torch.float16),  # subnormal
            torch.full((50,), 1.0, dtype=torch.float16),  # normal
        ]
    )
    assert _check(activation_grads={"l": grad}, layer_weights={"l": []}) is None


def test_overflow_threshold_respected() -> None:
    maxv = float(torch.finfo(torch.float16).max)
    # 5 saturating values dominate the summed magnitude (~0.9997 of it).
    grad = torch.cat(
        [
            torch.full((5,), maxv, dtype=torch.float16),
            torch.full((95,), 1.0, dtype=torch.float16),
        ]
    )
    assert _check(
        DebugSettings(threshold=0.5), activation_grads={"l": grad}, layer_weights={"l": []}
    ) is not None
    assert _check(
        DebugSettings(threshold=0.9999),
        activation_grads={"l": grad},
        layer_weights={"l": []},
    ) is None


def test_under_over_check_off_ignores_subnormals() -> None:
    grad = torch.full((100,), 1e-5, dtype=torch.float16)
    assert (
        _check(
            DebugSettings(check_under_over=False),
            activation_grads={"l": grad},
            layer_weights={"l": []},
        )
        is None
    )


# --- Dtype band: recorded on the report, finfo-backed ----------------------


def test_report_records_grad_dtype() -> None:
    # The report carries the scanned gradient's dtype so the UI can show the
    # band edges (the histogram/dialog read `finfo.tiny` / `finfo.max`).
    grad = torch.full((100,), 1e-5, dtype=torch.float16)
    error = _check(activation_grads={"l": grad}, layer_weights={"l": []})
    assert error is not None
    assert error.layers[0].dtype is torch.float16


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_dtype_band_matches_finfo(dtype: torch.dtype) -> None:
    finfo = torch.finfo(dtype)
    assert dtype_band(dtype) == (finfo.tiny, finfo.max)


# --- merged: accumulate later detections into the standing banner ----------


def test_merged_unions_reasons_and_keeps_first_position() -> None:
    first = DebugError(
        position=make_position("train", 0, 1),
        reasons=("nan",),
        checks_used=(debugger.NAN_INF,),
        layers=(LayerReport("a", nan=0.5, inf=0.0, underflow=0.0, overflow=0.0),),
    )
    later = DebugError(
        position=make_position("train", 2, 4),
        reasons=("underflow",),
        checks_used=(debugger.UNDER_OVER,),
        layers=(
            LayerReport(
                "b", nan=0.0, inf=0.0, underflow=0.3, overflow=0.0,
                dtype=torch.float16,
            ),
        ),
    )
    out = merged(first, later)
    # Reasons/checks union (in fixed order); the *first* error's position wins.
    assert out.reasons == ("nan", "underflow")
    assert out.checks_used == (debugger.NAN_INF, debugger.UNDER_OVER)
    assert out.position == first.position
    assert [r.layer for r in out.layers] == ["a", "b"]


def test_merged_takes_worst_fraction_per_layer() -> None:
    first = DebugError(
        position=_POS,
        reasons=("underflow",),
        checks_used=(debugger.UNDER_OVER,),
        layers=(
            LayerReport(
                "a", nan=0.0, inf=0.0, underflow=0.2, overflow=0.0,
                dtype=torch.float16,
            ),
        ),
    )
    later = DebugError(
        position=make_position("train", 1, 0),
        reasons=("underflow",),
        checks_used=(debugger.UNDER_OVER,),
        layers=(
            LayerReport(
                "a", nan=0.0, inf=0.0, underflow=0.7, overflow=0.0,
                dtype=torch.float16,
            ),
        ),
    )
    out = merged(first, later)
    # A layer seen in both keeps its largest observed share, and the dtype
    # carries through.
    assert len(out.layers) == 1
    assert out.layers[0].underflow == 0.7
    assert out.layers[0].dtype is torch.float16


# --- Multi-reason aggregation, checks_used, columns ------------------------


def test_both_categories_aggregate_and_record_checks_used() -> None:
    error = _check(
        activations={"a": torch.tensor([float("nan")])},
        activation_grads={"b": torch.full((100,), 1e-5, dtype=torch.float16)},
        layer_weights={"a": [], "b": []},
    )
    assert error is not None
    assert error.reasons == ("nan", "underflow")
    assert error.checks_used == (debugger.NAN_INF, debugger.UNDER_OVER)
    assert debugger.columns(error) == ["nan", "inf", "underflow", "overflow"]
    assert debugger.categories_present(error) == [
        debugger.NAN_INF,
        debugger.UNDER_OVER,
    ]
    assert {r.layer for r in error.layers} == {"a", "b"}


def test_columns_limited_to_checks_used() -> None:
    error = _check(
        DebugSettings(check_under_over=False),
        activations={"a": torch.tensor([float("inf")])},
        layer_weights={"a": []},
    )
    assert error is not None
    assert error.checks_used == (debugger.NAN_INF,)
    assert debugger.columns(error) == ["nan", "inf"]


def test_no_device_no_tensors_returns_none() -> None:
    assert _check(layer_weights={"l": []}) is None


def test_disabled_settings_short_circuit() -> None:
    assert (
        _check(
            DebugSettings(enabled=False),
            activations={"l": torch.tensor([float("nan")])},
            layer_weights={"l": []},
        )
        is None
    )


# --- without_category (the DISABLE button) --------------------------------


def _two_reason_error() -> DebugError:
    error = _check(
        activations={"a": torch.tensor([float("nan")])},
        activation_grads={"b": torch.full((100,), 1e-5, dtype=torch.float16)},
        layer_weights={"a": [], "b": []},
    )
    assert error is not None
    return error


def test_without_category_drops_nan_inf() -> None:
    trimmed = without_category(_two_reason_error(), debugger.NAN_INF)
    assert trimmed is not None
    assert trimmed.reasons == ("underflow",)
    assert trimmed.checks_used == (debugger.UNDER_OVER,)
    # The NaN-only layer drops out; the underflow layer remains.
    assert [r.layer for r in trimmed.layers] == ["b"]


def test_without_category_drops_under_over() -> None:
    trimmed = without_category(_two_reason_error(), debugger.UNDER_OVER)
    assert trimmed is not None
    assert trimmed.reasons == ("nan",)
    assert [r.layer for r in trimmed.layers] == ["a"]


def test_without_last_category_clears_banner() -> None:
    error = _check(
        activations={"a": torch.tensor([float("nan")])}, layer_weights={"a": []}
    )
    assert error is not None
    assert without_category(error, debugger.NAN_INF) is None


# --- DebugSettings --------------------------------------------------------


def test_any_check_requires_master_and_a_subcheck() -> None:
    assert DebugSettings().any_check()
    assert not DebugSettings(enabled=False).any_check()
    assert not DebugSettings(
        check_nan_inf=False, check_under_over=False
    ).any_check()


# --- Session integration --------------------------------------------------


class _NanNet(nn.Module):
    """A net whose forward poisons its output with NaN (fx-traceable)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x) * float("nan")


def test_session_stops_on_detected_nan() -> None:
    model = _NanNet()
    session = nansense.start(model, epochs=1, phases={"train": 3})

    def loop() -> None:
        for _ in range(3):
            with session.batch(phase="train", epoch=0):
                train_step(model)

    thread = run_in_thread(loop)
    try:
        assert session.wait_until_paused(timeout=5.0)
        error = session.debug_error
        assert error is not None
        assert "nan" in error.reasons
        # Detection stops training: the session is parked in STEP mode.
        assert session.mode is Mode.STEP
    finally:
        # The model produces NaN every batch, so resuming would immediately
        # re-stop; disable the debugger first so the loop can run to the end.
        session.set_debug_settings(enabled=False)
        session.detach()
        thread.join(timeout=5.0)
    assert not thread.is_alive()


def test_resume_keeps_error_and_rewind_clears_it() -> None:
    """A standing banner survives resuming (so later detections accumulate into
    it, request 3); only a time-travel rewind (new timeline) clears it."""
    model = nn.Linear(4, 3)
    session = nansense.start(model, epochs=1, phases={"train": 2})
    fake = DebugError(
        position=_POS,
        reasons=("nan",),
        checks_used=(debugger.NAN_INF,),
        layers=(LayerReport("l", nan=1.0, inf=0.0, underflow=0.0, overflow=0.0),),
    )

    # stop() resumes nothing, so the banner persists for inspection.
    session._debug_error = fake
    session.stop()
    assert session.debug_error is fake

    # Resuming no longer clears it — the warning stays up while training runs.
    session.detach()
    assert session.debug_error is fake

    # A time-travel rewind belongs to the new timeline and drops it.
    session._rewind_to_epoch(0)
    assert session.debug_error is None


def test_resume_after_error_runs_on_without_re_stopping() -> None:
    """Once stopped on a numerical issue, resuming runs to the end without
    pausing again; the standing banner persists and accumulates."""
    model = _NanNet()
    session = nansense.start(model, epochs=1, phases={"train": 3})
    session.set_debug_settings(interval=1)  # check every batch

    def loop() -> None:
        for _ in range(3):
            with session.batch(phase="train", epoch=0):
                train_step(model)

    thread = run_in_thread(loop)
    try:
        # First batch detects the NaN and stops.
        assert session.wait_until_paused(timeout=5.0)
        assert session.debug_error is not None
        pauses = session._pause_count
        # Resume: the run completes without pausing again, banner persists.
        session.detach()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert session.debug_error is not None
        assert session._pause_count == pauses  # no further error-stops
    finally:
        session.set_debug_settings(enabled=False)
        session.detach()
        thread.join(timeout=5.0)


def test_first_detection_prints_once_to_console(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The episode's onset prints one console line (for headless runs); later
    merges stay quiet."""
    model = _NanNet()
    session = nansense.start(model, epochs=1, phases={"train": 3})
    session.set_debug_settings(interval=1)

    def loop() -> None:
        for _ in range(3):
            with session.batch(phase="train", epoch=0):
                train_step(model)

    thread = run_in_thread(loop)
    try:
        assert session.wait_until_paused(timeout=5.0)
        # Resume: batches 2-3 still NaN, but they merge — no second print.
        session.detach()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
    finally:
        session.set_debug_settings(enabled=False)
        session.detach()
        thread.join(timeout=5.0)
    out = capsys.readouterr().out
    assert out.count("nansense: numerical issue detected") == 1
    assert "nan" in out.lower()


def test_rewind_prints_to_console(capsys: pytest.CaptureFixture[str]) -> None:
    model = nn.Linear(4, 3)
    session = nansense.start(model, epochs=3, phases={"train": 2})
    session._rewind_to_epoch(2)
    assert "time-traveled to the start of epoch 2" in capsys.readouterr().out


def test_disable_debug_check_toggles_setting_and_trims_banner() -> None:
    model = nn.Linear(4, 3)
    session = nansense.start(model, epochs=1, phases={"train": 2})
    session._debug_error = _two_reason_error()

    session.disable_debug_check(debugger.NAN_INF)
    assert session.debug_settings.check_nan_inf is False
    remaining = session.debug_error
    assert remaining is not None
    assert remaining.reasons == ("underflow",)

    session.disable_debug_check(debugger.UNDER_OVER)
    assert session.debug_settings.check_under_over is False
    assert session.debug_error is None


def test_set_debug_settings_updates_fields_and_resets_counter() -> None:
    model = nn.Linear(4, 3)
    session = nansense.start(model, epochs=1, phases={"train": 2})
    session._debug_counter = 7
    session.set_debug_settings(interval=5, threshold=0.25, check_under_over=False)
    settings = session.debug_settings
    assert settings.interval == 5
    assert settings.threshold == 0.25
    assert settings.check_under_over is False
    assert settings.check_nan_inf is True  # untouched
    assert session._debug_counter == 0

