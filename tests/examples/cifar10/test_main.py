"""Tests for the CIFAR10 example entrypoint helpers."""

from __future__ import annotations

import io

import pytest

from examples.cifar10 import main as main_module


def test_enable_line_buffering_sets_line_buffering(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.TextIOWrapper(io.BytesIO(), line_buffering=False)
    monkeypatch.setattr(main_module.sys, "stdout", stream)

    main_module.enable_line_buffering()

    assert stream.line_buffering is True


def test_enable_line_buffering_tolerates_non_textiowrapper_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture/proxy stdout (not a TextIOWrapper) must be left untouched, not raise."""
    monkeypatch.setattr(main_module.sys, "stdout", io.StringIO())

    main_module.enable_line_buffering()  # must be a no-op, not an error
