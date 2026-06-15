"""Tests for the shared example plumbing in `examples.common` — here the
`--dtype` flag helpers and the autocast context they feed."""

from __future__ import annotations

import argparse
import importlib
import sys

import pytest
import torch
from torch import nn

from examples import common

# Every training entrypoint must wire the shared --dtype flag (fp32 default).
_ENTRYPOINTS = [
    "examples.standard.main",
    "examples.audio_keywords.main",
    "examples.depth_make3d.main",
    "examples.game_of_life.main",
    "examples.pytorch_lightning.main",
]


@pytest.mark.parametrize(
    ("name", "expected"),
    [("fp32", None), ("fp16", torch.float16), ("bf16", torch.bfloat16)],
)
def test_amp_dtype_from_name(name: str, expected: torch.dtype | None) -> None:
    assert common.amp_dtype_from_name(name) == expected


def test_add_dtype_arg_default_and_choices() -> None:
    """The shared flag defaults to fp32 and accepts exactly the three names."""
    parser = argparse.ArgumentParser()
    common.add_dtype_arg(parser)

    assert parser.parse_args([]).dtype == "fp32"
    assert parser.parse_args(["--dtype", "fp16"]).dtype == "fp16"
    assert parser.parse_args(["--dtype", "bf16"]).dtype == "bf16"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dtype", "int8"])


def test_dtype_help_documents_no_grad_scaling() -> None:
    """The deliberate 'no GradScaler' caveat must stay in the user-facing help."""
    assert "GradScaler" in common.DTYPE_HELP


def test_autocast_fp32_is_passthrough() -> None:
    """fp32 -> None disables autocast, so the forward stays in fp32."""
    layer = nn.Linear(4, 4)
    with common.autocast(torch.device("cpu"), None):
        out = layer(torch.randn(2, 4))
    assert out.dtype == torch.float32


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_autocast_casts_forward_but_keeps_weights_fp32(amp_dtype: torch.dtype) -> None:
    """Under autocast the forward runs in the low-precision dtype while the
    parameters stay fp32 — the invariant the --dtype flag promises."""
    layer = nn.Linear(4, 4)
    with common.autocast(torch.device("cpu"), amp_dtype):
        out = layer(torch.randn(2, 4))
    assert out.dtype == amp_dtype
    assert layer.weight.dtype == torch.float32


@pytest.mark.parametrize("module_name", _ENTRYPOINTS)
def test_every_entrypoint_wires_dtype_flag(
    module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each training script must expose `--dtype`, defaulting to fp32 and
    accepting the lower-precision names."""
    main_module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert main_module.parse_args().dtype == "fp32"
    monkeypatch.setattr(sys, "argv", ["main.py", "--dtype", "fp16"])
    assert main_module.parse_args().dtype == "fp16"
