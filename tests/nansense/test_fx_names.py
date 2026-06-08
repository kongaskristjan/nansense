"""Tests for scope-qualified fx node naming."""

from __future__ import annotations

import torch
from torch import fx, nn

from nansense.fx_names import friendly_names


class Block(nn.Module):
    """A submodule with two functional relus and one functional add."""

    def __init__(self) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(4)
        self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(x))
        out = torch.relu(self.bn2(out))
        return self.conv(out) + x


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.block = Block()
        self.head_bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.stem(x))
        return torch.relu(self.head_bn(x)).flatten(1)


def _names(model: nn.Module) -> list[str]:
    return list(friendly_names(fx.symbolic_trace(model).graph).values())


def test_repeated_op_in_a_scope_is_numbered_by_submodule() -> None:
    names = _names(Net())
    # The two relus inside `block` are disambiguated by a per-scope index.
    assert "block.relu1" in names
    assert "block.relu2" in names


def test_single_op_in_a_scope_is_not_numbered() -> None:
    names = _names(Net())
    # `block` has exactly one functional add, so it stays unsuffixed.
    assert "block.add" in names
    assert "block.add1" not in names


def test_root_scope_op_has_no_prefix() -> None:
    names = _names(Net())
    # The head relu and flatten run in the top-level forward (no submodule).
    assert "relu" in names
    assert "flatten" in names


def test_module_calls_keep_their_dotted_path() -> None:
    names = _names(Net())
    assert "stem" in names
    assert "block.bn1" in names
    assert "block.bn2" in names
    assert "block.conv" in names
    assert "head_bn" in names


def test_placeholder_and_output_keep_fx_names() -> None:
    names = _names(Net())
    assert "x" in names
    assert "output" in names


def test_all_names_are_unique() -> None:
    names = _names(Net())
    assert len(names) == len(set(names))
