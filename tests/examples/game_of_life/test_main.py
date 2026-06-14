"""Tests for the Game-of-Life example: rule, model, and a training smoke step."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.fx
from torch import nn

from examples.game_of_life import main as main_module
from examples.game_of_life.life import GameOfLifeDataset, life_step, life_steps
from examples.game_of_life.model import LifeNet


def test_blinker_oscillates_with_period_two() -> None:
    """A vertical 3-cell blinker flips to horizontal and back (period 2)."""
    board = torch.zeros(1, 1, 5, 5)
    board[0, 0, 1:4, 2] = 1.0  # vertical bar

    horizontal = life_step(board)
    expected_horizontal = torch.zeros(1, 1, 5, 5)
    expected_horizontal[0, 0, 2, 1:4] = 1.0
    assert torch.equal(horizontal, expected_horizontal)

    # Two steps return to the original phase.
    assert torch.equal(life_steps(board, 2), board)


def test_block_is_a_still_life() -> None:
    """A 2x2 live block is stable under any number of steps."""
    board = torch.zeros(1, 1, 6, 6)
    board[0, 0, 2:4, 2:4] = 1.0
    assert torch.equal(life_step(board), board)
    assert torch.equal(life_steps(board, 4), board)


def test_toroidal_wrap() -> None:
    """A blinker straddling the wrap edge still oscillates with period 2,
    confirming the boundary is toroidal rather than zero-padded."""
    board = torch.zeros(1, 1, 5, 5)
    # Vertical bar at rows 4, 0, 1 (wrapping over the top/bottom edge), col 2.
    board[0, 0, 4, 2] = 1.0
    board[0, 0, 0, 2] = 1.0
    board[0, 0, 1, 2] = 1.0
    expected = torch.zeros(1, 1, 5, 5)
    expected[0, 0, 0, 1:4] = 1.0  # centred on the wrapped row 0
    assert torch.equal(life_step(board), expected)
    assert torch.equal(life_steps(board, 2), board)


def test_dataset_is_deterministic_and_shaped() -> None:
    a = GameOfLifeDataset(size=8, board_size=12, steps=2, density=0.4, seed=3)
    b = GameOfLifeDataset(size=8, board_size=12, steps=2, density=0.4, seed=3)
    board, target = a[0]
    assert board.shape == (1, 12, 12)
    assert target.shape == (1, 12, 12)
    assert set(board.unique().tolist()) <= {0.0, 1.0}
    # Same seed -> identical boards and targets; the target is the rule applied.
    assert torch.equal(board, b[0][0])
    assert torch.equal(target, life_steps(board.unsqueeze(0), 2).squeeze(0))


@pytest.mark.parametrize("board_size", [8, 16])
@pytest.mark.parametrize("steps", [1, 3])
def test_build_model_shape_and_fx_traceable(board_size: int, steps: int) -> None:
    model = main_module.build_model(channels=8, steps=steps)
    assert isinstance(model, LifeNet)
    x = torch.randn(2, 1, board_size, board_size)
    out = model(x)
    assert out.shape == (2, 1, board_size, board_size)
    # nansense traces the graph to name layers; tracing must succeed.
    torch.fx.symbolic_trace(model)


def test_default_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented default is kept modest for low GPU memory."""
    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert main_module.parse_args().batch_size == 128


def test_training_reduces_bce_loss() -> None:
    """A few optimization steps on a tiny synthetic batch must reduce the
    per-cell BCE loss (inline BCEWithLogitsLoss analogue of the shared helper)."""
    torch.manual_seed(0)
    dataset = GameOfLifeDataset(size=16, board_size=8, steps=1, density=0.3, seed=0)
    x = dataset.boards
    y = dataset.targets

    model = LifeNet(channels=8, depth=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    initial = criterion(model(x), y).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final = criterion(model(x), y).item()

    assert final < initial
