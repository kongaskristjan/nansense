"""Conway's Game of Life rule and a synthetic dataset of (board, K-steps-ahead).

The rule is computed with a fixed 3x3 neighbour-count convolution (an all-ones
kernel with a zeroed centre) under *toroidal* (wrap-around) boundaries via
circular padding, then the classic birth/survive thresholds:

  * a dead cell with exactly 3 live neighbours becomes alive (birth);
  * a live cell with 2 or 3 live neighbours stays alive (survival);
  * every other cell is dead next step.

Applying this `steps` (K) times gives the prediction target. Everything is
deterministic: boards are drawn once in ``__init__`` from a seeded generator,
so epoch replay and NaNsense time travel see identical data.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

# 3x3 all-ones-minus-centre neighbour-count kernel, shape [1, 1, 3, 3].
_NEIGHBOR_KERNEL: Tensor = torch.tensor(
    [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
).view(1, 1, 3, 3)


def life_step(board: Tensor) -> Tensor:
    """Advance a batch of boards one Game-of-Life step under toroidal wrap.

    `board` is a float tensor in {0, 1} of shape ``[B, 1, H, W]``; the returned
    tensor has the same shape and dtype/device. Circular padding implements the
    wrap-around (toroidal) boundary, so the rule is translation-invariant.
    """
    kernel = _NEIGHBOR_KERNEL.to(board.dtype).to(board.device)
    padded = torch.nn.functional.pad(board, (1, 1, 1, 1), mode="circular")
    neighbors = torch.nn.functional.conv2d(padded, kernel)
    alive = board > 0.5
    # Birth on exactly 3 neighbours; survival on 2 or 3 neighbours.
    next_alive = (neighbors == 3) | (alive & (neighbors == 2))
    return next_alive.to(board.dtype)


def life_steps(board: Tensor, steps: int) -> Tensor:
    """Apply :func:`life_step` ``steps`` (>= 0) times."""
    out = board
    for _ in range(steps):
        out = life_step(out)
    return out


class GameOfLifeDataset(Dataset[tuple[Tensor, Tensor]]):
    """Finite, map-style dataset of random binary boards and their futures.

    ``size`` random boards of shape ``[1, board_size, board_size]`` are sampled
    once (each cell alive with probability ``density``) from a generator seeded
    with ``seed``, making the dataset fully deterministic. Each random board is
    then advanced one *silent* Game-of-Life step to form the model input: a
    single step is enough for sparse random noise to die off and local structure
    (blocks, blinkers, glider fragments) to emerge, so the input looks less
    random than a raw draw. ``__getitem__`` returns ``(board, target)`` where
    ``board`` is that one-step input and ``target`` is it advanced a further
    ``steps`` (K) Game-of-Life steps under toroidal boundaries.
    """

    def __init__(
        self,
        size: int,
        board_size: int = 32,
        steps: int = 2,
        density: float = 0.3,
        seed: int = 0,
    ) -> None:
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        self.steps = steps
        generator = torch.Generator().manual_seed(seed)
        # [size, 1, H, W] float in {0, 1}; one fixed draw for the whole dataset.
        probs = torch.rand(size, 1, board_size, board_size, generator=generator)
        random_boards = (probs < density).float()
        # One silent GoL step turns the sparse random draw into something with
        # local structure, so inputs look less random; targets advance further.
        self.boards: Tensor = life_step(random_boards)
        # Precompute targets once: cheap, and keeps __getitem__ trivially fast.
        self.targets: Tensor = life_steps(self.boards, steps)

    def __len__(self) -> int:
        return self.boards.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.boards[index], self.targets[index]
