"""Per-input display configuration: normalization stats and channel transforms.

The UI denormalizes and renders one model input at a time (the input-image
pane, plus the experiment/stats input crops). Each of `input_mean`,
`input_std` and `input_transform` passed to `nansense.start` / `serve` is
either a single value applied to *every* input, or a `dict` keyed by input
name when a multi-input model needs different display handling per input.

`resolve_per_input` collapses either form to the value for one input name, so
the rest of the code never has to care which form the user gave, and
`InputDisplay` bundles all three for a caller that resolves them repeatedly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from torch import Tensor

# Normalization stats: `x * std + mean` denormalizes a `[0, 1]`-trained input.
MeanStd = tuple[float, ...]
# Maps a `[B, C, H, W]` model input to a displayable `[B, 1|3, H, W]` tensor in
# `[0, 1]`, preserving `B` and `H × W` (see `nansense.ui.render`).
InputTransform = Callable[[Tensor], Tensor]

_T = TypeVar("_T")


def resolve_per_input(config: _T | dict[str, _T] | None, name: str | None) -> _T | None:
    """The config for input `name`: a dict's entry, or the shared single value.

    `None` when there is no config, no input name, or the dict has no entry for
    `name`. A non-dict `config` (a single `MeanStd` tuple or `InputTransform`
    callable) applies to every input.
    """
    if config is None or name is None:
        return None
    if isinstance(config, dict):
        return cast("dict[str, _T]", config).get(name)
    return cast("_T", config)


class InputDisplay:
    """All three display configs together, resolvable per input name.

    `serve` receives them from the training script and the pages resolve them
    for whichever input they are showing. A caller that renders several inputs
    — the MCP server's image tools, which take an input name per call — would
    otherwise repeat the same three `resolve_per_input` calls everywhere.
    """

    def __init__(
        self,
        *,
        mean: MeanStd | dict[str, MeanStd] | None = None,
        std: MeanStd | dict[str, MeanStd] | None = None,
        transform: InputTransform | dict[str, InputTransform] | None = None,
    ) -> None:
        self._mean = mean
        self._std = std
        self._transform = transform

    def stats(self, name: str | None) -> tuple[MeanStd | None, MeanStd | None]:
        """The `(mean, std)` denormalization pair for input `name`."""
        return resolve_per_input(self._mean, name), resolve_per_input(self._std, name)

    def transform(self, name: str | None) -> InputTransform | None:
        """The displayable-image transform for input `name`, if any."""
        return resolve_per_input(self._transform, name)
