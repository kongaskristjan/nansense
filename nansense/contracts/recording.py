"""Frozen page configurations passed from UI/MCP adapters to frame renderers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Literal, cast, final

from nansense.input_config import InputTransform
from nansense.patches import PATCH_TYPES, PatchType

ValueMode = Literal["unchanged", "abs", "square"]
AxisRole = Literal["x", "y", "tile", "index"]


def patch_types(values: Iterable[str]) -> tuple[PatchType, ...]:
    result = tuple(values)
    if any(value not in PATCH_TYPES for value in result):
        raise ValueError("unknown patch grid type")
    return cast(tuple[PatchType, ...], result)


@dataclass(frozen=True)
class MainView:
    page: ClassVar[Literal["main"]] = "main"
    layers: tuple[str, ...] = ()
    sample_idx: int = 0
    input_name: str | None = None
    input_mean: tuple[float, ...] | None = None
    input_std: tuple[float, ...] | None = None
    input_transform: InputTransform | None = None
    render_average: bool = False
    render_values: ValueMode = "unchanged"


@dataclass(frozen=True)
class WeightPanelConfig:
    name: str
    roles: tuple[AxisRole, ...] = ()
    indices: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_values(
        cls,
        name: str,
        roles: Iterable[str],
        indices: Iterable[tuple[int, int]],
    ) -> WeightPanelConfig:
        choices = tuple(roles)
        if any(role not in ("x", "y", "tile", "index") for role in choices):
            raise ValueError("unknown weight axis role")
        return cls(name, cast(tuple[AxisRole, ...], choices), tuple(indices))


@dataclass(frozen=True)
class WeightsView:
    page: ClassVar[Literal["weights"]] = "weights"
    layer: str = ""
    panels: tuple[WeightPanelConfig, ...] = ()


@dataclass(frozen=True)
class HistogramView:
    page: ClassVar[Literal["watch_histogram"]] = "watch_histogram"
    layers: tuple[str, ...] = ()
    phase: str = ""
    log_x: bool = False
    log_y: bool = False


@dataclass(frozen=True)
class PatchView:
    page: ClassVar[Literal["watch_minmax"]] = "watch_minmax"
    layers: tuple[str, ...] = ()
    phase: str = ""
    grids: tuple[PatchType, ...] = ()
    heatmap: bool = False
    input_mean: tuple[float, ...] | None = None
    input_std: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ExperimentView:
    page: ClassVar[Literal["experiment"]] = "experiment"
    layer: str = ""
    seq: int = 0
    auto_key: str = ""
    input_mean: tuple[float, ...] | None = None
    input_std: tuple[float, ...] | None = None
    overlay: bool = False


ViewConfig = MainView | WeightsView | HistogramView | PatchView | ExperimentView


@final
@dataclass(frozen=True)
class RecordedView:
    """A recording's identity and immutable configuration for one view kind."""

    key: str
    label: str
    config: ViewConfig

    @property
    def page(
        self,
    ) -> Literal["main", "weights", "watch_histogram", "watch_minmax", "experiment"]:
        return self.config.page

    @property
    def auto_key(self) -> str:
        return self.config.auto_key if isinstance(self.config, ExperimentView) else ""
