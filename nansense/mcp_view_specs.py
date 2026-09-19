"""Typed view specifications for the MCP recording and snapshot tools.

`start_recording` and `save_snapshot` both draw one of the debugger's pages,
and each page takes its own arguments. A flat keyword bag would have to say so
in prose and hope the agent read it; these models say it in the JSON schema
instead — a union discriminated on `view`, so only the fields that view
actually consumes are offered, described and validated.

The two tools do not take the same set: a recording *registers* a re-running
experiment (`kind`, `params`), while a snapshot draws an experiment result that
was already published (`seq`). Every other view is shared.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

#: The keys of `experiments.EXPERIMENT_KINDS`, as a type. Shared with
#: `run_experiment`'s own `kind` so the two cannot drift apart.
ExperimentKind = Literal[
    "deep_dream", "gradcam", "neuron_gradient", "neuron_ig", "occlusion"
]

_OVERLAY_DOC = (
    "Blend the attribution over the input it explains instead of drawing it "
    "alongside. Inert for deep dream, which synthesizes an input."
)
_PHASE_DOC = (
    "Training phase whose accumulators to read, e.g. 'train'; omit for the "
    "newest phase with data."
)
_LAYERS_DOC = (
    "Layers to draw; omit for every layer whose watch accumulators are retained."
)


class LayersViewSpec(BaseModel):
    """The main page: per-channel activation and gradient strips."""

    view: Literal["layers"] = Field(
        default="layers",
        description="Selects this variant: the main page's per-channel strips.",
    )
    layers: list[str] | None = Field(
        default=None,
        description=(
            "Layers to draw, in model order; omit for the watched layers, which "
            "is what the page shows."
        ),
    )
    sample: int = Field(
        default=0, description="Zero-based index of the sample within the batch."
    )
    average: bool = Field(
        default=False,
        description="Collapse each strip's channels into one mean tile.",
    )
    values: Literal["unchanged", "abs", "square"] = Field(
        default="unchanged",
        description=(
            "Per-value transform before colouring: 'abs' or 'square' drop the "
            "sign for a magnitude picture, applied before `average`."
        ),
    )


class WeightsViewSpec(BaseModel):
    """The weights page: one layer's parameters, gradients and optimizer state."""

    view: Literal["weights"] = Field(
        default="weights",
        description="Selects this variant: the weights page for one layer.",
    )
    layer: str = Field(
        description="The one layer to draw; it must have parameters of its own."
    )


class HistogramsViewSpec(BaseModel):
    """The `/stats` page's histograms over the watch accumulators."""

    view: Literal["histograms"] = Field(
        default="histograms",
        description="Selects this variant: the `/stats` page's histograms.",
    )
    layers: list[str] | None = Field(default=None, description=_LAYERS_DOC)
    phase: str | None = Field(default=None, description=_PHASE_DOC)
    log_x: bool = Field(
        default=False, description="Spread the bins evenly by magnitude."
    )
    log_y: bool = Field(
        default=False, description="Log-scale the counts, revealing sparse tails."
    )


class PatchesViewSpec(BaseModel):
    """The `/stats` page's MIN/MAX grids: the inputs that most excite a channel."""

    view: Literal["patches"] = Field(
        default="patches",
        description="Selects this variant: the `/stats` page's MIN/MAX grids.",
    )
    layers: list[str] | None = Field(default=None, description=_LAYERS_DOC)
    phase: str | None = Field(default=None, description=_PHASE_DOC)
    heatmap: bool = Field(
        default=False,
        description="Blend the channel's activation map over each patch.",
    )


class ExperimentRecordingSpec(BaseModel):
    """An experiment re-run once per frame, as the page's auto experiment does."""

    view: Literal["experiment"] = Field(
        default="experiment",
        description="Selects this variant: an experiment re-run once per frame.",
    )
    layer: str = Field(description="Layer to run the experiment on.")
    kind: ExperimentKind = Field(
        description=(
            "Experiment to re-run for every frame; `list_experiments` describes "
            "each one."
        )
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Knobs for that kind, overriding its defaults; `list_experiments` "
            "has the keys. Omit for the defaults."
        ),
    )
    overlay: bool = Field(default=False, description=_OVERLAY_DOC)


class ExperimentSnapshotSpec(BaseModel):
    """One already-published experiment result, drawn as it stands."""

    view: Literal["experiment"] = Field(
        default="experiment",
        description=(
            "Selects this variant: an already-published experiment result."
        ),
    )
    seq: int | None = Field(
        default=None,
        description=(
            "Sequence number of the published experiment result to draw; omit "
            "for the newest."
        ),
    )
    overlay: bool = Field(default=False, description=_OVERLAY_DOC)


RecordingViewSpec = Annotated[
    Union[
        LayersViewSpec,
        WeightsViewSpec,
        HistogramsViewSpec,
        PatchesViewSpec,
        ExperimentRecordingSpec,
    ],
    Field(
        discriminator="view",
        description="The view to record, and the arguments that view takes.",
    ),
]

SnapshotViewSpec = Annotated[
    Union[
        LayersViewSpec,
        WeightsViewSpec,
        HistogramsViewSpec,
        PatchesViewSpec,
        ExperimentSnapshotSpec,
    ],
    Field(
        discriminator="view",
        description="The view to freeze, and the arguments that view takes.",
    ),
]
