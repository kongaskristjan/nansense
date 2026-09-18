"""Experiment result cards; consumes results without scheduling experiments."""

from __future__ import annotations

from collections.abc import Callable

from nicegui import ui
from torch import Tensor

from nansense.experiments import ExperimentResult
from nansense.ui.common import _b64_img_src, _label_bar_html, _strip_html
from nansense.ui.render import (
    INPUT_IMAGE_SIZE,
    StripRender,
    attribution_vmax,
    render_attribution_overlay,
    render_image,
    render_strip,
    tensor_hw,
)
from nansense.ui.theme import caption_color


def _experiment_img_html(image: bytes | None) -> str:
    """Input-space experiment image, CSS-upscaled like the input pane."""
    if image is None:
        return '<div class="text-xs text-slate-400 italic">not renderable</div>'
    return (
        f'<img src="{_b64_img_src(image)}" '
        f'style="width:{INPUT_IMAGE_SIZE}px; image-rendering:pixelated; '
        'display:block; max-width:none;" />'
    )


class ExperimentResults:
    def __init__(
        self, mean: tuple[float, ...] | None, std: tuple[float, ...] | None
    ) -> None:
        self.mean = mean
        self.std = std
        self.overlay = False
        self.container = ui.column().classes("gap-2 w-full")

    def _image_widget(self, tensor: Tensor | None, sample_idx: int) -> None:
        ui.html(
            _experiment_img_html(
                render_image(tensor, sample_idx, mean=self.mean, std=self.std)
            )
        )

    def _strip_widget(self, strip: StripRender | None) -> None:
        with ui.element("div").classes("max-w-full overflow-x-auto"):
            ui.html(_strip_html(strip, show_labels=True))

    def _captioned_cells(self, cells: list[tuple[str, Callable[[], None]]]) -> None:
        """A horizontal, scrollable row of captioned cells (caption over body),
        consistent with the watch / weights cards. Captions are filled color
        bars matching the main view's markers: input green, attribution/overlay
        purple, deep-dream channels slate (`theme.caption_color`)."""
        with ui.row().classes("items-start gap-4 no-wrap w-full overflow-x-auto"):
            for caption, build in cells:
                with ui.column().classes("items-center gap-1 shrink-0"):
                    ui.html(
                        _label_bar_html(caption.upper(), color=caption_color(caption))
                    ).classes("w-full")
                    build()

    def _sample_card(
        self, idx: int, cells: list[tuple[str, Callable[[], None]]]
    ) -> None:
        """One Captum result card per sample: a label over a row of captioned
        cells."""
        with ui.card().classes("w-full p-3 gap-2"):
            ui.label(f"Sample {idx}").classes(
                "font-mono text-sm font-bold text-slate-600"
            )
            self._captioned_cells(cells)

    def render(self, result: ExperimentResult, *, overlay: bool) -> None:
        """Render dream channels together, or one attribution card per sample."""
        self.overlay = overlay
        self.container.clear()
        with self.container:
            if result.image is not None:
                self._render_image_row(result.reference, result.image)
            elif result.attribution is not None:
                self._render_attribution_cards(result)

    def _render_image_row(self, reference: Tensor | None, image: Tensor) -> None:
        # One card, one horizontal row: the shared input (current-batch start
        # only) then one dreamed image per channel.
        cells: list[tuple[str, Callable[[], None]]] = []
        if reference is not None:
            cells.append(("input", lambda: self._image_widget(reference, 0)))
        for i in range(int(image.shape[0])):
            cells.append((f"channel {i}", lambda i=i: self._image_widget(image, i)))
        with ui.card().classes("w-full p-3 gap-2"):
            self._captioned_cells(cells)

    def _render_attribution_cards(self, result: ExperimentResult) -> None:
        attribution = result.attribution
        reference = result.reference
        assert attribution is not None
        attr = attribution  # narrowed; safe to index inside the cell closures
        n = int(attr.shape[0])
        if self.overlay and reference is not None:
            ref = reference  # narrowed for the closures below
            vmax = attribution_vmax(attr)
            for i in range(n):
                self._sample_card(
                    i,
                    [
                        (
                            "overlay",
                            lambda i=i: self._strip_widget(
                                render_attribution_overlay(
                                    ref[i],
                                    attr[i],
                                    mean=self.mean,
                                    std=self.std,
                                    vmax=vmax,
                                    tile_px=INPUT_IMAGE_SIZE,
                                )
                            ),
                        )
                    ],
                )
            return
        input_hw = tensor_hw(reference)
        for i in range(n):
            # Attribution map first, input second.
            cells: list[tuple[str, Callable[[], None]]] = [
                (
                    "attribution",
                    lambda i=i: self._strip_widget(
                        render_strip(
                            attr, i, input_hw=input_hw, tile_px=INPUT_IMAGE_SIZE
                        )
                    ),
                )
            ]
            if reference is not None:
                cells.append(("input", lambda i=i: self._image_widget(reference, i)))
            self._sample_card(i, cells)
