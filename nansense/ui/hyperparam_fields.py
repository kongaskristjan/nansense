"""Editable optimizer hyperparameters on the `/weights` page.

Each parameter panel shows its param group's numeric hyperparameters (`lr`,
`momentum`, ...) as small inputs. While training is paused, editing one and
pressing Enter (or leaving the field) writes it through
`Session.set_hyperparameters`. A value changed from NaNsense gets a tinted
background, and its tooltip names the value it replaced.

Next to `lr`, a "scale schedule" checkbox (one setting shared by every panel
on the page) makes an edit a multiplier on the scheduler's schedule instead,
and an active multiplier shows as a small `×factor` that ends it on click.
"""

from __future__ import annotations

from dataclasses import dataclass

from nicegui import ui

from nansense.hyperparams import HyperparameterEdit, source_key
from nansense.session import Session

# Applied to an input whose value was changed from NaNsense.
_EDITED_PROPS: str = "bg-color=amber-1"

# Quasar's dense inputs are 40px tall; these sit on a line of small text.
HYPERPARAM_CSS: str = """
<style>
  .nansense-hparam .q-field__control,
  .nansense-hparam .q-field__native { height: 26px; min-height: 26px; }
  .nansense-hparam .q-field__control { padding: 0 6px; }
</style>
"""


@dataclass
class RescaleChoice:
    """The page's "scale schedule" checkbox, shared by every panel's `lr`."""

    on: bool = False


def format_hyperparameter(value: float) -> str:
    """A value as the input shows it (the precision the rest of the page uses)."""
    return f"{value:.4g}"


class HyperparameterFields:
    """One input per numeric hyperparameter of a parameter's param group.

    Built once from the live optimizer (a group's keys don't change), then
    kept current by `sync`, which the page calls on every tick: values come
    from the shown snapshot, which `set_hyperparameters` updates in place.
    """

    def __init__(
        self, session: Session, param: str, rescale: RescaleChoice | None = None
    ) -> None:
        self._session = session
        self._param = param
        self._rescale = rescale if rescale is not None else RescaleChoice()
        self._multiplier: ui.label | None = None
        self._multiplier_tip: ui.tooltip | None = None
        self.group = session.optimizer_group_indices().get(param)
        self._group_size = sum(
            1 for g in session.optimizer_group_indices().values() if g == self.group
        )
        keys = sorted(session.current_optimizer_hyperparams().get(param, {}))
        self._inputs: dict[str, ui.input] = {}
        self._tooltips: dict[str, ui.tooltip] = {}
        # The text each input last showed, so leaving a field untouched never
        # writes a rounded value back.
        self._shown: dict[str, str] = {}
        self._last_state: object = None
        optimizer = session.optimizer
        if self.group is not None and optimizer is not None and len(
            optimizer.param_groups
        ) > 1:
            ui.label(f"param group {self.group}")
        for key in keys:
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.label(f"{key} =")
                field = (
                    ui.input()
                    .props("dense outlined input-class='font-mono text-xs'")
                    .classes("nansense-hparam w-24")
                )
                field.on("keydown.enter", lambda _, key=key: self._submit(key))
                field.on("blur", lambda _, key=key: self._submit(key))
                with field:
                    self._tooltips[key] = ui.tooltip("").classes("max-w-xs")
                self._inputs[key] = field
                if key == "lr":
                    self._add_rescale_controls()

    def _add_rescale_controls(self) -> None:
        """The active multiplier (hidden at 1) and the "scale schedule" toggle."""
        self._multiplier = (
            ui.label("")
            .classes("text-amber-9 cursor-pointer")
            .on("click", lambda: self._end_multiplier())
        )
        with self._multiplier:
            self._multiplier_tip = ui.tooltip("").classes("max-w-xs")
        self._multiplier.set_visibility(False)
        has_scheduler = self._session.scheduler is not None
        # The tooltip sits on a wrapper: a disabled checkbox gets no hover.
        with ui.element("span"):
            box = (
                ui.checkbox("scale schedule")
                .props("dense size=xs")
                .classes("text-xs")
                .bind_value(self._rescale, "on")
            )
            box.set_enabled(has_scheduler)
            ui.tooltip(
                "When checked, editing lr sets a multiplier (new lr ÷ the "
                "schedule's current, unscaled lr) applied after every later "
                "scheduler step, so the schedule keeps its shape. Only lr is "
                "rescaled; schedulers still control any momentum or betas they "
                "cycle."
                if has_scheduler
                else "Requires a scheduler: pass scheduler= to nansense.start()."
            ).classes("max-w-xs")

    @property
    def keys(self) -> list[str]:
        return list(self._inputs)

    def _editable(self) -> bool:
        session = self._session
        return not (
            session.locked
            or session.closed
            or session.training_lost
            or session.is_running
        )

    def sync(self) -> None:
        """Show the current values, edit marks and enabled state (cheap no-op
        when nothing changed)."""
        if not self._inputs:
            return
        snap = self._session.snapshot
        values = (
            snap.optimizer_hyperparams.get(self._param, {}) if snap is not None else {}
        )
        edits = self._session.hyperparameter_edits
        editable = self._editable()
        factor = self._factor()
        state = (tuple(sorted(values.items())), edits, editable, factor)
        if state == self._last_state:
            return
        self._last_state = state
        self._sync_multiplier(factor, values.get("lr"))
        originals = {e.key: e for e in edits if e.group == self.group}
        for key, field in self._inputs.items():
            if key in values:
                text = format_hyperparameter(values[key])
                self._shown[key] = text
                field.value = text
            field.set_enabled(editable)
            edit = originals.get(key)
            if edit is not None:
                field.props(_EDITED_PROPS)
            else:
                field.props(remove="bg-color")
            self._tooltips[key].text = self._tooltip(key, edit)

    def _factor(self) -> float | None:
        """This group's schedule multiplier, or `None` at 1."""
        if self.group is None:
            return None
        return self._session.lr_multipliers.get(self.group)

    def _sync_multiplier(self, factor: float | None, lr: float | None) -> None:
        if self._multiplier is None or self._multiplier_tip is None:
            return
        self._multiplier.set_visibility(factor is not None)
        if factor is None or lr is None:
            return
        self._multiplier.text = f"×{format_hyperparameter(factor)}"
        self._multiplier_tip.text = (
            f"lr = scheduled lr ({format_hyperparameter(lr / factor)}) "
            f"× {format_hyperparameter(factor)}. While paused, click to remove "
            "the multiplier and restore the schedule's current value."
        )

    def _end_multiplier(self) -> None:
        factor = self._factor()
        snap = self._session.snapshot
        lr = (
            snap.optimizer_hyperparams.get(self._param, {}).get("lr")
            if snap is not None
            else None
        )
        if factor is None or lr is None or not self._editable():
            return
        try:
            self._session.set_hyperparameters(
                {"lr": lr / factor}, group=self.group, rescale_schedule=True
            )
        except ValueError as error:
            ui.notify(str(error), type="warning")
            return
        self.sync()

    def _tooltip(self, key: str, edit: HyperparameterEdit | None) -> str:
        scope = (
            "Applies to this parameter only"
            if self._group_size == 1
            else f"Shared by {self._group_size} parameters in parameter group {self.group}"
        )
        optimizer = self._session.optimizer
        source = (
            source_key(optimizer.param_groups[self.group], key)
            if optimizer is not None and self.group is not None
            else key
        )
        text = (
            f"Optimizer setting: {source}. {scope}. While paused, enter a new value "
            "and press Enter or leave the field to apply it from the next "
            "optimizer step."
        )
        if edit is not None:
            text += (
                f" Before the first NaNsense edit: "
                f"{format_hyperparameter(edit.original)}."
            )
        return text

    def _submit(self, key: str) -> None:
        field = self._inputs[key]
        text = str(field.value or "").strip()
        if text == self._shown.get(key) or self.group is None:
            return
        try:
            value = float(text)
        except ValueError:
            ui.notify(f"{key}: {text!r} is not a number.", type="warning")
            field.value = self._shown.get(key, "")
            return
        try:
            self._session.set_hyperparameters(
                {key: value},
                group=self.group,
                rescale_schedule=key == "lr" and self._rescale.on,
            )
        except ValueError as error:
            ui.notify(str(error), type="warning")
            field.value = self._shown.get(key, "")
            return
        self.sync()
