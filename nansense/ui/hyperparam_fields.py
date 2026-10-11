"""Editable optimizer hyperparameters on the `/weights` page.

Each parameter panel shows its param group's numeric hyperparameters (`lr`,
`momentum`, ...) as small inputs. While training is paused, editing one and
pressing Enter (or leaving the field) writes it through
`Session.set_hyperparameters`. A value changed from NaNsense gets a tinted
background, and its tooltip names the value it replaced.
"""

from __future__ import annotations

from nicegui import ui

from nansense.hyperparams import HyperparameterEdit
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


def format_hyperparameter(value: float) -> str:
    """A value as the input shows it (the precision the rest of the page uses)."""
    return f"{value:.4g}"


class HyperparameterFields:
    """One input per numeric hyperparameter of a parameter's param group.

    Built once from the live optimizer (a group's keys don't change), then
    kept current by `sync`, which the page calls on every tick: values come
    from the shown snapshot, which `set_hyperparameters` updates in place.
    """

    def __init__(self, session: Session, param: str) -> None:
        self._session = session
        self._param = param
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
        state = (tuple(sorted(values.items())), edits, editable)
        if state == self._last_state:
            return
        self._last_state = state
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

    def _tooltip(self, key: str, edit: HyperparameterEdit | None) -> str:
        scope = (
            "Applies to this parameter only"
            if self._group_size == 1
            else f"Shared by all {self._group_size} parameters in param group {self.group}"
        )
        text = (
            f"The optimizer's {key}. {scope}. While paused, enter a new value "
            "and press Enter or leave the field; it applies from the next "
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
            self._session.set_hyperparameters({key: value}, group=self.group)
        except ValueError as error:
            ui.notify(str(error), type="warning")
            field.value = self._shown.get(key, "")
            return
        self.sync()
