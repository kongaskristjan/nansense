"""Per-tab layer selection and shared collection-scope transitions."""

from __future__ import annotations

from dataclasses import dataclass

from nansense.probe import ProbeResult
from nansense.session import BatchSnapshot, Session, StatsScope


@dataclass
class MainState:
    last_snapshot: BatchSnapshot | None = None
    last_probe: ProbeResult | None = None
    dirty: bool = False
    rendering: bool = False
    # The shown set this connection last reflected in its DOM (card
    # visibility, amber classes, chip). The tick compares it against the
    # current shown set so changes made elsewhere (another tab, in the
    # coupled scope) propagate here too.
    last_watched: frozenset[str] = frozenset()


def _seed_shown(
    watched: frozenset[str], focus_layer: str, layer_names: list[str]
) -> set[str]:
    """A new tab's decoupled shown set: the deep link, or the watched seed."""
    if focus_layer in layer_names:
        return {focus_layer}
    return set(watched)


class MainController:
    def __init__(
        self, session: Session, layers: list[str], focus_layer: str = ""
    ) -> None:
        self.state = MainState()
        self.session = session
        self.layers = tuple(layers)
        self.scope = session.collecting_scope
        self.shown = _seed_shown(
            session.watched_layers, focus_layer if session.locked else "", layers
        )

    @property
    def decoupled(self) -> bool:
        """Whether this tab's shown set is its own rather than the watched set.

        Only a collecting scope of `ALL` decouples: pausing (`NONE`) keeps
        whichever coupling the resume scope has, so the toggle never moves
        cards around.
        """
        return self.session.collecting_scope is StatsScope.ALL

    @property
    def shown_layers(self) -> frozenset[str]:
        return frozenset(self.shown) if self.decoupled else self.session.watched_layers

    def reconcile_scope(self) -> bool:
        scope = self.session.collecting_scope
        if scope is self.scope:
            return False
        if scope is StatsScope.ALL:
            self.shown = set(self.session.watched_layers)
        self.scope = scope
        return True

    def toggle(self, name: str) -> bool:
        if name not in self.layers:
            return False
        if self.decoupled:
            if name in self.shown:
                self.shown.remove(name)
            else:
                self.shown.add(name)
        elif name in self.session.watched_layers:
            self.session.unwatch(name)
        else:
            return self.session.watch(name)
        return True
