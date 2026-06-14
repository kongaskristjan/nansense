"""Automatic numerical-error detection: NaN/Inf and gradient under/overflow.

The neural-network debugger runs a small set of checks every *n*th batch
(configurable, default 10) and, when it finds trouble, records a
`DebugError`, stops training, and the UI raises a red banner. Two checks:

- **NaN / ±Inf** — trips if a single non-finite value appears anywhere in a
  checked tensor (forward activations, activation gradients, weight
  gradients). One bad value poisons the rest of the run, so there is no
  fraction threshold here.
- **Underflow / overflow** — trips when a layer's *gradient* magnitude
  collapses into a precision-losing band. The band is dtype-aware:
  underflow is the subnormal range (nonzero ``|x|`` below the dtype's
  smallest *normal* value — where the mantissa starts encoding scale rather
  than precision), overflow is ``|x|`` at or above the dtype's largest
  finite value. A layer trips when the summed ``|x|`` inside the band is at
  least `threshold` (default 0.1) of the layer's total summed ``|x|``;
  non-finite values are excluded from those sums (they are the NaN/Inf
  check's concern).

Everything runs *on the computing device*: per-layer reductions are stacked
into one tensor and pulled to the CPU in a single transfer, so the per-batch
cost is a handful of fused reductions plus one small sync — cheap enough to
run inline at every checked batch's ``__exit__``.

The pure functions here own all the math and the banner's display logic, so
they are unit-testable without a session or a UI; `nansense.session` holds
the per-batch state and calls `run_checks`, and `nansense.ui.top_bar`
renders the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from nansense.schedule import BatchPosition

# A check "category" groups the reasons that a single settings toggle (and a
# single banner DISABLE button) controls.
NAN_INF = "nan_inf"
UNDER_OVER = "under_over"

# Reasons, in display order. `nan`/`inf` belong to the NAN_INF category;
# `underflow`/`overflow` to UNDER_OVER.
REASONS: tuple[str, ...] = ("nan", "inf", "underflow", "overflow")
_CATEGORY_REASONS: dict[str, tuple[str, ...]] = {
    NAN_INF: ("nan", "inf"),
    UNDER_OVER: ("underflow", "overflow"),
}
REASON_OF_CATEGORY: dict[str, str] = {
    reason: category
    for category, reasons in _CATEGORY_REASONS.items()
    for reason in reasons
}

# Human-readable labels for reasons (banner text, table headers).
REASON_LABELS: dict[str, str] = {
    "nan": "NaN",
    "inf": "±Inf",
    "underflow": "underflow",
    "overflow": "overflow",
}
CATEGORY_LABELS: dict[str, str] = {
    NAN_INF: "NaN/Inf",
    UNDER_OVER: "underflow/overflow",
}


@dataclass(frozen=True)
class DebugSettings:
    """User-facing configuration of the debugger (mutated via the gear menu).

    `interval` is the batch cadence (1 = every batch). `threshold` is the
    underflow/overflow trip fraction (of summed ``|grad|``). Both checks can
    be toggled independently; `enabled` is the master switch.
    """

    enabled: bool = True
    interval: int = 10
    check_nan_inf: bool = True
    check_under_over: bool = True
    threshold: float = 0.1

    def any_check(self) -> bool:
        """Whether at least one check would run (master + one sub-toggle)."""
        return self.enabled and (self.check_nan_inf or self.check_under_over)


@dataclass(frozen=True)
class LayerReport:
    """Per-layer fractions for one detected error (a row in the dialog table).

    `nan` / `inf` are fractions of *element count*; `underflow` / `overflow`
    are fractions of the layer's summed ``|grad|``. All in ``[0, 1]``.
    """

    layer: str
    nan: float
    inf: float
    underflow: float
    overflow: float


@dataclass(frozen=True)
class DebugError:
    """An immutable record of one detected numerical error.

    `reasons` is the subset of `REASONS` that tripped; `checks_used` is the
    subset of categories that actually ran (so the UI shows only the relevant
    table columns even for an error that tripped just one of them);
    `layers` are the affected layers, in `layer_names` order.
    """

    position: BatchPosition
    reasons: tuple[str, ...]
    checks_used: tuple[str, ...]
    layers: tuple[LayerReport, ...]


def _pick_device(*tensor_maps: dict[str, Tensor]) -> torch.device | None:
    """The device to run reductions on — the first checked tensor's device."""
    for tensors in tensor_maps:
        for t in tensors.values():
            if isinstance(t, Tensor):
                return t.device
    return None


def _ordered_layers(
    layer_weights: dict[str, list[str]], activations: dict[str, Tensor]
) -> list[str]:
    """Layer names to scan, in a stable order keyed like `layer_names`."""
    if layer_weights:
        return list(layer_weights)
    return list(activations)


def _layer_metrics(
    ni_tensors: list[Tensor], grad_tensors: list[Tensor], device: torch.device
) -> Tensor:
    """One layer's raw counters as a 6-vector, computed on `device`.

    ``[nan_count, inf_count, total_count, underflow_abssum, overflow_abssum,
    finite_abssum]`` — counts over `ni_tensors` (NaN/Inf scan), abs-sums over
    `grad_tensors` (band metric). Returned as a device tensor so the caller
    can stack every layer and sync once.
    """
    nan_count = torch.zeros((), device=device, dtype=torch.float64)
    inf_count = torch.zeros((), device=device, dtype=torch.float64)
    total_count = 0
    for t in ni_tensors:
        nan_count = nan_count + torch.isnan(t).sum().to(torch.float64)
        inf_count = inf_count + torch.isinf(t).sum().to(torch.float64)
        total_count += t.numel()

    under_sum = torch.zeros((), device=device, dtype=torch.float64)
    over_sum = torch.zeros((), device=device, dtype=torch.float64)
    finite_sum = torch.zeros((), device=device, dtype=torch.float64)
    zero = torch.zeros((), device=device, dtype=torch.float64)
    for t in grad_tensors:
        finfo = torch.finfo(t.dtype)
        absx = t.abs()
        finite = torch.isfinite(t)
        # Non-finite slots contribute 0 to every sum — they belong to the
        # NaN/Inf check, and an inf would otherwise poison the totals.
        absf = torch.where(finite, absx.to(torch.float64), zero)
        finite_sum = finite_sum + absf.sum()
        underflow = finite & (absx > 0) & (absx < finfo.tiny)
        overflow = finite & (absx >= finfo.max)
        under_sum = under_sum + torch.where(underflow, absf, zero).sum()
        over_sum = over_sum + torch.where(overflow, absf, zero).sum()

    return torch.stack(
        [
            nan_count,
            inf_count,
            torch.full((), float(total_count), device=device, dtype=torch.float64),
            under_sum,
            over_sum,
            finite_sum,
        ]
    )


def run_checks(
    settings: DebugSettings,
    *,
    position: BatchPosition,
    activations: dict[str, Tensor],
    activation_grads: dict[str, Tensor],
    weight_grads: dict[str, Tensor],
    layer_weights: dict[str, list[str]],
) -> DebugError | None:
    """Run the enabled checks over one batch's tensors; `None` if all clean.

    `activations` / `activation_grads` are keyed by layer name; `weight_grads`
    by qualified parameter name, mapped to layers via `layer_weights`.
    """
    if not settings.any_check():
        return None
    check_ni = settings.check_nan_inf
    check_uo = settings.check_under_over
    threshold = settings.threshold

    device = _pick_device(activations, activation_grads, weight_grads)
    if device is None:
        return None

    names: list[str] = []
    vectors: list[Tensor] = []
    for name in _ordered_layers(layer_weights, activations):
        grad_tensors: list[Tensor] = []
        g = activation_grads.get(name)
        if isinstance(g, Tensor) and g.is_floating_point():
            grad_tensors.append(g)
        for pname in layer_weights.get(name, ()):
            wg = weight_grads.get(pname)
            if isinstance(wg, Tensor) and wg.is_floating_point():
                grad_tensors.append(wg)

        ni_tensors: list[Tensor] = []
        if check_ni:
            act = activations.get(name)
            if isinstance(act, Tensor) and act.is_floating_point():
                ni_tensors.append(act)
            ni_tensors.extend(grad_tensors)

        scan_grads = grad_tensors if check_uo else []
        if not ni_tensors and not scan_grads:
            continue
        names.append(name)
        vectors.append(_layer_metrics(ni_tensors, scan_grads, device))

    if not vectors:
        return None

    # One GPU->CPU sync for the whole batch.
    stacked = torch.stack(vectors).to("cpu").tolist()

    reports: list[LayerReport] = []
    nan_hit = inf_hit = under_hit = over_hit = False
    for name, (nan_c, inf_c, total_c, under_s, over_s, finite_s) in zip(
        names, stacked
    ):
        nan_f = nan_c / total_c if total_c > 0 else 0.0
        inf_f = inf_c / total_c if total_c > 0 else 0.0
        under_f = under_s / finite_s if finite_s > 0 else 0.0
        over_f = over_s / finite_s if finite_s > 0 else 0.0

        layer_nan = check_ni and nan_f > 0.0
        layer_inf = check_ni and inf_f > 0.0
        layer_under = check_uo and under_f >= threshold
        layer_over = check_uo and over_f >= threshold
        nan_hit = nan_hit or layer_nan
        inf_hit = inf_hit or layer_inf
        under_hit = under_hit or layer_under
        over_hit = over_hit or layer_over

        if layer_nan or layer_inf or layer_under or layer_over:
            reports.append(
                LayerReport(
                    layer=name,
                    nan=nan_f,
                    inf=inf_f,
                    underflow=under_f,
                    overflow=over_f,
                )
            )

    reasons = tuple(
        r
        for r, hit in (
            ("nan", nan_hit),
            ("inf", inf_hit),
            ("underflow", under_hit),
            ("overflow", over_hit),
        )
        if hit
    )
    if not reasons:
        return None
    checks_used = tuple(
        c
        for c, used in ((NAN_INF, check_ni), (UNDER_OVER, check_uo))
        if used
    )
    return DebugError(
        position=position,
        reasons=reasons,
        checks_used=checks_used,
        layers=tuple(reports),
    )


def without_category(error: DebugError, category: str) -> DebugError | None:
    """`error` with `category`'s reasons/columns dropped (DISABLE button).

    Returns `None` when nothing remains — the banner then disappears.
    """
    drop = set(_CATEGORY_REASONS.get(category, ()))
    reasons = tuple(r for r in error.reasons if r not in drop)
    if not reasons:
        return None
    checks_used = tuple(c for c in error.checks_used if c != category)
    layers = tuple(
        report for report in error.layers if _affected_by(report, reasons)
    )
    return replace(error, reasons=reasons, checks_used=checks_used, layers=layers)


def _affected_by(report: LayerReport, reasons: tuple[str, ...]) -> bool:
    """Whether `report` has a nonzero fraction for any of `reasons`."""
    return any(getattr(report, reason) > 0.0 for reason in reasons)


def categories_present(error: DebugError) -> list[str]:
    """The categories with at least one tripped reason, in fixed order."""
    present = {REASON_OF_CATEGORY[r] for r in error.reasons}
    return [c for c in (NAN_INF, UNDER_OVER) if c in present]


def columns(error: DebugError) -> list[str]:
    """The reason columns the dialog table shows (only for checks that ran)."""
    used = set(error.checks_used)
    return [r for r in REASONS if REASON_OF_CATEGORY[r] in used]


def reasons_text(error: DebugError) -> str:
    """A comma-joined, human-readable list of the tripped reasons."""
    return ", ".join(REASON_LABELS[r] for r in error.reasons)
