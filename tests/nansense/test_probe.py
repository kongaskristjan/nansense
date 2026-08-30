"""Tests for probe runs: pinned-batch forwards between batches."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import nansense
from nansense.capture import _CaptureInterpreter
from nansense.probe import (
    _MAX_PROBE_CLIENTS,
    _shared_base_caps,
    apply_perturbations,
    gc_probe_clients,
    request_probe_locked,
    run_probe_guarded,
)
from nansense.session import Session
from nansense.ui.render import probe_act_tensor
from tests.nansense.helpers import DynamicNet, paused_session, train_step


class BnDropNet(nn.Module):
    """Conv + BatchNorm + Dropout: exercises mode isolation (stats, RNG)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.bn(self.conv(x)))
        h = self.drop(h)
        return self.fc(h.flatten(1))


def _bn_drop_step(model: BnDropNet) -> None:
    train_step(model, input_shape=(3, 4, 4), num_classes=2)


def _dynamic_step(model: DynamicNet) -> None:
    train_step(model, num_classes=2)


def test_pin_before_any_snapshot_returns_false() -> None:
    session = nansense.start(BnDropNet(), epochs=1, phases={"train": 1})
    assert session.pin_current_batch() is False
    assert session.is_pinned is False
    assert session.probe_result is None


def test_pin_on_disabled_session_returns_false() -> None:
    session = nansense.start(
        BnDropNet(), epochs=1, phases={"train": 1}, enabled=False
    )
    assert session.pin_current_batch() is False


def test_pin_while_paused_runs_probe_without_stepping() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        snap = session.snapshot
        assert snap is not None
        assert session.pause_count == 1

        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None
        assert session.probe_error is None
        # The probe ran on the paused training thread: no extra pause happened.
        assert session.pause_count == 1
        # The pinned input is the snapshot's input batch, position included.
        torch.testing.assert_close(probe.inputs["x"], snap.activations["x"])
        assert session.pinned_position == snap.position
        # Default mode is unchanged.
        assert probe.mode == "unchanged"
        # Every layer the snapshot knows shows up in the probe too, as
        # independent CPU clones.
        assert set(probe.activations) == set(session.layer_names)
        for name, tensor in probe.activations.items():
            assert tensor.device.type == "cpu", name
            assert not tensor.requires_grad, name


def test_probe_reruns_on_every_capture_while_pinned() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)
        first = session.probe_result
        assert first is not None
        first_count = session.probe_count

        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5)
        assert session.probe_count == first_count + 1
        second = session.probe_result
        assert second is not None and second is not first
        # Same pinned input, fresh activations against the stepped weights.
        torch.testing.assert_close(second.inputs["x"], first.inputs["x"])


@pytest.mark.parametrize(
    "mode, model_training, dropout_active",
    [
        ("eval", True, False),
        ("train", False, True),
        ("unchanged", True, True),
        ("unchanged", False, False),
    ],
)
def test_probe_mode_controls_dropout(
    mode: str, model_training: bool, dropout_active: bool
) -> None:
    """In eval, dropout is identity (drop output == relu output); in train it
    rescales and zeroes; "unchanged" follows the model's current flags."""
    model = BnDropNet()
    with paused_session(model, _bn_drop_step) as session:
        model.train(model_training)  # safe: the training thread is paused
        session.set_probe_mode(mode)
        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None and probe.mode == mode

        identical = torch.equal(probe.activations["drop"], probe.activations["relu"])
        assert identical != dropout_active
        # The probe restored the flags it flipped.
        assert all(m.training == model_training for m in model.modules())


@pytest.mark.parametrize("mode", ["unchanged", "eval", "train"])
def test_probe_leaves_no_trace_on_training_state(mode: str) -> None:
    """Probes restore BN buffers and the RNG stream in every mode — even
    "train", where the forward itself updates running stats in place."""
    model = BnDropNet()
    with paused_session(model, _bn_drop_step) as session:
        session.set_probe_mode(mode)
        # Selecting eval/train runs a probe right away (no pin needed), which
        # transiently flips the module flags. Drain it first so the baselines
        # below capture the worker's restored resting state, not a mid-probe one.
        if mode != "unchanged":
            assert session.wait_for_probe(timeout=5)
            assert session.probe_error is None

        running_mean = model.bn.running_mean
        running_var = model.bn.running_var
        batches_tracked = model.bn.num_batches_tracked
        assert running_mean is not None and running_var is not None
        assert batches_tracked is not None
        mean_before = running_mean.clone()
        var_before = running_var.clone()
        tracked_before = batches_tracked.clone()
        rng_before = torch.get_rng_state()
        flags_before = [m.training for m in model.modules()]

        count = session.probe_count
        assert session.pin_current_batch() is True
        assert session.wait_for_probe(after_count=count, timeout=5)
        assert session.probe_error is None

        torch.testing.assert_close(running_mean, mean_before)
        torch.testing.assert_close(running_var, var_before)
        assert torch.equal(batches_tracked, tracked_before)
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert [m.training for m in model.modules()] == flags_before


def test_probe_works_in_hook_fallback_mode() -> None:
    model = DynamicNet()
    with paused_session(model, _dynamic_step) as session:
        assert not session.fx_traced

        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None
        assert set(probe.activations) == {"x", "fc"}
        # The temporary probe hooks were removed again.
        assert session._hook_handles == []  # type: ignore[reportPrivateUsage]
        assert len(model._forward_hooks) == 0
        assert len(model._forward_pre_hooks) == 0


@pytest.mark.parametrize(
    "make_model, step",
    [(BnDropNet, _bn_drop_step), (DynamicNet, _dynamic_step)],
)
def test_probe_runs_after_batch_activations_are_freed(
    make_model: Callable[[], nn.Module],
    step: Callable[[Any], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The training batch's live activations are dropped before any probe
    forward — a pinned probe must not stack a second batch's worth of
    (GPU) memory on top of the captured step's own."""
    with paused_session(make_model(), step) as session:
        seen: list[int] = []
        original = session._capture_forward  # type: ignore[reportPrivateUsage]

        def spy(inputs: list[Tensor]) -> dict[str, Tensor]:
            seen.append(len(session._activations))  # type: ignore[reportPrivateUsage]
            return original(inputs)

        monkeypatch.setattr(session, "_capture_forward", spy)

        assert session.pin_current_batch() is True  # probe inside the pause loop
        assert session.wait_for_probe(timeout=5)
        session.step_batch()  # probe right after the next capture
        assert session.wait_until_paused(after_pauses=1, timeout=5)

        assert len(seen) == 2
        assert seen == [0, 0]


def test_capture_interpreter_to_cpu_stores_detached_clones() -> None:
    """`to_cpu=True` (the probe path) clones each output to CPU as it is
    produced — nothing captured stays grad-connected to the live forward."""
    gm = torch.fx.symbolic_trace(BnDropNet())
    x = torch.randn(2, 3, 4, 4, requires_grad=True)

    eager: dict[str, Tensor] = {}
    _CaptureInterpreter(gm, eager, to_cpu=True).run(x)
    assert eager
    for name, tensor in eager.items():
        assert tensor.device.type == "cpu", name
        assert not tensor.requires_grad, name

    # The default keeps live tensors so the batch path's backward can
    # populate `.grad` on them.
    live: dict[str, Tensor] = {}
    _CaptureInterpreter(gm, live).run(x)
    assert any(tensor.requires_grad for tensor in live.values())


def test_unpin_clears_probe_result() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)
        assert session.probe_result is not None

        session.unpin_batch()
        assert session.is_pinned is False
        assert session.probe_result is None
        assert session.pinned_position is None


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_set_mode_without_pin_probes_snapshot_input(mode: str) -> None:
    """Selecting eval/train re-runs the model on the current snapshot's batch
    without a pin — the documented "no pin required" behaviour."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        snap = session.snapshot
        assert snap is not None
        assert session.probe_result is None

        session.set_probe_mode(mode)
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None
        assert session.probe_error is None
        assert session.is_pinned is False
        assert probe.mode == mode
        # The base is the snapshot's (unpinned) input batch.
        torch.testing.assert_close(probe.inputs["x"], snap.activations["x"])


def test_mode_back_to_unchanged_clears_result_when_not_pinned() -> None:
    """eval/train -> unchanged with nothing pinned/perturbed drops the result
    so the page falls back to the live snapshot."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.set_probe_mode("eval")
        assert session.wait_for_probe(timeout=5)
        assert session.probe_result is not None

        session.set_probe_mode("unchanged")
        assert session.probe_result is None
        assert session.is_pinned is False


def test_mode_reruns_on_every_capture_without_pin() -> None:
    """An eval/train mode tracks the changing batch: each capture re-runs the
    probe on the just-stepped batch, like a pin does but on a live input."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.set_probe_mode("eval")
        assert session.wait_for_probe(timeout=5)
        first = session.probe_result
        assert first is not None
        count = session.probe_count

        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5)
        assert session.probe_count == count + 1
        second = session.probe_result
        assert second is not None and second is not first


def test_unpin_with_eval_mode_keeps_probing() -> None:
    """Unpinning while an eval/train mode is selected keeps probing — now
    against the snapshot input — instead of clearing the result."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.set_probe_mode("eval")
        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)
        count = session.probe_count

        session.unpin_batch()
        assert session.wait_for_probe(after_count=count, timeout=5)
        probe = session.probe_result
        assert probe is not None  # eval mode keeps the probe alive
        assert session.is_pinned is False
        assert probe.mode == "eval"


def test_set_probe_mode_rejects_unknown_mode() -> None:
    session = nansense.start(BnDropNet(), epochs=1, phases={"train": 1})
    with pytest.raises(ValueError, match="unknown probe mode"):
        session.set_probe_mode("bogus")


def test_apply_perturbations_writes_pixels_and_skips_bad_entries() -> None:
    base = torch.zeros(2, 3, 4, 4)
    perturbed = apply_perturbations(
        {"x": base},
        {
            ("x", 0, (1, 2)): (1.0, 2.0, 3.0),
            ("x", 1, (3, 0)): (4.0, 5.0, 6.0),
            ("x", 9, (0, 0)): (7.0, 7.0, 7.0),  # sample out of range — skipped
            ("x", 0, (0, 0)): (8.0, 8.0),  # wrong channel count — skipped
            ("missing", 0, (0, 0)): (1.0, 1.0, 1.0),  # absent input — skipped
        },
    )
    assert perturbed is not None
    out = perturbed["x"]
    torch.testing.assert_close(out[0, :, 1, 2], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(out[1, :, 3, 0], torch.tensor([4.0, 5.0, 6.0]))
    assert float(out.abs().sum()) == pytest.approx(21.0)  # nothing else
    assert float(base.abs().sum()) == 0.0  # the base is untouched


def test_apply_perturbations_writes_flat_channels_and_skips_bad_entries() -> None:
    base = torch.zeros(2, 4)
    perturbed = apply_perturbations(
        {"x": base},
        {
            ("x", 0, (2,)): (5.0,),
            ("x", 1, (9,)): (1.0,),  # channel out of range — skipped
            ("x", 0, (0, 0)): (1.0,),  # wrong index rank for a flat input — skipped
        },
    )
    assert perturbed is not None
    out = perturbed["x"]
    assert float(out[0, 2]) == 5.0
    assert float(out.abs().sum()) == pytest.approx(5.0)
    assert float(base.abs().sum()) == 0.0


def test_apply_perturbations_only_clones_perturbed_inputs() -> None:
    """Inputs without any in-range entry are shared by reference, not cloned."""
    img = torch.zeros(2, 3, 4, 4)
    other = torch.zeros(2, 5)
    perturbed = apply_perturbations(
        {"img": img, "other": other}, {("img", 0, (0, 0)): (1.0, 2.0, 3.0)}
    )
    assert perturbed is not None
    assert perturbed["img"] is not img  # cloned and edited
    assert perturbed["other"] is other  # untouched — same object


@pytest.mark.parametrize(
    "bases, perturbations",
    [
        ({"x": torch.zeros(2, 3, 4, 4)}, {}),  # nothing to apply
        ({"x": torch.zeros(2, 3, 4, 4)}, {("x", 9, (0, 0)): (1.0, 1.0, 1.0)}),  # all skipped
        ({"x": torch.zeros(2, 3, 4, 4)}, {("y", 0, (0, 0)): (1.0, 1.0, 1.0)}),  # absent
    ],
)
def test_apply_perturbations_returns_none_when_nothing_applies(
    bases: dict[str, Tensor],
    perturbations: dict[tuple[str, int, tuple[int, ...]], tuple[float, ...]],
) -> None:
    assert apply_perturbations(bases, perturbations) is None


def test_perturbation_without_pin_probes_snapshot_input() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        snap = session.snapshot
        assert snap is not None

        session.add_perturbation(
            input_name="x", sample=0, index=(1, 2), values=(5.0, 5.0, 5.0)
        )
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None
        assert session.is_pinned is False
        base = probe.inputs["x"]
        torch.testing.assert_close(base, snap.activations["x"])
        assert probe.perturbed_inputs is not None
        pert = probe.perturbed_inputs["x"]
        torch.testing.assert_close(pert[0, :, 1, 2], torch.tensor([5.0, 5.0, 5.0]))
        # Only the clicked pixel differs from the base input.
        mask = torch.ones_like(base, dtype=torch.bool)
        mask[0, :, 1, 2] = False
        torch.testing.assert_close(pert[mask], base[mask])
        # The second forward saw the edit: downstream activations differ.
        assert probe.perturbed_activations is not None
        assert not torch.equal(
            probe.perturbed_activations["conv"], probe.activations["conv"]
        )


def test_clear_perturbations_without_pin_clears_result() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.add_perturbation(
            input_name="x", sample=0, index=(0, 0), values=(1.0, 1.0, 1.0)
        )
        assert session.wait_for_probe(timeout=5)
        assert session.probe_result is not None

        session.clear_perturbations()
        assert session.probe_result is None
        assert session.perturbations == {}


def test_unpin_with_perturbations_keeps_probing() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        assert session.pin_current_batch() is True
        session.add_perturbation(
            input_name="x", sample=0, index=(0, 0), values=(1.0, 1.0, 1.0)
        )
        assert session.wait_for_probe(timeout=5)
        count = session.probe_count

        session.unpin_batch()
        assert session.wait_for_probe(after_count=count, timeout=5)
        probe = session.probe_result
        assert probe is not None  # perturbations keep the probe alive
        assert session.is_pinned is False
        assert probe.perturbed_inputs is not None


def test_out_of_range_perturbation_publishes_base_only() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.add_perturbation(
            input_name="x", sample=99, index=(0, 0), values=(1.0, 1.0, 1.0)
        )
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None
        assert probe.perturbed_inputs is None
        assert probe.perturbed_activations is None


def test_failing_probe_publishes_error_not_crash() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)

        # Sabotage the pinned input with an incompatible shape; the next probe
        # must fail gracefully into `probe_error` instead of killing the worker.
        session._pinned_inputs = {"x": torch.randn(2, 3, 9, 9)}  # type: ignore[reportPrivateUsage]
        count = session.probe_count
        with session._cv:  # type: ignore[reportPrivateUsage]
            request_probe_locked(session)
        assert session.wait_for_probe(after_count=count, timeout=5)
        assert session.probe_error is not None
        # The worker is still paused and responsive.
        assert session.pause_count == 1
        session.step_batch()
        assert session.wait_until_paused(after_pauses=1, timeout=5)


class TwoInputNet(nn.Module):
    """An image + a flat vector input: exercises multi-input probe forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4 + 2, 2)

    def forward(self, img: Tensor, vec: Tensor) -> Tensor:
        h = torch.relu(self.conv(img)).flatten(1)
        return self.fc(torch.cat([h, vec], dim=1))


def _two_input_step(model: TwoInputNet) -> None:
    img = torch.randn(2, 3, 4, 4)
    vec = torch.randn(2, 2)
    y = torch.randint(0, 2, (2,))
    model.zero_grad(set_to_none=True)
    nn.functional.cross_entropy(model(img, vec), y).backward()


def test_pin_captures_and_reforwards_every_input() -> None:
    with paused_session(TwoInputNet(), _two_input_step) as session:
        assert session.input_names == ["img", "vec"]
        snap = session.snapshot
        assert snap is not None
        assert session.pin_current_batch() is True
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None and session.probe_error is None
        # Both inputs are pinned and re-forwarded (the whole model runs).
        assert set(probe.inputs) == {"img", "vec"}
        torch.testing.assert_close(probe.inputs["img"], snap.activations["img"])
        torch.testing.assert_close(probe.inputs["vec"], snap.activations["vec"])


def test_perturb_one_input_reforwards_whole_multi_input_model() -> None:
    with paused_session(TwoInputNet(), _two_input_step) as session:
        session.add_perturbation(
            input_name="img", sample=0, index=(1, 2), values=(5.0, 5.0, 5.0)
        )
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None and session.probe_error is None
        assert probe.perturbed_inputs is not None
        # Only the targeted input is cloned/edited; the other is shared.
        assert probe.perturbed_inputs["vec"] is probe.inputs["vec"]
        torch.testing.assert_close(
            probe.perturbed_inputs["img"][0, :, 1, 2], torch.tensor([5.0, 5.0, 5.0])
        )
        # The full forward saw the edit: some layer's activation changed.
        assert probe.perturbed_activations is not None
        assert any(
            not torch.equal(probe.perturbed_activations[n], probe.activations[n])
            for n in probe.activations
            if n in probe.perturbed_activations
        )


def test_perturb_flat_input_writes_single_channel() -> None:
    with paused_session(TwoInputNet(), _two_input_step) as session:
        session.add_perturbation(
            input_name="vec", sample=1, index=(0,), values=(9.0,)
        )
        assert session.wait_for_probe(timeout=5)
        probe = session.probe_result
        assert probe is not None and session.probe_error is None
        assert probe.perturbed_inputs is not None
        assert float(probe.perturbed_inputs["vec"][1, 0]) == 9.0
        # The image input is untouched (shared by reference).
        assert probe.perturbed_inputs["img"] is probe.inputs["img"]


def test_failing_probe_publishes_error_when_still_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe that fails while its config is still current surfaces the error
    for the UI (the original behaviour)."""
    session = nansense.start(BnDropNet(), epochs=1, phases={"train": 1})

    def fail(_s: Session) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("nansense.probe._run_probe", fail)
    run_probe_guarded(session)
    assert session.probe_error is not None and "boom" in session.probe_error


def test_failing_probe_does_not_publish_a_superseded_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A probe whose config changed mid-run (version bumped) must not leave a
    stuck error behind the newer config — the newer request re-runs and owns
    the result. Without the staleness guard the failure path published
    unconditionally, stranding an error the new config never cleared."""
    session = nansense.start(BnDropNet(), epochs=1, phases={"train": 1})

    def bump_then_fail(s: Session) -> None:
        with s._cv:
            s._probe_version += 1  # a newer request supersedes this run
        raise RuntimeError("boom")

    monkeypatch.setattr("nansense.probe._run_probe", bump_then_fail)
    run_probe_guarded(session)
    assert session.probe_error is None


# --- Per-client perturbation (locked / shared demo sessions) ----------------
#
# The per-connection routing works regardless of lock (a `client` key always
# targets a private container), so these exercise the mechanics on an unlocked
# paused session; `test_lock.py` covers that it is *allowed while locked* when
# the shared path is refused.


def test_per_client_perturbation_runs() -> None:
    """A per-connection key perturbs a private copy and gets its own probe
    result, leaving the shared probe state untouched."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        snap = session.snapshot
        assert snap is not None

        session.add_perturbation(
            input_name="x",
            sample=0,
            index=(1, 2),
            values=(5.0, 5.0, 5.0),
            client="A",
        )
        assert session.wait_for_probe(client="A", timeout=5)
        probe = session.probe_result_for("A")
        assert probe is not None
        assert session.probe_result is None  # shared result untouched
        assert probe.perturbed_inputs is not None
        torch.testing.assert_close(
            probe.perturbed_inputs["x"][0, :, 1, 2],
            torch.tensor([5.0, 5.0, 5.0]),
        )
        assert probe.perturbed_activations is not None
        assert session.perturbations_for("A") != {}


def test_per_client_perturbations_are_isolated_and_share_the_base() -> None:
    """Two clients edit different pixels; each sees only its own edit, and the
    unperturbed base activations are computed once and shared."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.add_perturbation(
            input_name="x", sample=0, index=(0, 0), values=(9.0, 9.0, 9.0),
            client="A",
        )
        session.add_perturbation(
            input_name="x", sample=0, index=(3, 3), values=(1.0, 1.0, 1.0),
            client="B",
        )
        assert session.wait_for_probe(client="A", timeout=5)
        assert session.wait_for_probe(client="B", timeout=5)
        a = session.probe_result_for("A")
        b = session.probe_result_for("B")
        assert a is not None and b is not None and a is not b
        assert a.perturbed_inputs is not None and b.perturbed_inputs is not None
        torch.testing.assert_close(
            a.perturbed_inputs["x"][0, :, 0, 0], torch.tensor([9.0, 9.0, 9.0])
        )
        torch.testing.assert_close(
            b.perturbed_inputs["x"][0, :, 3, 3], torch.tensor([1.0, 1.0, 1.0])
        )
        assert len(session.perturbations_for("A")) == 1
        assert len(session.perturbations_for("B")) == 1
        # The shared base forward is computed once and reused (same object).
        assert a.activations is b.activations


def test_clear_perturbations_for_one_client_leaves_others() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        for key in ("A", "B"):
            session.add_perturbation(
                input_name="x", sample=0, index=(0, 0), values=(1.0, 1.0, 1.0),
                client=key,
            )
        assert session.wait_for_probe(client="A", timeout=5)
        assert session.wait_for_probe(client="B", timeout=5)

        session.clear_perturbations(client="A")
        assert session.probe_result_for("A") is None
        assert session.perturbations_for("A") == {}
        # B is untouched.
        assert session.probe_result_for("B") is not None
        assert session.perturbations_for("B") != {}


def test_probe_retains_only_the_perturbed_sample_rows() -> None:
    """The perturbed forward runs the whole batch but only the edited rows are
    kept — the difference between megabytes and gigabytes at the client cap."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.add_perturbation(
            input_name="x", sample=1, index=(0, 0), values=(4.0, 4.0, 4.0),
            client="A",
        )
        assert session.wait_for_probe(client="A", timeout=5)
        probe = session.probe_result_for("A")
        assert probe is not None and probe.perturbed_activations is not None
        assert probe.perturbed_samples == (1,)
        # The shared base keeps the full batch; the per-client forward keeps one row.
        assert all(t.shape[0] == 2 for t in probe.activations.values() if t.ndim)
        assert all(
            t.shape[0] == 1 for t in probe.perturbed_activations.values() if t.ndim
        )


def test_probe_act_tensor_resolves_perturbed_and_unperturbed_samples() -> None:
    """The strip shows the edited row's perturbed activations, and falls back to
    the base row for a sample nobody perturbed (its diff is zero)."""
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.add_perturbation(
            input_name="x", sample=1, index=(0, 0), values=(9.0, 9.0, 9.0),
            client="A",
        )
        assert session.wait_for_probe(client="A", timeout=5)
        probe = session.probe_result_for("A")
        assert probe is not None

        edited = probe_act_tensor(probe, "conv", compare=True, sample_idx=1)
        untouched = probe_act_tensor(probe, "conv", compare=True, sample_idx=0)
        assert edited is not None and untouched is not None
        assert edited.shape[0] == 1 and edited.abs().sum() > 0
        torch.testing.assert_close(untouched, torch.zeros_like(untouched))

        # Without `compare`, an unperturbed sample renders its base row.
        base_row = probe_act_tensor(probe, "conv", compare=False, sample_idx=0)
        assert base_row is not None
        torch.testing.assert_close(base_row, probe.activations["conv"][0:1])


def test_shared_base_cache_keys_on_base_identity() -> None:
    """A fresh base recomputes even when it holds equal values.

    Keying on `id()` would let a new snapshot reuse a freed one's address and
    serve every client a stale base to diff against.
    """
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        bases = session._snapshot_inputs()
        first = _shared_base_caps(session, bases, "eval")
        assert _shared_base_caps(session, bases, "eval") is first  # cache hit
        assert _shared_base_caps(session, bases, "train") is not first  # mode differs
        replacement = {name: t.clone() for name, t in bases.items()}
        assert _shared_base_caps(session, replacement, "eval") is not first


def test_probe_clients_capped_lru() -> None:
    """Only the most-recent `_MAX_PROBE_CLIENTS` containers are retained."""
    session = nansense.start(BnDropNet(), epochs=1, phases={"train": 1})
    for i in range(_MAX_PROBE_CLIENTS + 3):
        session.register_probe_client(f"c{i}")
    assert len(session._probe_clients) == _MAX_PROBE_CLIENTS
    assert "c0" not in session._probe_clients  # oldest evicted
    assert f"c{_MAX_PROBE_CLIENTS + 2}" in session._probe_clients  # newest kept


def test_probe_clients_reaped_after_ttl() -> None:
    """A container whose page heartbeat lapsed is dropped; a fresh one stays."""
    session = nansense.start(BnDropNet(), epochs=1, phases={"train": 1})
    session.register_probe_client("stale")
    session.register_probe_client("fresh")
    session._probe_clients["stale"].expires_at = time.monotonic() - 1.0
    gc_probe_clients(session)
    assert "stale" not in session._probe_clients
    assert "fresh" in session._probe_clients


def test_unregister_probe_client_drops_state() -> None:
    with paused_session(BnDropNet(), _bn_drop_step) as session:
        session.add_perturbation(
            input_name="x", sample=0, index=(0, 0), values=(1.0, 1.0, 1.0),
            client="A",
        )
        assert session.wait_for_probe(client="A", timeout=5)
        assert session.probe_result_for("A") is not None

        session.unregister_probe_client("A")
        assert session.probe_result_for("A") is None
        assert session.perturbations_for("A") == {}
