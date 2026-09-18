"""Tests for fx tracing, scoped intermediates, and the layer_weights and
layer_info mappings."""

from __future__ import annotations

import torch
import pytest
from torch import Tensor, nn

import nansense
from tests.nansense.helpers import (
    DynamicNet,
    TinyNet,
    make_session,
    paused_session,
    paused_worker,
    train_step,
)


@pytest.mark.parametrize("style", ["branch", "identity", "getattr", "dropout"])
@pytest.mark.parametrize("nested", [False, True])
def test_mode_dependent_forward_uses_hooks(style: str, nested: bool) -> None:
    class ModeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x: Tensor) -> Tensor:
            x = self.fc(x)
            if style == "dropout":
                return nn.functional.dropout(x, p=1.0, training=self.training)
            if style == "identity":
                return x * (2 if self.training is True else 3)
            if style == "getattr":
                return x * (2 if getattr(self, "training", False) else 3)
            return x * (2 if self.training else 3)

    from nansense.ui.graph import ROOT_ID, build_mermaid

    model = nn.Sequential(ModeNet()) if nested else ModeNet()
    getattribute = nn.Module.__getattribute__
    session = nansense.start(model, epochs=3, phases={"phase": 1})
    assert not session.fx_traced
    assert nn.Module.__getattribute__ is getattribute
    assert ROOT_ID in build_mermaid(model)
    session.detach()
    session.set_update_frequency(unit="batch", n=1)
    x = torch.ones(2, 2, requires_grad=True)
    for epoch, training in enumerate((True, False, True)):
        model.train(training)
        expected = model(x)
        expected_grad = torch.autograd.grad(expected.sum(), x)[0]
        with session.batch(phase="phase", epoch=epoch):
            actual = model(x)
            actual_grad = torch.autograd.grad(actual.sum(), x)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)
        assert session.snapshot is not None
        assert ("0.fc" if nested else "fc") in session.snapshot.activations
    session.close()


def test_fx_mode_captures_function_call_outputs() -> None:
    """When fx.symbolic_trace succeeds, call_function results show up too."""

    class BasicBlockLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(4)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.bn1(self.conv1(x)))

    model = BasicBlockLike()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.fx_traced
    assert "relu" in session.layer_names
    assert "conv1" in session.layer_names
    assert "bn1" in session.layer_names

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 3, 4, 4)
            y = torch.randint(0, 2, (2, 4, 4))
            model.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()

    with paused_worker(session, loop):
        snap = session.snapshot
        assert snap is not None
        assert "relu" in snap.activations
        # relu was applied to a tensor that requires grad, so we should also
        # have captured the gradient of its output.
        assert "relu" in snap.activation_gradients


def test_fx_mode_scopes_repeated_function_ops_by_submodule() -> None:
    """Two relus in a submodule capture under distinct scope-qualified keys."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn1 = nn.BatchNorm2d(4)
            self.bn2 = nn.BatchNorm2d(4)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.bn2(torch.relu(self.bn1(x))))

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.block = Block()

        def forward(self, x: Tensor) -> Tensor:
            return self.block(self.conv(x))

    model = Wrapper()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.fx_traced
    # The two functional relus are disambiguated by their submodule scope.
    assert "block.relu1" in session.layer_names
    assert "block.relu2" in session.layer_names
    assert "relu" not in session.layer_names  # the bare fx name is gone

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            x = torch.randn(2, 3, 4, 4)
            y = torch.randint(0, 2, (2, 4, 4))
            model.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()

    with paused_worker(session, loop):
        snap = session.snapshot
        assert snap is not None
        assert "block.relu1" in snap.activations
        assert "block.relu2" in snap.activations
        assert session.watch("block.relu2") is True


def test_fx_mode_restores_original_forward_after_batch() -> None:
    """The interpreter patch is reverted before the worker pauses, so the
    user's original forward is the live one whenever the batch isn't actively
    running. (The patch is only in place between __enter__ and __exit__.)"""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.fc(x))

    model = Tiny()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    original_forward = model.forward
    assert "forward" not in model.__dict__

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            train_step(model, num_classes=2)

    with paused_worker(session, loop):
        assert "forward" not in model.__dict__
        assert model.forward == original_forward
    assert "forward" not in model.__dict__


def test_fx_mode_captures_keyword_argument_call() -> None:
    """A model called with a keyword arg (`model(x=batch)`) still captures.

    The fx interpreter takes positional args matched to placeholder order, so
    the patched forward must normalize kwargs back to that order rather than
    raising `TypeError: fx_forward() got an unexpected keyword argument`.
    """

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.fc(x))

    model = Tiny()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.fx_traced

    x = torch.randn(2, 4)
    captured: dict[str, Tensor] = {}

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            model.zero_grad(set_to_none=True)
            out = model(x=x)  # keyword call: the bug crashed here on batch 0
            captured["out"] = out
            out.sum().backward()

    with paused_worker(session, loop):
        snap = session.snapshot
        assert snap is not None
        assert "relu" in snap.activations
        assert "fc" in snap.activations
        # The captured output matches the model's true (unpatched) forward.
        with torch.no_grad():
            expected = torch.relu(model.fc(x))
        assert torch.allclose(captured["out"], expected)


def test_fx_mode_positional_call_still_captures() -> None:
    """Regression: the common positional call path is unchanged by the
    kwargs-normalization fix (output and activations both populate)."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.fc(x))

    model = Tiny()
    session = nansense.start(model, epochs=1, phases={"train": 1})
    assert session.fx_traced

    x = torch.randn(2, 4)
    captured: dict[str, Tensor] = {}

    def loop() -> None:
        with session.batch(phase="train", epoch=0):
            model.zero_grad(set_to_none=True)
            out = model(x)  # plain positional call
            captured["out"] = out
            out.sum().backward()

    with paused_worker(session, loop):
        snap = session.snapshot
        assert snap is not None
        assert "relu" in snap.activations
        with torch.no_grad():
            expected = torch.relu(model.fc(x))
        assert torch.allclose(captured["out"], expected)


def test_fx_failure_falls_back_to_hooks() -> None:
    session = nansense.start(DynamicNet(), epochs=1, phases={"train": 1})
    assert not session.fx_traced
    # Hook-mode layer_names: inputs + module names.
    assert session.layer_names == ["x", "fc"]


def test_layer_weights_maps_modules_to_their_parameters_fx() -> None:
    """fx mode maps each call_module node to the params under its target and
    leaves weightless nodes (relu, input) with an empty list."""

    class ConvBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(4)

        def forward(self, x: Tensor) -> Tensor:
            return torch.relu(self.bn1(self.conv1(x)))

    session = nansense.start(ConvBlock(), epochs=1, phases={"train": 1})
    assert session.fx_traced
    lw = session.layer_weights
    # Every layer name has an entry; weightless ones map to [].
    assert set(lw) == set(session.layer_names)
    assert lw["conv1"] == ["conv1.bias", "conv1.weight"]
    assert lw["bn1"] == ["bn1.bias", "bn1.weight"]
    assert lw["relu"] == []
    assert lw["x"] == []


def test_layer_weights_covers_every_parameter_exactly() -> None:
    """In fx mode the per-layer mapping accounts for all of the model's
    parameters (TinyNet's two Linear layers, weight + bias each)."""
    session, model = make_session()
    mapped = {p for params in session.layer_weights.values() for p in params}
    assert mapped == {n for n, _ in model.named_parameters()}
    assert session.layer_weights["fc1"] == ["fc1.bias", "fc1.weight"]
    assert session.layer_weights["fc2"] == ["fc2.bias", "fc2.weight"]


def test_layer_weights_detects_functional_parameter_use() -> None:
    """A parameter used through F.conv2d (not a submodule call) is picked up
    via the get_attr node feeding the call_function node."""

    class Functional(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))

        def forward(self, x: Tensor) -> Tensor:
            return nn.functional.conv2d(x, self.weight, padding=1)

    session = nansense.start(Functional(), epochs=1, phases={"train": 1})
    assert session.fx_traced
    assert session.layer_weights["conv2d"] == ["weight"]


def test_layer_weights_uses_module_subtree_in_hook_fallback() -> None:
    """When fx tracing fails, a module maps to every parameter in its subtree."""
    session = nansense.start(DynamicNet(), epochs=1, phases={"train": 1})
    assert not session.fx_traced
    assert session.layer_weights == {"x": [], "fc": ["fc.bias", "fc.weight"]}


def test_layer_info_reports_module_and_functional_hyperparameters() -> None:
    """Module layers carry their extra_repr signature, fx ops their literal
    call arguments, and tensor-only ops / graph inputs carry nothing."""

    class PoolNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            y = torch.relu(self.conv(x))
            y = nn.functional.max_pool2d(y, 2)
            return torch.flatten(y, 1)

    session = nansense.start(PoolNet(), epochs=1, phases={"train": 1})
    assert session.fx_traced
    info = session.layer_info
    assert set(info) == set(session.layer_names)
    assert info["conv"] == (
        "Conv2d(3, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)"
    )
    assert info["x"] == ""
    assert info["relu"] == ""  # all-tensor args: nothing to report
    assert info["max_pool2d"].startswith("max_pool2d(2")
    assert info["flatten"] == "flatten(1)"


def test_layer_info_uses_module_repr_in_hook_fallback() -> None:
    session = nansense.start(DynamicNet(), epochs=1, phases={"train": 1})
    assert not session.fx_traced
    assert session.layer_info == {
        "x": "",
        "fc": "Linear(in_features=4, out_features=2, bias=True)",
    }


def test_layer_weights_keys_index_into_snapshot_weights() -> None:
    """Every parameter named by layer_weights is present in a snapshot."""
    with paused_session(TinyNet(), phases={"train": 1}) as session:
        snap = session.snapshot
        assert snap is not None
        for params in session.layer_weights.values():
            for name in params:
                assert name in snap.weights
