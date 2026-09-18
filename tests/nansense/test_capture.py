"""Capturing a batch must preserve the training computation."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import Tensor, nn

import nansense


class StatefulNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.bn(self.linear(x))).square()


class BranchingNet(StatefulNet):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * (2 if self.training else 3)


@pytest.mark.parametrize("model_type", [StatefulNet, BranchingNet])
def test_capture_preserves_outputs_gradients_buffers_and_rng(
    model_type: type[StatefulNet],
) -> None:
    torch.manual_seed(17)
    model = model_type()
    reference = deepcopy(model)
    session = nansense.start(model, epochs=3, phases={"phase": 1})
    session.detach()
    session.set_update_frequency(unit="batch", n=1)
    inputs = torch.randn(3, 4)
    try:
        for epoch, training in enumerate((True, False, True)):
            model.train(training)
            reference.train(training)
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            expected_input = inputs.clone().requires_grad_()
            actual_input = inputs.clone().requires_grad_()
            rng = torch.get_rng_state()
            expected = reference(expected_input)
            expected.sum().backward()
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            with session.batch(phase="phase", epoch=epoch):
                actual = model(actual_input)
                actual.sum().backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_input.grad, expected_input.grad)
            for actual_param, expected_param in zip(
                model.parameters(), reference.parameters()
            ):
                torch.testing.assert_close(actual_param.grad, expected_param.grad)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, reference.state_dict()[name])
            assert torch.equal(torch.get_rng_state(), expected_rng)
            snapshot = session.snapshot
            assert snapshot is not None
            assert "bn" in snapshot.activations
            assert "bn" in snapshot.activation_gradients
    finally:
        session.close()
