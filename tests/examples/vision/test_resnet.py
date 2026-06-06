"""Smoke tests for the small CIFAR ResNet."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from examples.vision.resnet import PreActBlock, ResNetCIFAR, resnet20
from examples.vision.train import evaluate, train_one_epoch


@pytest.mark.parametrize(
    ("in_channels", "out_channels", "stride", "expected_hw"),
    [
        (16, 16, 1, 8),
        (16, 32, 2, 4),
        (32, 64, 2, 4),
    ],
)
def test_preact_block_shapes(in_channels: int, out_channels: int, stride: int, expected_hw: int) -> None:
    block = PreActBlock(in_channels, out_channels, stride=stride)
    x = torch.randn(2, in_channels, 8, 8)
    y = block(x)
    assert y.shape == (2, out_channels, expected_hw, expected_hw)


def test_preact_block_same_shape_has_no_shortcut_submodule() -> None:
    """A same-shape block must add the input directly with no shortcut submodule."""
    block = PreActBlock(16, 16, stride=1)
    assert block.shortcut is None
    assert "shortcut" not in dict(block.named_children())


def test_preact_block_downsample_uses_avgpool_shortcut() -> None:
    """ResNet-D: downsampling shortcuts avg-pool first, then 1x1 conv (no BN)."""
    block = PreActBlock(16, 32, stride=2)
    assert block.shortcut is not None
    children = list(block.shortcut.children())
    assert isinstance(children[0], nn.AvgPool2d)
    assert isinstance(children[1], nn.Conv2d)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in block.shortcut.modules())


@pytest.mark.parametrize("blocks_per_stage", [1, 2, 3])
def test_resnet_forward_shape(blocks_per_stage: int) -> None:
    model = ResNetCIFAR(num_classes=10, blocks_per_stage=blocks_per_stage)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_resnet20_param_count() -> None:
    model = resnet20()
    n_params = sum(p.numel() for p in model.parameters())
    assert 250_000 < n_params < 300_000


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = ResNetCIFAR(num_classes=10, blocks_per_stage=1)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.train()
    initial = criterion(model(x), y).item()
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final = criterion(model(x), y).item()

    assert final < initial


@pytest.mark.parametrize("amp_dtype", [None, torch.bfloat16])
def test_train_and_eval_loops_run(amp_dtype: torch.dtype | None) -> None:
    torch.manual_seed(0)
    model = ResNetCIFAR(num_classes=10, blocks_per_stage=1)
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    train_stats = train_one_epoch(model, loader, optimizer, criterion, device, amp_dtype=amp_dtype)
    eval_stats = evaluate(model, loader, criterion, device, amp_dtype=amp_dtype)

    assert 0.0 <= train_stats.accuracy <= 1.0
    assert 0.0 <= eval_stats.accuracy <= 1.0
    assert train_stats.loss > 0.0
    assert eval_stats.loss > 0.0
