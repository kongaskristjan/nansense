"""Smoke tests for the small pre-activation ResNet."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import nansense
from examples.vision.resnet import PreActBlock, PreActResNet
from examples.common import evaluate, train_one_epoch
from tests.examples.helpers import assert_training_reduces_loss


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
    model = PreActResNet(num_classes=10, blocks_per_stage=blocks_per_stage)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (4, 10)


def test_resnet20_param_count() -> None:
    model = PreActResNet()  # the defaults are exactly ResNet-20
    n_params = sum(p.numel() for p in model.parameters())
    assert 250_000 < n_params < 300_000


@pytest.mark.parametrize(
    ("num_stages", "expected_stages", "expected_width"),
    [
        (3, ["stage1", "stage2", "stage3"], 64),
        (5, ["stage1", "stage2", "stage3", "stage4", "stage5"], 256),
    ],
)
def test_resnet_stage_layout(
    num_stages: int, expected_stages: list[str], expected_width: int
) -> None:
    """Stages keep their `stageN` names and double channels per downsample."""
    model = PreActResNet(blocks_per_stage=1, num_stages=num_stages)
    stage_names = [n for n, _ in model.named_children() if n.startswith("stage")]
    assert stage_names == expected_stages
    assert model.fc.in_features == expected_width
    assert model.head_bn.num_features == expected_width


@pytest.mark.parametrize("image_size", [32, 128])
def test_resnet_deep_forward_shape(image_size: int) -> None:
    model = PreActResNet(num_classes=10, blocks_per_stage=1, num_stages=5)
    x = torch.randn(2, 3, image_size, image_size)
    assert model(x).shape == (2, 10)


def test_resnet_deep_is_fx_traceable() -> None:
    """The stage loop in `forward` must unroll statically under fx tracing."""
    model = PreActResNet(blocks_per_stage=1, num_stages=5)
    traced = torch.fx.symbolic_trace(model)
    x = torch.randn(2, 3, 32, 32)
    assert torch.allclose(traced(x), model(x))


def test_resnet_rejects_zero_stages() -> None:
    with pytest.raises(ValueError):
        PreActResNet(num_stages=0)


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = PreActResNet(num_classes=10, blocks_per_stage=1)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert_training_reduces_loss(model, x, y, optimizer=optimizer)


@pytest.mark.parametrize("amp_dtype", [None, torch.bfloat16])
def test_train_and_eval_loops_run(amp_dtype: torch.dtype | None) -> None:
    torch.manual_seed(0)
    model = PreActResNet(num_classes=10, blocks_per_stage=1)
    device = torch.device("cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    inputs = torch.randn(16, 3, 32, 32)
    targets = torch.randint(0, 10, (16,))
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    # A disabled session is the loops' no-op off switch, exactly as in main().
    session = nansense.start(
        model, epochs=1, phases={"train": 4, "val": 4}, enabled=False
    )

    train_stats = train_one_epoch(
        model, loader, optimizer, criterion, device, amp_dtype=amp_dtype, session=session
    )
    eval_stats = evaluate(
        model, loader, criterion, device, amp_dtype=amp_dtype, session=session
    )
    session.close()

    assert 0.0 <= train_stats.accuracy <= 1.0
    assert 0.0 <= eval_stats.accuracy <= 1.0
    assert train_stats.loss > 0.0
    assert eval_stats.loss > 0.0
