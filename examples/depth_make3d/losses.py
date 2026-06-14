"""Scale-invariant log-depth loss and the delta<1.25 accuracy metric.

Both treat a target depth of `0` as the *invalid* sentinel: Make3D ground
truth has unobserved pixels, so the dataset writes `0` there and the loss /
metric recompute the validity mask as `target > 0`. The model predicts
log-depth; the loss compares predictions to the log of the (valid) target and
the metric exponentiates back to metres for the threshold ratio. Keeping the
mask implicit in the target (rather than a third tensor) lets these plug
straight into `examples.common.train_one_epoch` / `evaluate`, whose
`(output, target)` signature carries exactly two tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# Floor applied to depths before taking logs, in metres. Guards `log(0)` for
# the (masked-out) invalid pixels and clamps any tiny positive depths.
_EPS_DEPTH: float = 1e-3


class ScaleInvariantLogLoss(nn.Module):
    """Eigen et al. (2014) scale-invariant log-depth loss.

    For valid pixels with log-error `d = log(pred_depth) - log(gt_depth)`:

        L = mean(d^2) - lambda * mean(d)^2

    The second term discounts a constant global scale offset, so the network is
    rewarded for getting *relative* depth right even when its absolute scale is
    off. `pred` is the model's raw log-depth output; `target` is depth in metres
    with `0` marking invalid pixels.
    """

    def __init__(self, variance_weight: float = 0.85) -> None:
        super().__init__()
        self.variance_weight = variance_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        mask = target > 0
        log_pred = pred
        log_gt = torch.log(target.clamp_min(_EPS_DEPTH))
        diff = (log_pred - log_gt) * mask

        n = mask.sum().clamp_min(1.0)
        sum_sq = (diff**2).sum()
        sum_d = diff.sum()
        # mean(d^2) - lambda * mean(d)^2, computed from masked sums.
        return sum_sq / n - self.variance_weight * (sum_d / n) ** 2


@torch.no_grad()
def delta_accuracy(pred: Tensor, target: Tensor, threshold: float = 1.25) -> float:
    """Fraction of valid pixels with `max(pred/gt, gt/pred) < threshold`.

    `pred` is log-depth (exponentiated here); `target` is depth in metres with
    `0` for invalid pixels. Returns a float in `[0, 1]` (0.0 if no valid pixel).
    """
    mask = target > 0
    if not bool(mask.any()):
        return 0.0
    pred_depth = torch.exp(pred).clamp_min(_EPS_DEPTH)
    gt_depth = target.clamp_min(_EPS_DEPTH)
    ratio = torch.maximum(pred_depth / gt_depth, gt_depth / pred_depth)
    correct = ((ratio < threshold) & mask).sum().float()
    return float(correct / mask.sum().float())
