"""Physics-aware metrics for grid-based lensing super-resolution.

These metrics are meant to complement PSNR/SSIM. They evaluate whether a
super-resolved reconstruction is consistent with the low-resolution observation
and with the lensing/source-plane structure used by the training pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn.functional as F


EPS = 1e-8


def _as_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert common image tensor shapes to [B, C, H, W]."""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected 2D, 3D, or 4D tensor; got shape {tuple(x.shape)}")


def _broadcast_like(weight: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
    """Broadcast a mask/weight/variance tensor to a target BCHW shape."""
    if weight is None:
        return None
    weight = weight.to(device=target.device, dtype=target.dtype)
    if weight.ndim == 2:
        weight = weight.unsqueeze(0).unsqueeze(0)
    elif weight.ndim == 3:
        weight = weight.unsqueeze(1)
    elif weight.ndim != 4:
        raise ValueError(f"Cannot broadcast tensor with shape {tuple(weight.shape)}")
    return weight.expand_as(target)


def _safe_mean(x: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight is None:
        return x.mean()
    weight = _broadcast_like(weight, x)
    return (x * weight).sum() / (weight.sum() + EPS)


def normalized_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    variance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted MSE normalized by target power.

    If variance is supplied, the residual is divided by variance, giving a
    diagonal-noise chi-square style metric. If weight is also supplied, both are
    applied.
    """
    pred = _as_bchw(pred)
    target = _as_bchw(target).to(device=pred.device, dtype=pred.dtype)
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}")

    residual2 = (pred - target) ** 2
    target2 = target ** 2

    if variance is not None:
        variance = _broadcast_like(variance, residual2).clamp_min(EPS)
        residual2 = residual2 / variance
        target2 = target2 / variance

    numerator = _safe_mean(residual2, weight)
    denominator = _safe_mean(target2, weight).clamp_min(EPS)
    return numerator / denominator


def lr_redegradation_error(
    degraded_prediction: torch.Tensor,
    lr_observation: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    variance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Metric for A(x_SR) ~= y_LR.

    degraded_prediction should already be PSF-convolved and downsampled to the
    LR observation grid.
    """
    return normalized_weighted_mse(degraded_prediction, lr_observation, weight=weight, variance=variance)


def source_plane_reconstruction_consistency(
    predicted_source_lr: torch.Tensor,
    reference_source_lr: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Consistency between downsampled SR source and LR source reconstruction.

    In the current fixed-grid pipeline, the available source-plane consistency
    check is whether the super-resolved source, after downsampling, agrees with
    the source reconstructed directly from the LR image.
    """
    return normalized_weighted_mse(predicted_source_lr, reference_source_lr, weight=weight)


def total_flux(
    image: torch.Tensor,
    area: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-sample flux, optionally with pixel-area and mask weights."""
    image = _as_bchw(image)
    weight = torch.ones_like(image)
    if area is not None:
        weight = weight * _broadcast_like(area, image)
    if mask is not None:
        weight = weight * _broadcast_like(mask, image)
    return (image * weight).sum(dim=(1, 2, 3))


def flux_conservation_error(
    before: torch.Tensor,
    after: torch.Tensor,
    area_before: Optional[torch.Tensor] = None,
    area_after: Optional[torch.Tensor] = None,
    mask_before: Optional[torch.Tensor] = None,
    mask_after: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Relative absolute flux difference between two image tensors.

    Returns the batch mean of |F_after - F_before| / (|F_before| + eps).
    """
    f_before = total_flux(before, area=area_before, mask=mask_before)
    f_after = total_flux(after, area=area_after, mask=mask_after)
    return (torch.abs(f_after - f_before) / (torch.abs(f_before) + EPS)).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Batch-mean PSNR for tensors already on a common grid."""
    pred = _as_bchw(pred)
    target = _as_bchw(target).to(device=pred.device, dtype=pred.dtype)
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3)).clamp_min(EPS)
    return (20.0 * torch.log10(torch.tensor(float(data_range), device=pred.device, dtype=pred.dtype)) - 10.0 * torch.log10(mse)).mean()


def lens_parameter_bias(
    pred_params: Dict[str, torch.Tensor],
    true_params: Dict[str, torch.Tensor],
    relative_keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute absolute or relative lens-parameter bias.

    This is a utility for future simulated-data evaluations. It is not called
    by the current fixed-lens training script because the dataset returns only
    images, not simulation metadata.
    """
    relative_keys = set(relative_keys or [])
    out = {}
    for key, pred in pred_params.items():
        if key not in true_params:
            continue
        true = true_params[key].to(device=pred.device, dtype=pred.dtype)
        diff = pred - true
        if key in relative_keys:
            diff = diff / (torch.abs(true) + EPS)
        out[f"bias/{key}"] = diff.mean()
        out[f"mae/{key}"] = diff.abs().mean()
    return out


def classification_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Accuracy and per-class recall for downstream substructure classifiers.

    This can be used in a separate evaluation script once a classifier is
    available. It deliberately does not create a training dependency on a
    morphology classifier.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, num_classes]")
    targets = targets.to(device=logits.device).long().view(-1)
    preds = logits.argmax(dim=1)
    if preds.shape[0] != targets.shape[0]:
        raise ValueError("Number of predictions and targets differ")

    num_classes = int(num_classes or logits.shape[1])
    metrics = {"classification/accuracy": (preds == targets).float().mean()}
    for cls in range(num_classes):
        cls_mask = targets == cls
        if cls_mask.any():
            metrics[f"classification/recall_class_{cls}"] = (preds[cls_mask] == targets[cls_mask]).float().mean()
    return metrics


class MetricTracker:
    """Accumulate scalar tensor metrics over batches."""

    def __init__(self) -> None:
        self._values = defaultdict(float)
        self._counts = defaultdict(int)

    def update(self, **metrics: Union[torch.Tensor, float]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                value = float(value.detach().cpu().item())
            self._values[key] += float(value)
            self._counts[key] += 1

    def mean(self, key: str) -> float:
        return self._values[key] / max(self._counts[key], 1)

    def as_dict(self) -> Dict[str, float]:
        return {key: self.mean(key) for key in sorted(self._values)}

    def summary(self, prefix: str = "") -> str:
        values = self.as_dict()
        if prefix:
            values = {f"{prefix}{key}": value for key, value in values.items()}
        return " | ".join(f"{key}: {value:.6f}" for key, value in values.items())