from __future__ import annotations

import torch

from monai.metrics.meandice import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meaniou import MeanIoU

from src.surgical_copilot.bench.metrics.temporal_metrics.temporal_consistency import TemporalConsistencyMetric
from src.surgical_copilot.bench.metrics.temporal_metrics.temporal_iou import TemporalIoU


class MetricsManager:
    def __init__(self, device: torch.device):
        self.device = device

        self.dice_metric = DiceMetric(reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(percentile=95)
        self.iou_metric = MeanIoU(reduction="mean")

        self.temporal_iou_metric = TemporalIoU(
            threshold=0.5,
            from_logits=False,
            eps=1e-6
        )
        self.temporal_consistency_metric = None

    def _build_temporal_consistency(self):
        if self.temporal_consistency_metric is None:
            self.temporal_consistency_metric = TemporalConsistencyMetric(device=self.device)

    def reset(self):
        self.dice_metric.reset()
        self.hd95_metric.reset()
        self.iou_metric.reset()
        self.temporal_iou_metric.reset()

        if self.temporal_consistency_metric is not None:
            self.temporal_consistency_metric.reset()

    def update(self, preds: torch.Tensor, labels: torch.Tensor, images: torch.Tensor | None = None, is_first_frame: bool = False):
        if preds.ndim == 5:
            self._update_sequence(preds=preds, labels=labels, images=images, is_first_frame=is_first_frame)
            return

        self._update_spatial(preds=preds, labels=labels)
        self._update_temporal(preds=preds, labels=labels, images=images, is_first_frame=is_first_frame)

    def _update_sequence(self, preds: torch.Tensor, labels: torch.Tensor, images: torch.Tensor | None, is_first_frame: bool):
        b, t = preds.shape[:2]
        preds_flat = preds.reshape(b * t, *preds.shape[2:])
        labels_flat = labels.reshape(b * t, *labels.shape[2:])
        self._update_spatial(preds=preds_flat, labels=labels_flat)

        for time_idx in range(t):
            preds_t = preds[:, time_idx, ...]
            labels_t = labels[:, time_idx, ...] if labels is not None else None
            images_t = images[:, time_idx, ...] if images is not None else None

            first_frame = is_first_frame or (time_idx == 0)
            self._update_temporal(
                preds=preds_t,
                labels=labels_t,
                images=images_t,
                is_first_frame=first_frame
            )

    def _update_spatial(self, preds: torch.Tensor, labels: torch.Tensor):
        self.dice_metric(y_pred=preds, y=labels)
        self.hd95_metric(y_pred=preds, y=labels)
        self.iou_metric(y_pred=preds, y=labels)

    def _update_temporal(self, preds: torch.Tensor, labels: torch.Tensor | None, images: torch.Tensor | None, is_first_frame: bool):
        self.temporal_iou_metric(preds=preds, is_first_frame=is_first_frame)

        if images is None:
            return

        self._build_temporal_consistency()
        if is_first_frame:
            self.temporal_consistency_metric.reset_sequence()
        self.temporal_consistency_metric(preds=preds, labels=labels, images=images)

    def aggregate(self) -> dict:
        metrics = {
            "dice": self.dice_metric.aggregate().item(),
            "hd95": self.hd95_metric.aggregate().item(),
            "iou": self.iou_metric.aggregate().item(),
            **self.temporal_iou_metric.aggregate(),
        }

        if self.temporal_consistency_metric is not None:
            metrics.update(self.temporal_consistency_metric.aggregate())
        else:
            metrics["temporal_consistency"] = 0.0

        return metrics
