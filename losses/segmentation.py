import torch
import torch.nn as nn
from monai.losses import DiceCELoss, HausdorffDTLoss


class CombinedSegmentationLoss(nn.Module):
    """Combine region and boundary supervision for multiclass segmentation."""

    def __init__(
        self,
        region_weight: float = 0.7,
        boundary_weight: float = 0.3,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.region_weight = region_weight
        self.boundary_weight = boundary_weight
        self.region_loss = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            smooth_nr=smooth,
            smooth_dr=smooth,
            lambda_ce=1.0,
            lambda_dice=1.0,
        )
        self.boundary_loss = HausdorffDTLoss(
            alpha=1.0,
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target = target.long().unsqueeze(1)
        region = self.region_loss(logits, target)
        boundary = self.boundary_loss(logits, target)
        boundary = boundary / max(logits.shape[2:])
        return (
            self.region_weight * region
            + self.boundary_weight * boundary
        )
