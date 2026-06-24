import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.utils.losses import DiceLoss

class Dice_BCE_Loss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(Dice_BCE_Loss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, input, target):
        return self.bce_weight * self.dice_loss(input, target) + self.dice_weight * self.bce_loss(input, target)


class BoundaryAwareLoss(nn.Module):
    """
    Combined loss: L_total = L_seg + lambda_bd * L_boundary

    L_seg: standard Dice + BCE on the main segmentation mask
    L_boundary: BCE between predicted boundary map and morphologically-derived
                ground truth boundary (dilation - erosion of target mask)

    Motivation: Hausdorff distance is a key evaluation metric in medical image
    segmentation. Standard Dice+BCE loss optimizes for region overlap but
    provides weak gradients at boundaries. Explicit boundary supervision
    directly improves boundary localization quality.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, boundary_weight=0.3,
                 boundary_kernel_size=3):
        super(BoundaryAwareLoss, self).__init__()
        self.seg_loss = Dice_BCE_Loss(bce_weight=bce_weight, dice_weight=dice_weight)
        self.boundary_weight = boundary_weight

        # Pre-define morphological kernel for boundary extraction
        ks = boundary_kernel_size
        self.register_buffer(
            'morph_kernel',
            torch.ones(1, 1, ks, ks)
        )
        self.pad = ks // 2

    def _extract_boundary(self, mask):
        """Extract boundary from binary mask via morphological dilation - erosion."""
        with torch.no_grad():
            kernel = self.morph_kernel.to(mask.device)
            # Dilation: max pooling equivalent
            dilated = F.conv2d(mask, kernel, padding=self.pad)
            dilated = (dilated > 0).float()
            # Erosion: check if all neighbors are 1
            eroded = F.conv2d(mask, kernel, padding=self.pad)
            eroded = (eroded >= kernel.sum()).float()
            # Boundary = dilation - erosion
            boundary = (dilated - eroded).clamp(0, 1)
        return boundary

    def forward(self, seg_pred, boundary_pred, target):
        """
        Args:
            seg_pred: (B, 1, H, W) — main segmentation prediction (with sigmoid)
            boundary_pred: (B, 1, H, W) — boundary prediction (with sigmoid)
            target: (B, 1, H, W) — ground truth mask
        Returns:
            total_loss: scalar
        """
        # Main segmentation loss
        l_seg = self.seg_loss(seg_pred, target)

        # Boundary loss
        gt_boundary = self._extract_boundary(target)
        # Resize boundary_pred to match target if needed
        if boundary_pred.shape[-2:] != target.shape[-2:]:
            boundary_pred = F.interpolate(boundary_pred, size=target.shape[-2:],
                                          mode='bilinear', align_corners=False)
        l_boundary = F.binary_cross_entropy(boundary_pred, gt_boundary)

        return l_seg + self.boundary_weight * l_boundary
