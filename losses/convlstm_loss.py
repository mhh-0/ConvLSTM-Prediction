"""
ConvLSTM 流场预测专用损失函数

包含:
 - BoundaryMSELoss : 加重边界区域权重的 MSE
 - GradientLoss   : 空间梯度一致性损失
 - CombinedLoss   : 组合上述三项
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 边界加权 MSE
# ==========================================
class BoundaryMSELoss(nn.Module):

    def __init__(
            self,
            boundary_pixels=4,
            boundary_weight=5.0
    ):
        """
        Parameters
        ----------
        boundary_pixels:
            边界宽度（像素）

        boundary_weight:
            边界区域在 MSE 中的额外倍数
        """
        super().__init__()

        self.boundary_pixels = boundary_pixels
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        """
        pred / target: (B, C, H, W)
        """

        B, C, H, W = pred.shape

        # 构造空间权重 mask
        weight = torch.ones(
            H, W,
            device=pred.device,
            dtype=pred.dtype
        )

        bp = self.boundary_pixels

        # 四条边
        weight[:bp, :] = self.boundary_weight
        weight[-bp:, :] = self.boundary_weight
        weight[:, :bp] = self.boundary_weight
        weight[:, -bp:] = self.boundary_weight

        # 广播到 (1, 1, H, W)
        weight = weight.view(1, 1, H, W)

        squared_diff = (pred - target) ** 2

        return torch.mean(weight * squared_diff)


# ==========================================
# 梯度损失（保持空间结构）
# ==========================================
class GradientLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        pred / target: (B, C, H, W)
        """

        # X 方向梯度
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

        # Y 方向梯度
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = F.mse_loss(pred_dx, target_dx)
        loss_dy = F.mse_loss(pred_dy, target_dy)

        return loss_dx + loss_dy


# ==========================================
# 组合损失
# ==========================================
class CombinedLoss(nn.Module):

    def __init__(
            self,
            boundary_pixels=4,
            boundary_weight=5.0,
            lambda_boundary=0.2,
            lambda_gradient=0.05
    ):
        """
        loss = MSE + lambda_boundary * boundary_MSE
                    + lambda_gradient * gradient_loss
        """
        super().__init__()

        self.mse = nn.MSELoss()
        self.boundary_mse = BoundaryMSELoss(
            boundary_pixels=boundary_pixels,
            boundary_weight=boundary_weight
        )
        self.gradient = GradientLoss()

        self.lambda_boundary = lambda_boundary
        self.lambda_gradient = lambda_gradient

    def forward(self, pred, target):
        """
        pred / target: (B, C, H, W)

        Returns
        -------
        total_loss : scalar
        components : dict
            {"mse", "boundary", "gradient"}
        """

        loss_mse = self.mse(pred, target)

        loss_boundary = self.boundary_mse(pred, target)

        loss_grad = self.gradient(pred, target)

        total = (
            loss_mse
            + self.lambda_boundary * loss_boundary
            + self.lambda_gradient * loss_grad
        )

        components = {
            "mse": loss_mse.item(),
            "boundary": loss_boundary.item(),
            "gradient": loss_grad.item(),
            "total": total.item()
        }

        return total, components
