"""Pairwise Spatial Transformer Network (STNReID, arXiv:1903.07072).

The STN module samples an *affined* view ``I_a`` from the holistic image so that its
ReID feature matches the one of the *partial* image ``I_p`` (eq. 1-2). The alignment is
supervised by ``L_STN = ||f_p - f_a||^2`` (eq. 5, see ``model/federated_losses.py``).

Integrated into ``model/tbps.py`` as ``self.stn_module`` (the ``stn*`` name is picked up by
``federated.strategy_subspace._is_heads`` so STN params are aggregated as a shared head).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import log as logger


class LocalizationNet(nn.Module):
    """Localization network of the STN (STNReID Table I).

    ``Conv(7x7,16,s2) -> BN -> ReLU -> MaxPool -> Conv(3x3,32,s2) -> BN -> ReLU -> MaxPool``
    then four FC layers regressing the 6 affine parameters of ``theta``. An
    ``AdaptiveAvgPool2d((16, 8))`` is inserted before the flatten so the FC input size is
    fixed at ``32*16*8 = 4096`` regardless of the configured image size (the paper assumes
    256x128; this project uses 256x256).
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((16, 8)),
        )
        self.regressor = nn.Sequential(
            nn.Linear(32 * 16 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),  # FC4: regression layer, no ReLU
        )
        # Initialise theta to the identity affine transform [1,0,0,0,1,0]
        # (weights zeroed) so the affined view starts equal to the holistic image.
        self.regressor[-1].weight.data.zero_()
        self.regressor[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x_concat: torch.Tensor) -> torch.Tensor:
        x = self.features(x_concat)
        x = torch.flatten(x, 1)
        theta = self.regressor(x)
        return theta.view(-1, 2, 3)


class SpatialTransformer(nn.Module):
    """Predicts ``theta`` from the (holistic, partial) pair and samples the affined image."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.loc_net = LocalizationNet(in_channels=2 * in_channels)

    def theta_from_pair(
        self, holistic: torch.Tensor, partial: torch.Tensor
    ) -> torch.Tensor:
        x_concat = torch.cat([holistic, partial], dim=1)  # [B, 2C, H, W]
        return self.loc_net(x_concat)

    def forward(self, holistic: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
        theta = self.theta_from_pair(holistic, partial)
        grid = F.affine_grid(theta, holistic.size(), align_corners=False)
        affined = F.grid_sample(holistic, grid, align_corners=False)  # I_a (eq. 1-2)
        return affined


class STNReIDModule(nn.Module):
    """STN module wired to the ReID encoder (``TBPS.encode_image``).

    Args:
        encode_image_fn: callable mapping an image tensor to its global feature
            (e.g. the bound ``TBPS.encode_image``). Stored as a plain attribute, so it is
            *not* registered as a submodule/parameter.
        config: full config (unused for now beyond input-channel inference, kept for API).
    """

    def __init__(self, encode_image_fn, config):
        super().__init__()
        self.spatial_transformer = SpatialTransformer(in_channels=3)
        self.encode_image = encode_image_fn
        logger.info("Initialised STNReIDModule (Spatial Transformer for partial ReID)")

    def transform(self, holistic: torch.Tensor, partial: torch.Tensor) -> torch.Tensor:
        """Return the affined image ``I_a`` sampled from the holistic image."""
        return self.spatial_transformer(holistic, partial)

    def forward(self, holistic: torch.Tensor, partial: torch.Tensor):
        """Return ``(f_h, f_p, f_a)`` global features of holistic/partial/affined views."""
        affined = self.transform(holistic, partial)
        f_h = self.encode_image(holistic)
        f_p = self.encode_image(partial)
        f_a = self.encode_image(affined)
        return f_h, f_p, f_a

    def compute_stn_loss(
        self, f_partial: torch.Tensor, f_affined: torch.Tensor
    ) -> torch.Tensor:
        from model.federated_losses import compute_stn_loss

        return compute_stn_loss(f_partial, f_affined)

    def freeze_stn(self) -> None:
        for param in self.spatial_transformer.parameters():
            param.requires_grad = False

    def unfreeze_stn(self) -> None:
        for param in self.spatial_transformer.parameters():
            param.requires_grad = True
