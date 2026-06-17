"""Federated / partial-ReID loss utilities.

- ``compute_stn_loss`` : STN alignment loss ``L_STN = ||f_p - f_a||^2`` (STNReID eq. 5).
- ``proximal_term`` : re-exported from ``federated.fedprox`` for API cohesion (FedProx).
"""

import torch
import torch.nn.functional as F


def compute_stn_loss(f_partial: torch.Tensor, f_affined: torch.Tensor) -> torch.Tensor:
    """L2 alignment between the partial and affined ReID features (STNReID eq. 5)."""
    return F.mse_loss(f_partial, f_affined)
