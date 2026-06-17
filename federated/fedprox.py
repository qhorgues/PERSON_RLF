"""FedProx proximal term (client-side).

Flower's native ``FedProx`` strategy aggregates exactly like ``FedAvg`` and simply forwards
``proximal_mu`` through the fit config; the regularisation must be added by the client. We
add ``(mu/2) * sum_k ||w_k - w_global_k||^2`` over the **shared subspace** parameters only
(FedSH aggregates a subspace, not the whole two-tower), so the proximal anchor is the global
subspace received at the start of the round.
"""

from typing import List

import numpy as np
import torch


def proximal_term(
    model: torch.nn.Module,
    global_subspace_tensors: List[torch.Tensor],
    selected_keys: List[str],
    mu: float,
) -> torch.Tensor:
    """``(mu/2) * sum_k ||w_k - w_global_k||^2`` over the subspace params of ``model``.

    ``model`` must be the module whose ``named_parameters()`` keys match ``selected_keys``
    (i.e. the ``LitTBPS`` used to build the ``SubspaceSelector``).
    """
    params = dict(model.named_parameters())
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    for key, global_w in zip(selected_keys, global_subspace_tensors):
        total = total + ((params[key] - global_w.to(device)) ** 2).sum()
    return 0.5 * mu * total


def attach_fedprox_hook(lit_model, global_subspace, mu: float, selected_keys: List[str]) -> None:
    """Store the global subspace anchor + mu + keys on ``lit_model`` for ``training_step``.

    ``global_subspace`` is the list of ``NDArrays`` received from the server (ordered like
    ``selector.selected_keys()``); they are converted to detached tensors on the model device.
    """
    device = next(lit_model.parameters()).device
    tensors = [
        torch.from_numpy(np.asarray(arr)).to(device=device, dtype=torch.float)
        for arr in global_subspace
    ]
    lit_model.global_subspace = tensors
    lit_model.proximal_mu = float(mu)
    lit_model.subspace_keys = list(selected_keys)
