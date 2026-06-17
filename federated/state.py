"""Persistence of per-client local (non-shared) parameters across FL rounds.

Flower simulation clients are ephemeral: `client_fn` recreates the
TBPSFlowerClient on every round. Since FedSH does not share the full model,
the parameters outside the aggregated subspace (`SubspaceSelector.local_keys`)
must be persisted to disk between rounds, keyed by client id.
"""

import os
import shutil
from typing import Dict, Optional

import torch


class ClientStateStore:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def path(self, client_id) -> str:
        return os.path.join(self.root_dir, f"client_{client_id}.pt")

    def save(self, client_id, local_state_dict: Dict[str, torch.Tensor]) -> None:
        torch.save(local_state_dict, self.path(client_id))

    def load(self, client_id) -> Optional[Dict[str, torch.Tensor]]:
        if not self.exists(client_id):
            return None
        return torch.load(self.path(client_id), map_location="cpu")

    def exists(self, client_id) -> bool:
        return os.path.exists(self.path(client_id))

    def clear(self) -> None:
        """Purge all persisted client states (called at the start of a run)."""
        if os.path.isdir(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
