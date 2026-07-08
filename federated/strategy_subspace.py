"""Subspace selection: which parameters of LitTBPS are shared/aggregated by Flower
vs. kept purely local/personalized per client (FedSH "configurable common subspace").

Operates on `model.named_parameters()` (which de-duplicates aliased parameters, e.g.
`backbone.text_model.*` vs `model.text_model.*` which point to the same tensors) so
that every shared tensor is communicated exactly once, with a deterministic
(sorted) ordering matching Flower's `NDArrays`.
"""

import fnmatch
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from utils.logger import log as logger


def _match_any(name: str, patterns: List[str]) -> bool:
    """True if `name` (or any of its dot-separated components) matches any glob pattern."""
    parts = name.split(".")
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    return False


def _is_text_tower(name: str) -> bool:
    return _match_any(name, ["text_model*"])


def _is_vision_tower(name: str) -> bool:
    return _match_any(name, ["vision_model*"])


def _is_heads(name: str) -> bool:
    return _match_any(name, ["*projection*", "classifier*", "simclr_mlp*", "mlm_head*", "stn*"])


def _is_lora(name: str) -> bool:
    return _match_any(name, ["*lora_*"])


class SubspaceSelector:
    """Selects the shared subspace of `LitTBPS.named_parameters()` for a given preset."""

    PRESETS: Dict[str, Callable[[str], bool]] = {
        "full": lambda name: True,
        "text_tower": _is_text_tower,
        "vision_tower": _is_vision_tower,
        "heads_only": _is_heads,
        "text_tower+heads": lambda name: _is_text_tower(name) or _is_heads(name),
        "lora": lambda name: _is_lora(name) or _is_heads(name),
    }

    def __init__(self, reference_model, preset: str, custom_keys: Optional[List[str]] = None):
        if custom_keys is not None:
            self.preset = "custom"
            predicate = lambda name: _match_any(name, custom_keys)  # noqa: E731
        else:
            if preset not in self.PRESETS:
                raise ValueError(
                    f"Unknown subspace preset '{preset}'. Available: {list(self.PRESETS)}"
                )
            self.preset = preset
            predicate = self.PRESETS[preset]

        named_params = dict(reference_model.named_parameters())
        all_keys = sorted(named_params.keys())

        self._selected_keys = [k for k in all_keys if predicate(k)]
        self._local_keys = [k for k in all_keys if not predicate(k)]
        self._numel = {k: p.numel() for k, p in named_params.items()}

        logger.info(
            f"SubspaceSelector[{self.preset}]: {len(self._selected_keys)} shared / "
            f"{len(self._local_keys)} local parameter tensors "
            f"({self.num_shared_parameters():,} shared scalars)"
        )

    def selected_keys(self) -> List[str]:
        """Keys of the shared subspace, in deterministic (sorted) order."""
        return self._selected_keys

    def local_keys(self) -> List[str]:
        """Complement of `selected_keys`, persisted locally via ClientStateStore."""
        return self._local_keys

    def extract(self, model) -> List[np.ndarray]:
        """state_dict[selected_keys] -> list of ndarrays, ordered like `selected_keys`."""
        params = dict(model.named_parameters())
        return [params[key].detach().cpu().numpy() for key in self._selected_keys]

    def inject(self, model, ndarrays: List[np.ndarray]) -> None:
        """In-place copy of `ndarrays` into the subspace parameters of `model`."""
        params = dict(model.named_parameters())
        with torch.no_grad():
            for key, array in zip(self._selected_keys, ndarrays):
                param = params[key]
                param.copy_(torch.from_numpy(array).to(device=param.device, dtype=param.dtype))

    def extract_local(self, model) -> Dict[str, torch.Tensor]:
        """Snapshot of the non-shared (local/personalized) parameters, for ClientStateStore."""
        params = dict(model.named_parameters())
        return {key: params[key].detach().cpu().clone() for key in self._local_keys}

    def inject_local(self, model, local_state_dict: Dict[str, torch.Tensor]) -> None:
        """Restore the non-shared parameters previously saved via `extract_local`."""
        params = dict(model.named_parameters())
        with torch.no_grad():
            for key, tensor in local_state_dict.items():
                if key in params:
                    param = params[key]
                    param.copy_(tensor.to(device=param.device, dtype=param.dtype))

    def num_shared_parameters(self) -> int:
        """Number of scalars communicated per round (communication-cost metric)."""
        return sum(self._numel[k] for k in self._selected_keys)


def build_subspace_selector(config, reference_model) -> SubspaceSelector:
    """Factory: build the SubspaceSelector from `config.federated.subspace`."""
    federated_cfg = config.federated
    custom_keys = federated_cfg.get("custom_subspace_keys", None)
    return SubspaceSelector(
        reference_model, preset=federated_cfg.subspace, custom_keys=custom_keys
    )
