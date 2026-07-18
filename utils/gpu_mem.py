"""GPU memory instrumentation + allocator configuration helpers.

Supports the "allocate once at the start of the run, no cudaMalloc mid-run" goal of
the federated simulation:

- `PYTORCH_CUDA_ALLOC_CONF` is set at process start (see `trainer_federated.py`),
  `torch.cuda.empty_cache()` is gated off during rounds, and the model is built once
  per Ray worker (see `federated/client.py`).
- These helpers let us *verify* the effect: if `torch.cuda.max_memory_reserved`
  stops growing across rounds, the allocator no longer requests memory from the
  driver mid-run.

All functions are no-ops when CUDA is unavailable, so they are safe to call from the
CPU fallback path.
"""

import gc
import time
from typing import Callable, Dict, Optional, TypeVar

import torch
from utils.logger import log as logger

_BYTES_PER_GIB = 1024**3

_T = TypeVar("_T")


def _device_index(device: Optional[torch.device] = None) -> int:
    """Resolve a bare device / None to a concrete CUDA device index."""
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        return device.index if device.index is not None else torch.cuda.current_device()
    return device


def reset_peak_stats(device: Optional[torch.device] = None) -> None:
    """Reset the CUDA peak-memory counters (call once after warm-up / at run start)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats_gib(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Current + peak allocated/reserved memory, in GiB. Empty dict if no CUDA."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": torch.cuda.memory_allocated(device) / _BYTES_PER_GIB,
        "reserved": torch.cuda.memory_reserved(device) / _BYTES_PER_GIB,
        "max_allocated": torch.cuda.max_memory_allocated(device) / _BYTES_PER_GIB,
        "max_reserved": torch.cuda.max_memory_reserved(device) / _BYTES_PER_GIB,
    }


def format_memory(device: Optional[torch.device] = None, prefix: str = "") -> str:
    """One-line human-readable summary of GPU memory usage."""
    stats = memory_stats_gib(device)
    if not stats:
        return f"{prefix}GPU memory: (CUDA unavailable)"
    return (
        f"{prefix}GPU mem GiB — reserved {stats['reserved']:.2f} "
        f"(peak {stats['max_reserved']:.2f}), allocated {stats['allocated']:.2f} "
        f"(peak {stats['max_allocated']:.2f})"
    )


def apply_memory_fraction(
    fraction: Optional[float], device: Optional[torch.device] = None
) -> None:
    """Cap this process' GPU allocations to `fraction` of total memory (optional)."""
    if fraction is None:
        return
    if not torch.cuda.is_available():
        return
    # set_per_process_memory_fraction requires a concrete device index, not a bare
    # torch.device("cuda") (which has index None) — resolve it to the current device.
    torch.cuda.set_per_process_memory_fraction(float(fraction), _device_index(device))


def free_memory_gib(device: Optional[torch.device] = None) -> Optional[float]:
    """Actually-free GPU memory (GiB), across all processes. None if CUDA unavailable."""
    if not torch.cuda.is_available():
        return None
    free_bytes, _total = torch.cuda.mem_get_info(_device_index(device))
    return free_bytes / _BYTES_PER_GIB


def release_cuda_cache(device: Optional[torch.device] = None) -> None:
    """Best-effort: run the GC and hand cached-but-unused blocks back to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_cuda_oom(exc: BaseException) -> bool:
    """True if `exc` (or anything in its cause/context chain) is a CUDA OOM error."""
    seen: set = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, torch.cuda.OutOfMemoryError):
            return True
        if isinstance(cur, RuntimeError) and "out of memory" in str(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def run_with_oom_retry(
    fn: Callable[[], _T],
    *,
    wait_seconds: float = 30.0,
    max_retries: Optional[int] = None,
    device: Optional[torch.device] = None,
    label: str = "",
) -> _T:
    """Call `fn`; on CUDA OOM, free the cache, wait, and retry until it succeeds.

    Motivation: on a shared GPU the free memory fluctuates as co-tenant jobs grow and
    shrink. Rather than failing a federated round when the card is momentarily
    saturated, we release our cached blocks and wait for a window where the allocation
    fits. `max_retries=None` waits indefinitely ("until we can allocate"); set an int to
    cap the attempts (the final OOM then propagates, and the server-side guard aborts).
    Only CUDA-OOM errors are retried; any other exception propagates immediately.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised unless it is a CUDA OOM
            if not is_cuda_oom(exc):
                raise
            attempt += 1
            release_cuda_cache(device)
            if max_retries is not None and attempt > max_retries:
                logger.error(
                    f"{label} CUDA OOM persisted after {max_retries} retr"
                    f"{'y' if max_retries == 1 else 'ies'}; giving up."
                )
                raise
            free_gib = free_memory_gib(device)
            free_str = "n/a" if free_gib is None else f"{free_gib:.2f} GiB free"
            cap = "" if max_retries is None else f"/{max_retries}"
            logger.warning(
                f"{label} CUDA out of memory (attempt {attempt}{cap}, {free_str}) — "
                f"waiting {wait_seconds:.0f}s for memory to free up, then retrying."
            )
            time.sleep(wait_seconds)
