"""FedAvg/FedProx strategy construction and metric aggregation for the subspace FL run.

From Flower's point of view the "model" IS the shared subspace (whatever
`SubspaceSelector` extracts) - so the native `FedAvg`/`FedProx` parameter
aggregation (weighted by num_examples, FedSH eq. 21) applies unchanged.
`weighted_average` is provided for completeness / custom aggregation but is not
required for Phase 1.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from utils.logger import log as logger
from flwr.common import Metrics, NDArrays, Scalar
from flwr.server.strategy import FedAvg, FedProx, Strategy


def weighted_average(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """FedAvg aggregation of NDArrays, weighted by |D^k| (FedSH eq. 21)."""
    total_examples = sum(num_examples for _, num_examples in results)
    weighted_layers = [
        [layer * num_examples for layer in arrays] for arrays, num_examples in results
    ]
    return [
        np.sum(layers, axis=0) / total_examples for layers in zip(*weighted_layers)
    ]


# Routing key injected by the client (federated/client.py); not a real metric.
_CLIENT_ID_KEY = "client_id"


class CommunicationTracker:
    """Accumulates the real client<->server fit traffic over the whole run.

    Measured from Flower's own serialization: a shared subspace is carried as a
    `flwr.common.Parameters` whose `.tensors` is a `List[bytes]`, so
    `sum(len(t) for t in parameters.tensors)` is the exact wire size (numpy `.npy`
    bytes, headers included). Downlink is summed in `configure_fit` (server->clients),
    uplink in `aggregate_fit` (clients->server). Evaluation traffic is out of scope.

    Lives in the driver process (same as the strategy in simulation), so the
    trainer can read the totals after `run_simulation`.
    """

    def __init__(self) -> None:
        self.fit_downlink_bytes: int = 0
        self.fit_uplink_bytes: int = 0
        self.num_fit_rounds: int = 0

    def add_downlink(self, num_bytes: int) -> None:
        self.fit_downlink_bytes += int(num_bytes)

    def add_uplink(self, num_bytes: int) -> None:
        self.fit_uplink_bytes += int(num_bytes)

    def summary(self, num_shared_parameters: int, num_rounds: int, num_clients: int) -> Dict[str, Scalar]:
        total = self.fit_downlink_bytes + self.fit_uplink_bytes
        return {
            "fit_downlink_bytes": self.fit_downlink_bytes,
            "fit_uplink_bytes": self.fit_uplink_bytes,
            "fit_total_bytes": total,
            "fit_total_MB": total / (1024 ** 2),
            "num_shared_parameters": num_shared_parameters,
            "num_rounds": num_rounds,
            "num_clients": num_clients,
        }


def _parameters_nbytes(parameters) -> int:
    """Serialized wire size of a Flower `Parameters` (sum of its tensors' byte lengths)."""
    return sum(len(tensor) for tensor in parameters.tensors)


def _weighted_metrics_average(
    metrics: List[Tuple[int, Metrics]],
) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = set()
    for _, client_metrics in metrics:
        keys.update(client_metrics.keys())
    keys.discard(_CLIENT_ID_KEY)  # routing tag, not averaged

    return {
        key: sum(
            num_examples * float(client_metrics.get(key, 0.0))
            for num_examples, client_metrics in metrics
        )
        / total_examples
        for key in keys
    }


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Weighted average of client training metrics (e.g. total_loss)."""
    return _weighted_metrics_average(metrics)


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Weighted average of client-local evaluation metrics (R1/R5/R10/mAP/mINP)."""
    return _weighted_metrics_average(metrics)


def make_fit_config_fn(config) -> Callable[[int], Dict[str, Scalar]]:
    """Build `on_fit_config_fn`: per-round config sent to clients in `fit`."""
    federated_cfg = config.federated
    proximal_mu = (
        float(federated_cfg.proximal_mu) if federated_cfg.algorithm == "fedprox" else 0.0
    )

    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        return {
            "server_round": server_round,
            "subspace": federated_cfg.subspace,
            "stn_enabled": bool(federated_cfg.stn.enabled),
            "proximal_mu": proximal_mu,
        }

    return on_fit_config_fn


class _PerClientMetricsLog:
    """Mixin: fan-out each client's returned metrics to a per-client `output`.

    Runs server-side (main process, façade configured with the run's sinks), so it
    sidesteps the fact that Ray simulation workers have an unconfigured façade.
    `aggregate_fit`/`aggregate_evaluate` are the only strategy hooks that receive
    `server_round`, hence the override (vs. the `*_metrics_aggregation_fn`).
    """

    def _log_per_client(self, server_round: int, results: list) -> None:
        for _client_proxy, res in results:
            metrics = getattr(res, "metrics", None) or {}
            cid = metrics.get(_CLIENT_ID_KEY)
            if cid is None:
                continue
            payload = {k: v for k, v in metrics.items() if k != _CLIENT_ID_KEY}
            if payload:
                logger.log_metrics(payload, step=server_round, output=f"client_{int(cid)}")

    def aggregate_fit(self, server_round, results, failures):
        self._log_per_client(server_round, results)
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        self._log_per_client(server_round, results)
        return super().aggregate_evaluate(server_round, results, failures)


class _CommTracking:
    """Mixin: count the real fit traffic into a shared `CommunicationTracker`.

    Reads the serialized byte size of the `Parameters` Flower actually sends/receives
    (see `CommunicationTracker`): downlink from the `FitIns` produced by `configure_fit`,
    uplink from the `FitRes` collected by `aggregate_fit`. No-op if `comm_tracker` is None.
    Placed before `_PerClientMetricsLog` in the MRO so both `super()` chains reach FedAvg.
    """

    def __init__(self, *args, comm_tracker: Optional[CommunicationTracker] = None, **kwargs):
        self._comm_tracker = comm_tracker
        super().__init__(*args, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = super().configure_fit(server_round, parameters, client_manager)
        if self._comm_tracker is not None:
            for _client_proxy, fit_ins in instructions:
                self._comm_tracker.add_downlink(_parameters_nbytes(fit_ins.parameters))
            self._comm_tracker.num_fit_rounds += 1
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        if self._comm_tracker is not None:
            for _client_proxy, fit_res in results:
                self._comm_tracker.add_uplink(_parameters_nbytes(fit_res.parameters))
        return super().aggregate_fit(server_round, results, failures)


class _AbortOnFrozenGlobal:
    """Fail fast when a fit round produces no usable update.

    If every client `fit` fails (the common case: all clients hit CUDA OOM), Flower's
    FedAvg/FedProx `aggregate_fit` returns `None` parameters and Flower silently keeps
    the previous global model. Over many rounds this yields a *perfectly flat* metric
    curve (the global model stays frozen at W0) instead of an obvious error — exactly
    the "constant curves" symptom. We turn that silent no-op into a loud abort so the
    real cause (client-side OOM / failures) surfaces immediately.

    First in the MRO so it observes the final aggregated result and can abort the run.
    """

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            logger.warning(
                f"[round {server_round}] {len(failures)} client fit failure(s) "
                f"vs {len(results)} success(es) — inspect client logs for CUDA OOM."
            )
        aggregated = super().aggregate_fit(server_round, results, failures)
        agg_params = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        if agg_params is None:
            logger.error(
                f"[round {server_round}] no client fit succeeded "
                f"({len(failures)} failure(s)); the global model would stay frozen at "
                "the previous parameters and every subsequent round would report the "
                "same (flat) metrics. Aborting instead of training nothing — the most "
                "common cause is CUDA out of memory during client fit."
            )
            raise RuntimeError(
                f"Federated fit round {server_round}: 0 successful client updates "
                f"({len(failures)} failures). Aborting to avoid a silently frozen "
                "global model (check for CUDA OOM in the client logs)."
            )
        return aggregated


class FedAvgSubspace(_AbortOnFrozenGlobal, _CommTracking, _PerClientMetricsLog, FedAvg):
    """FedAvg over the shared subspace + per-client metric fan-out + comm-cost tracking."""


class FedProxSubspace(_AbortOnFrozenGlobal, _CommTracking, _PerClientMetricsLog, FedProx):
    """FedProx over the shared subspace + per-client metric fan-out + comm-cost tracking."""


def build_strategy(config, initial_parameters, evaluate_fn, comm_tracker=None) -> Strategy:
    """Build the FedAvg (Phase 1) or FedProx (Phase 2) strategy.

    `comm_tracker` (optional `CommunicationTracker`): accumulates the real fit
    traffic; the mixin pops it before delegating to FedAvg/FedProx.
    """
    federated_cfg = config.federated

    common_kwargs = dict(
        fraction_fit=federated_cfg.fraction_fit,
        fraction_evaluate=federated_cfg.fraction_evaluate,
        min_fit_clients=federated_cfg.min_fit_clients,
        min_evaluate_clients=federated_cfg.get("min_evaluate_clients", federated_cfg.min_fit_clients),
        min_available_clients=federated_cfg.min_available_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=make_fit_config_fn(config),
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        comm_tracker=comm_tracker,
    )

    if federated_cfg.algorithm == "fedprox":
        logger.info(f"Using FedProx strategy (proximal_mu={federated_cfg.proximal_mu})")
        return FedProxSubspace(proximal_mu=float(federated_cfg.proximal_mu), **common_kwargs)

    if federated_cfg.algorithm != "fedavg":
        raise ValueError(f"Unknown federated algorithm: {federated_cfg.algorithm}")

    logger.info("Using FedAvg strategy")
    return FedAvgSubspace(**common_kwargs)
