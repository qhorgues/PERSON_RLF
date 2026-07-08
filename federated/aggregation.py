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


class FedAvgSubspace(_PerClientMetricsLog, FedAvg):
    """FedAvg over the shared subspace + per-client metric fan-out."""


class FedProxSubspace(_PerClientMetricsLog, FedProx):
    """FedProx over the shared subspace + per-client metric fan-out."""


def build_strategy(config, initial_parameters, evaluate_fn) -> Strategy:
    """Build the FedAvg (Phase 1) or FedProx (Phase 2) strategy."""
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
    )

    if federated_cfg.algorithm == "fedprox":
        logger.info(f"Using FedProx strategy (proximal_mu={federated_cfg.proximal_mu})")
        return FedProxSubspace(proximal_mu=float(federated_cfg.proximal_mu), **common_kwargs)

    if federated_cfg.algorithm != "fedavg":
        raise ValueError(f"Unknown federated algorithm: {federated_cfg.algorithm}")

    logger.info("Using FedAvg strategy")
    return FedAvgSubspace(**common_kwargs)
