"""FedAvg/FedProx strategy construction and metric aggregation for the subspace FL run.

From Flower's point of view the "model" IS the shared subspace (whatever
`SubspaceSelector` extracts) - so the native `FedAvg`/`FedProx` parameter
aggregation (weighted by num_examples, FedSH eq. 21) applies unchanged.
`weighted_average` is provided for completeness / custom aggregation but is not
required for Phase 1.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
from loguru import logger
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


def _weighted_metrics_average(
    metrics: List[Tuple[int, Metrics]],
) -> Dict[str, Scalar]:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = set()
    for _, client_metrics in metrics:
        keys.update(client_metrics.keys())

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
        return FedProx(proximal_mu=float(federated_cfg.proximal_mu), **common_kwargs)

    if federated_cfg.algorithm != "fedavg":
        raise ValueError(f"Unknown federated algorithm: {federated_cfg.algorithm}")

    logger.info("Using FedAvg strategy")
    return FedAvg(**common_kwargs)
