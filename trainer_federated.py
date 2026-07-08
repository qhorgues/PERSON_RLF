"""Hydra entrypoint for Phase 1 federated training (Flower simulation, FedAvg).

Mirrors trainer.py's setup (resolvers, seed, MLM vocab adjustment) and then:
  - builds a reference TBPSDataModule (full dataset) used for partitioning,
    the global reference model and centralized evaluation
  - partitions the training identities across clients (Dirichlet, non-IID)
  - builds the FederatedServer (global model + subspace selector + strategy)
  - builds the ClientApp (one TBPSFlowerClient per partition)
  - runs the Flower simulation (mono-GPU -> clients run sequentially)
"""

import os

import hydra
from flwr.simulation import run_simulation
from lightning.pytorch import seed_everything
from utils.logger import log as logger
from omegaconf import DictConfig, OmegaConf

from federated.client import make_client_app
from federated.partition import build_partitioner
from federated.server import FederatedServer
from federated.state import ClientStateStore
from lightning_data import TBPSDataModule
from utils.logger import setup_logging


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple, replace=True)
OmegaConf.register_new_resolver("eval", eval, replace=True)


def _reconcile_client_count(config: DictConfig, partitioner) -> None:
    """Align the config client-count knobs with the partitioner's effective client count.

    Location partitioning derives the number of clients from the data (one or more clients
    per physical location), so `num_clients` — consumed by `run_simulation(num_supernodes=)`,
    the strategy `min_*_clients` and each client's `TBPSDataModule` — must be updated to match,
    otherwise Flower would deadlock waiting for `min_available_clients`. Also fixes the case
    where `num_clients` is lowered but the hard-coded `min_available_clients` is left stale.
    """
    federated_cfg = config.federated
    effective = int(partitioner.num_clients)
    if (
        effective == federated_cfg.num_clients
        and effective >= federated_cfg.min_available_clients
    ):
        return

    logger.info(
        f"Reconciling client count: config num_clients={federated_cfg.num_clients} "
        f"-> effective={effective} (min_available/min_fit/min_evaluate clamped)."
    )
    federated_cfg.num_clients = effective
    federated_cfg.min_available_clients = effective
    federated_cfg.min_fit_clients = min(federated_cfg.min_fit_clients, effective)
    federated_cfg.min_evaluate_clients = min(
        federated_cfg.get("min_evaluate_clients", federated_cfg.min_fit_clients), effective
    )


@hydra.main(version_base="1.3", config_path="config")
def run(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    seed_everything(config.seed)

    # Modify the config if use MLM
    if config.loss.MLM:
        config.tokenizer.vocab_size += 1
        config.tokenizer.add_mask_token = True
        config.backbone.text_config.vocab_size = config.tokenizer.vocab_size

    # Ray simulation workers run in a different cwd than this (Hydra) process,
    # so relative paths must be made absolute before being captured by client_fn.
    config.dataset_root_dir = os.path.abspath(config.dataset_root_dir)
    config.backbone.path = os.path.abspath(config.backbone.path)
    config.tokenizer.pretrained_model_name_or_path = os.path.abspath(
        config.tokenizer.pretrained_model_name_or_path
    )
    config.federated.state_dir = os.path.abspath(config.federated.state_dir)

    # Build the unified façade early (console/file/csv/wandb/plot + interception) so
    # the config dump, the partition text AND the partition metrics reach the sinks.
    setup_logging(config)

    logger.info(f"Federated config:\n{OmegaConf.to_yaml(config.federated)}")

    # Reference data module: full training set, used for partitioning, the
    # global reference model and centralized (global) evaluation.
    dm = TBPSDataModule(config)
    dm.setup()

    partitioner = build_partitioner(config, dm.dataset.train)
    _reconcile_client_count(config, partitioner)
    partitioner.print_distribution()
    partitioner.log_partition_metrics()  # per-client partition stats -> partition.csv
    logger.info(
        f"Federated partition ({config.federated.partition.type}, "
        f"num_clients={partitioner.num_clients}):\n{partitioner.summary()}"
    )
    partitioner.save_partition("federated_partition.pkl")

    state_store = ClientStateStore(config.federated.state_dir)
    state_store.clear()

    server = FederatedServer(config, dm)
    server_app = server.make_server_app()
    client_app = make_client_app(config, partitioner, server.selector, state_store)

    logger.info(
        f"Starting Flower simulation: {config.federated.num_clients} clients, "
        f"{config.federated.num_rounds} rounds, subspace={config.federated.subspace} "
        f"({server.selector.num_shared_parameters():,} shared parameters)"
    )

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config.federated.num_clients,
        backend_config={"client_resources": OmegaConf.to_container(config.federated.client_resources)},
    )

    _log_round_history(config, server)


def _log_round_history(config: DictConfig, server: FederatedServer) -> None:
    """Final summary: convergence/comm-cost are logged per round in `evaluate_fn`;
    this just closes out the run (and all sinks: file, W&B, plot, ...)."""
    logger.info(
        f"Federated run finished: {config.federated.num_rounds} rounds, "
        f"{config.federated.num_clients} clients, algorithm={config.federated.algorithm}, "
        f"subspace={config.federated.subspace} "
        f"({server.selector.num_shared_parameters():,} shared parameters/round)."
    )

    # ferme tous les sinks (fichier, run W&B, plot, ...)
    logger.close()


if __name__ == "__main__":
    run()
