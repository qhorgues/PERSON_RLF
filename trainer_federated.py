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

# Configure the CUDA caching allocator before anything can initialise a CUDA context.
# `expandable_segments` lets the allocator grow existing segments instead of carving
# new ones, which — together with disabling `empty_cache` between rounds — keeps
# `memory_reserved` flat over the run (goal: allocate once, no cudaMalloc mid-run).
# The definitive value is re-applied from config in `run()`; this default just
# guarantees a sane setting even if an import touches CUDA first.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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


def _configure_cuda_allocator(config: DictConfig) -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF before any CUDA context initialises.

    Applies to this (driver) process AND — because Ray worker processes inherit the
    driver's environment when `run_simulation` calls `ray.init` internally — to every
    client actor. This is the collision-free way to propagate the allocator config to
    the Ray workers: flwr hardcodes `ray.init(runtime_env=...)`, so passing our own
    `runtime_env` via `backend_config["init_args"]` would raise a duplicate-kwarg error.
    """
    memory_cfg = config.federated.get("memory", {})
    alloc_conf = memory_cfg.get("cuda_alloc_conf", None)
    if alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)


@hydra.main(version_base="1.3", config_path="config")
def run(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)
    # Must run before seed_everything / any model build can initialise CUDA.
    _configure_cuda_allocator(config)
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
    logger.info(
        "GPU memory management: "
        f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset)')}, "
        f"reuse_model={config.federated.memory.get('reuse_model', True)}, "
        f"empty_cache_on_eval={config.federated.memory.get('empty_cache_on_eval', False)}, "
        f"warmup={config.federated.memory.get('warmup', False)}"
    )

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

    # Coût de communication réel (fit only), écrit une seule fois -> communication.csv
    tracker = server.comm_tracker
    summary = tracker.summary(
        num_shared_parameters=server.selector.num_shared_parameters(),
        num_rounds=config.federated.num_rounds,
        num_clients=config.federated.num_clients,
    )
    total_mib = summary["fit_total_bytes"] / (1024 ** 2)
    logger.info(
        f"[Communication cost] fit total = {total_mib:.2f} MiB "
        f"(down {tracker.fit_downlink_bytes / 1024 ** 2:.2f} / "
        f"up {tracker.fit_uplink_bytes / 1024 ** 2:.2f})"
    )
    logger.log_metrics(summary, step=config.federated.num_rounds, output="communication")

    # ferme tous les sinks (fichier, run W&B, plot, ...)
    logger.close()


if __name__ == "__main__":
    run()
