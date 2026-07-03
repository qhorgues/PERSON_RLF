"""Non-IID identity-based partitioning of the TBPS training set across FL clients.

Implements the Dirichlet(alpha) partitioning scheme: identities (pids) - not raw
samples - are split across clients. For each identity, a per-client proportion
vector is drawn from Dirichlet(alpha); a small alpha concentrates the mass on a
few clients (strong non-IID, little ID overlap), a large alpha spreads samples
almost evenly (close to IID).
"""

import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from prettytable import PrettyTable


class IdentityPartitioner:
    def __init__(
        self,
        train_samples: List[Tuple],
        num_clients: int,
        alpha: float,
        seed: Optional[int] = None,
    ):
        """
        Args:
            train_samples: dataset.train, i.e. a list of (pid, image_id, img_path, caption).
            num_clients: number of FL clients to partition the identities across.
            alpha: Dirichlet concentration parameter. Smaller => more non-IID.
            seed: RNG seed for reproducibility.
        """
        self.train_samples = train_samples
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._pid_to_indices = self._group_by_pid()
        self._client_indices = self._dirichlet_allocation()

    def _group_by_pid(self) -> Dict[int, List[int]]:
        """Group sample indices of `train_samples` by their person id (pid)."""
        pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, sample in enumerate(self.train_samples):
            pid = sample[0]
            pid_to_indices[pid].append(idx)
        return dict(pid_to_indices)

    def _dirichlet_allocation(self) -> Dict[int, List[int]]:
        """Allocate identities (and thus their samples) to clients via Dirichlet(alpha).

        Returns a mapping client_id -> list of sample indices into `train_samples`.
        """
        client_indices: Dict[int, List[int]] = {cid: [] for cid in range(self.num_clients)}

        for pid in sorted(self._pid_to_indices.keys()):
            indices = list(self._pid_to_indices[pid])
            self._rng.shuffle(indices)

            proportions = self._rng.dirichlet(np.repeat(self.alpha, self.num_clients))
            split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            splits = np.split(indices, split_points)

            for cid, split in enumerate(splits):
                client_indices[cid].extend(int(i) for i in split)

        return client_indices

    def partition(self) -> List[List[Tuple]]:
        """Return, for each client, its list of (pid, image_id, img_path, caption)."""
        return [self.client_samples(cid) for cid in range(self.num_clients)]

    def client_samples(self, client_id: int) -> List[Tuple]:
        return [self.train_samples[idx] for idx in self._client_indices[client_id]]

    def client_num_examples(self, client_id: int) -> int:
        return len(self._client_indices[client_id])

    def summary(self) -> PrettyTable:
        """Diagnostic table: number of IDs / samples and ID overlap across clients."""
        client_pids = {
            cid: {self.train_samples[idx][0] for idx in indices}
            for cid, indices in self._client_indices.items()
        }

        table = PrettyTable(["client_id", "num_ids", "num_samples", "shared_ids_pct"])
        for cid in range(self.num_clients):
            pids = client_pids[cid]
            other_pids = set().union(
                *(p for other_cid, p in client_pids.items() if other_cid != cid)
            ) if self.num_clients > 1 else set()
            shared = pids & other_pids
            shared_pct = 100.0 * len(shared) / len(pids) if pids else 0.0
            table.add_row(
                [cid, len(pids), len(self._client_indices[cid]), f"{shared_pct:.1f}"]
            )
        return table

    @staticmethod
    def _extract_location(img_path: str) -> str:
        """Derive a canonical location label from an image path.

        Handles two folder naming conventions found in VN3K:
          - Named locations: "HoGuom_00", "HoGuom_01"  -> "HoGuom"
          - Numbered persons: "Person10", "Person10.2"  -> "Person10"

        Any trailing underscore+digit or dot+digit suffix is stripped so that
        multiple cameras at the same site are grouped under a single label.
        """
        import re
        folder = img_path.rstrip("/").split("/")[-2]
        # Strip trailing camera index: "_00", "_01", ".2", etc.
        location = re.sub(r"[_.]\d+$", "", folder)
        return location

    def distribution_matrix(self) -> PrettyTable:
        """Location x client matrix: sample counts per canonical location per client.

        Returns:
            PrettyTable where:
              - the first column is the canonical location label,
              - each subsequent column corresponds to a client ("client_<id>"),
              - the last column is the row total,
              - the last row ("TOTAL") gives the per-client sample count.
        """
        # Pre-compute per-client sample counts broken down by location.
        # client_loc_counts[cid][location] = number of samples
        client_loc_counts: Dict[int, Dict[str, int]] = {
            cid: defaultdict(int) for cid in range(self.num_clients)
        }
        all_locations: set = set()
        for cid, indices in self._client_indices.items():
            for idx in indices:
                loc = self._extract_location(self.train_samples[idx][2])
                client_loc_counts[cid][loc] += 1
                all_locations.add(loc)

        displayed_locations = sorted(all_locations)

        # Build header: one column per client + a TOTAL column
        client_headers = [f"client_{cid}" for cid in range(self.num_clients)]
        headers = ["location"] + client_headers + ["TOTAL"]
        table = PrettyTable(headers)
        table.align = "r"
        table.align["location"] = "l"

        # One row per location
        totals_per_client: Dict[int, int] = defaultdict(int)
        for loc in displayed_locations:
            row = [loc]
            loc_total = 0
            for cid in range(self.num_clients):
                n = client_loc_counts[cid].get(loc, 0)
                row.append(n)
                totals_per_client[cid] += n
                loc_total += n
            row.append(loc_total)
            table.add_row(row)

        # Footer row: per-client totals across all locations
        total_row = ["TOTAL"]
        grand_total = 0
        for cid in range(self.num_clients):
            t = totals_per_client[cid]
            total_row.append(t)
            grand_total += t
        total_row.append(grand_total)
        table.add_row(total_row)

        return table

    def print_distribution(self) -> None:
        """Print the summary table followed by the location x client distribution matrix."""
        num_pids = len(self._pid_to_indices)
        print(
            f"\n=== Federated Partition  |  "
            f"clients={self.num_clients}  alpha={self.alpha}  "
            f"total_pids={num_pids}  seed={self.seed} ===\n"
        )

        print("── Summary ──────────────────────────────────────────────────────")
        print(self.summary())

        print("\n── Distribution matrix (samples per location per client) ────────")
        print(self.distribution_matrix())

    def save_partition(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "client_indices": self._client_indices,
                    "num_clients": self.num_clients,
                    "alpha": self.alpha,
                    "seed": self.seed,
                },
                f,
            )
        logger.info(f"Saved federated partition to {path}")

    def load_partition(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._client_indices = state["client_indices"]
        self.num_clients = state["num_clients"]
        self.alpha = state["alpha"]
        self.seed = state["seed"]
        logger.info(f"Loaded federated partition from {path}")


class PerDatasetPartitioner:
    """Stub for option (b): one dataset == one client (cross-institution heterogeneity).

    Not implemented in Phase 1. Each client would load a different TBPS dataset
    (e.g. CUHK-PEDES, ICFG-PEDES, RSTPReid, VN3K_*) instead of an identity-based
    shard of a single dataset.
    """

    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "PerDatasetPartitioner is documented (option b, cross-institution "
            "heterogeneity) but not implemented in Phase 1. Use "
            "partition.type=identity_dirichlet."
        )


def build_partitioner(config, train_samples: List[Tuple]) -> IdentityPartitioner:
    """Factory: build the configured partitioner from `config.federated.partition`."""
    partition_cfg = config.federated.partition

    if partition_cfg.type == "identity_dirichlet":
        return IdentityPartitioner(
            train_samples=train_samples,
            num_clients=config.federated.num_clients,
            alpha=partition_cfg.alpha,
            seed=partition_cfg.get("seed", config.seed),
        )
    if partition_cfg.type == "per_dataset":
        return PerDatasetPartitioner()

    raise ValueError(f"Unknown federated partition type: {partition_cfg.type}")
