"""Partitioning of the TBPS training set across FL clients.

Two schemes are provided:

- `IdentityPartitioner` (``type: identity_dirichlet``) — synthetic non-IID: identities
  (pids), not raw samples, are split across clients via a per-identity Dirichlet(alpha)
  proportion vector. Small alpha => strong non-IID (little ID overlap); large alpha =>
  close to IID. Every client still draws from the *same* physical sources.

- `LocationPartitioner` (``type: location``) — realistic "per-source" federation: each
  client is *anchored* to a physical image source (a site / camera location, recovered
  from ``img_path``). Large locations are split across several disjoint clients so a
  dominant site does not sit in one client; a shared ``other`` filler pool (messy folders
  + tiny sites) is distributed to top up the smallest clients so sizes stay homogeneous.
"""

import csv
import heapq
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from utils.logger import log as logger
from prettytable import PrettyTable


def extract_location(img_path: str) -> str:
    """Derive a canonical location label from an image path.

    Handles the folder naming conventions found in VN3K:
      - Named sites (image directly under the site folder):
        ``"HoGuom_00/xxx.jpg"``, ``"HoGuom_01/xxx.jpg"`` -> ``"HoGuom"``
        (trailing camera suffix ``_NN`` / ``.N`` is stripped so several cameras of one
        site collapse to a single label).
      - Per-identity folders (``TEST_DATA``/``TRAINNING_DATA`` layout), whose immediate
        parent is ``PersonN`` / ``PersonN.2`` and is *not* a physical location ->
        ``"other"`` (a catch-all bucket used as filler by `LocationPartitioner`).
    """
    parent = img_path.rstrip("/").split("/")[-2]
    if re.fullmatch(r"(?i)person\d+(\.\d+)?", parent):
        return "other"
    return re.sub(r"[_.]\d+$", "", parent)


def extract_camera(img_path: str) -> str:
    """Derive a camera-level label from an image path.

    Same buckets as `extract_location` but *keeps* the trailing camera suffix
    (``_NN`` / ``.N``), so cameras of a single site stay distinct
    (``HoGuom_00`` vs ``HoGuom_01``). Per-identity folders still collapse to
    ``"other"``.
    """
    parent = img_path.rstrip("/").split("/")[-2]
    if re.fullmatch(r"(?i)person\d+(\.\d+)?", parent):
        return "other"
    return parent


class _IndexPartitioner:
    """Shared machinery for index-based FL partitioners.

    Subclasses must populate `self.train_samples`, `self.num_clients` and
    `self._client_indices` (client_id -> list of sample indices into `train_samples`).
    """

    train_samples: List[Tuple]
    num_clients: int
    _client_indices: Dict[int, List[int]]

    def partition(self) -> List[List[Tuple]]:
        """Return, for each client, its list of (pid, image_id, img_path, caption)."""
        return [self.client_samples(cid) for cid in range(self.num_clients)]

    def client_samples(self, client_id: int) -> List[Tuple]:
        return [self.train_samples[idx] for idx in self._client_indices[client_id]]

    def client_num_examples(self, client_id: int) -> int:
        return len(self._client_indices[client_id])

    def _distribution_counts(
        self, label_fn=extract_location
    ) -> Tuple[List[str], Dict[int, Dict[str, int]]]:
        """Shared counting for the label x client distribution.

        Returns `(sorted_labels, client_counts)` where `client_counts[cid][label]` is
        the number of samples of that canonical label (via `label_fn`) held by client
        `cid`. Single source of truth for `distribution_matrix` (PrettyTable) and
        `write_distribution_csv` (CSV).
        """
        client_counts: Dict[int, Dict[str, int]] = {
            cid: defaultdict(int) for cid in range(self.num_clients)
        }
        all_labels: set = set()
        for cid, indices in self._client_indices.items():
            for idx in indices:
                label = label_fn(self.train_samples[idx][2])
                client_counts[cid][label] += 1
                all_labels.add(label)
        return sorted(all_labels), client_counts

    def distribution_matrix(
        self,
        label_fn=extract_location,
        label_name: str = "location",
    ) -> PrettyTable:
        """Label x client matrix: sample counts per canonical label per client.

        Rows are canonical labels (via `label_fn`, e.g. `extract_location` for site-level
        or `extract_camera` for camera-level), columns are clients plus a row/column
        total; the last row ("TOTAL") gives the per-client sample count.
        """
        displayed_locations, client_loc_counts = self._distribution_counts(label_fn)

        client_headers = [f"client_{cid}" for cid in range(self.num_clients)]
        headers = [label_name] + client_headers + ["TOTAL"]
        table = PrettyTable(headers)
        table.align = "r"
        table.align[label_name] = "l"

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

        total_row = ["TOTAL"]
        grand_total = 0
        for cid in range(self.num_clients):
            t = totals_per_client[cid]
            total_row.append(t)
            grand_total += t
        total_row.append(grand_total)
        table.add_row(total_row)

        return table

    def write_distribution_csv(
        self,
        path: str,
        label_fn=extract_location,
        label_name: str = "location",
    ) -> None:
        """Persist the label x client distribution matrix to `path` as CSV.

        Same content as `distribution_matrix` (a PrettyTable) but saved to disk: header
        ``<label_name>,client_0,…,client_{N-1},TOTAL``, one row per label, a final
        ``TOTAL`` row of per-client totals. Read back by `plot_distribution.py` to render
        the horizontal stacked bar chart.
        """
        labels, client_counts = self._distribution_counts(label_fn)
        headers = (
            [label_name]
            + [f"client_{cid}" for cid in range(self.num_clients)]
            + ["TOTAL"]
        )
        totals_per_client = [0] * self.num_clients
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for label in labels:
                row: List = [label]
                label_total = 0
                for cid in range(self.num_clients):
                    n = client_counts[cid].get(label, 0)
                    row.append(n)
                    totals_per_client[cid] += n
                    label_total += n
                row.append(label_total)
                writer.writerow(row)
            writer.writerow(["TOTAL"] + totals_per_client + [sum(totals_per_client)])
        logger.info(f"Saved distribution matrix ({label_name}) to {path}")


class IdentityPartitioner(_IndexPartitioner):
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

    def print_distribution(self) -> None:
        """Print the summary table followed by the location x client distribution matrix."""
        num_pids = len(self._pid_to_indices)
        logger.info(
            f"\n=== Federated Partition  |  "
            f"clients={self.num_clients}  alpha={self.alpha}  "
            f"total_pids={num_pids}  seed={self.seed} ===\n"
            "── Summary ──────────────────────────────────────────────────────\n"
            f"{self.summary()}\n"
            "── Distribution matrix (samples per site per client) ────────────\n"
            f"{self.distribution_matrix()}\n"
            "── Distribution matrix (samples per camera per client) ──────────\n"
            f"{self.distribution_matrix(extract_camera, 'camera')}"
        )
        self.write_distribution_csv("distribution_site.csv")
        self.write_distribution_csv("distribution_camera.csv", extract_camera, "camera")

    def log_partition_metrics(self, output: str = "partition") -> None:
        """Emit per-client partition stats to a dedicated metrics output (-> `<output>.csv`)."""
        client_pids = {
            cid: {self.train_samples[idx][0] for idx in indices}
            for cid, indices in self._client_indices.items()
        }
        for cid in range(self.num_clients):
            pids = client_pids[cid]
            other_pids = set().union(
                *(p for other_cid, p in client_pids.items() if other_cid != cid)
            ) if self.num_clients > 1 else set()
            shared_pct = 100.0 * len(pids & other_pids) / len(pids) if pids else 0.0
            logger.log_metrics(
                {
                    "num_ids": len(pids),
                    "num_samples": len(self._client_indices[cid]),
                    "shared_ids_pct": shared_pct,
                },
                step=cid,
                output=output,
            )

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


class LocationPartitioner(_IndexPartitioner):
    """Per-source partitioning: clients anchored to a physical location, homogenized.

    `num_clients` (config) is a *target granularity*: it sets the ideal client size
    ``per_client_target = len(train_samples) / num_clients``. Each named location gets
    ``max(1, round(len(loc) / per_client_target))`` disjoint, identity-coherent, size-
    balanced clients (so a dominant site does not sit in one client). The ``other`` pool
    (per-identity folders + sites below `min_samples`) is then distributed to the smallest
    clients as filler, so client sizes converge. The effective client count is derived and
    exposed as `self.num_clients`.
    """

    def __init__(
        self,
        train_samples: List[Tuple],
        num_clients: int,
        min_samples: int = 0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            train_samples: dataset.train, i.e. a list of (pid, image_id, img_path, caption).
            num_clients: *target* number of clients (granularity); the effective count is
                derived from the data and stored in `self.num_clients`.
            min_samples: named locations with fewer than this many samples are demoted into
                the `other` filler pool instead of getting their own client. 0 keeps all.
            seed: RNG seed for reproducibility.
        """
        self.train_samples = train_samples
        self.target_num_clients = max(1, num_clients)
        self.min_samples = min_samples
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        (
            self._client_indices,
            self._client_location,
            self._client_filler,
        ) = self._allocate()
        self.num_clients = len(self._client_indices)

    def _allocate(self) -> Tuple[Dict[int, List[int]], Dict[int, str], Dict[int, int]]:
        # 1. group sample indices by physical location
        loc_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, sample in enumerate(self.train_samples):
            loc_to_indices[extract_location(sample[2])].append(idx)

        # 2. peel off the 'other' filler pool; demote tiny named sites into it
        other_pool: List[int] = list(loc_to_indices.pop("other", []))
        named: Dict[str, List[int]] = {}
        for loc, indices in loc_to_indices.items():
            if len(indices) < self.min_samples:
                other_pool.extend(indices)
            else:
                named[loc] = indices

        per_client_target = len(self.train_samples) / self.target_num_clients

        # 3. anchor clients to named locations, splitting large ones
        client_indices: Dict[int, List[int]] = {}
        client_location: Dict[int, str] = {}
        cid = 0
        for loc in sorted(named.keys()):
            indices = named[loc]
            k = max(1, round(len(indices) / per_client_target)) if per_client_target else 1
            for shard in self._balanced_pid_shards(indices, k):
                client_indices[cid] = list(shard)
                client_location[cid] = loc
                cid += 1

        # Edge case: no named locations (everything is 'other') -> split the pool directly
        if not client_indices:
            k = max(1, min(self.target_num_clients, len(other_pool) or 1))
            for shard in self._even_shards(other_pool, k):
                client_indices[cid] = list(shard)
                client_location[cid] = "other"
                cid += 1
            other_pool = []

        # 4. homogenize: hand the 'other' filler to the smallest clients first
        client_filler = {c: 0 for c in client_indices}
        self._distribute_filler(client_indices, client_filler, other_pool)

        return client_indices, client_location, client_filler

    def _balanced_pid_shards(self, indices: List[int], k: int) -> List[List[int]]:
        """Split `indices` into up to `k` disjoint, identity-coherent, size-balanced shards.

        Identities are never split across shards (a pid's samples stay together), and `k`
        is clamped to the number of distinct pids so no shard is left empty.
        """
        pid_groups: Dict[int, List[int]] = defaultdict(list)
        for idx in indices:
            pid_groups[self.train_samples[idx][0]].append(idx)

        groups = list(pid_groups.values())
        self._rng.shuffle(groups)  # reproducible tie-breaking
        k = max(1, min(k, len(groups)))
        if k == 1:
            return [list(indices)]

        groups.sort(key=len, reverse=True)
        bins: List[List[int]] = [[] for _ in range(k)]
        sizes = [0] * k
        for group in groups:
            j = int(np.argmin(sizes))
            bins[j].extend(group)
            sizes[j] += len(group)
        return bins

    def _even_shards(self, indices: List[int], k: int) -> List[List[int]]:
        indices = list(indices)
        self._rng.shuffle(indices)
        return [list(shard) for shard in np.array_split(indices, k)]

    def _distribute_filler(
        self,
        client_indices: Dict[int, List[int]],
        client_filler: Dict[int, int],
        filler_pool: List[int],
    ) -> None:
        """Greedily assign filler samples to the currently-smallest client (disjoint)."""
        if not filler_pool:
            return
        filler = list(filler_pool)
        self._rng.shuffle(filler)
        heap = [(len(idxs), cid) for cid, idxs in client_indices.items()]
        heapq.heapify(heap)
        for idx in filler:
            size, cid = heapq.heappop(heap)
            client_indices[cid].append(idx)
            client_filler[cid] += 1
            heapq.heappush(heap, (size + 1, cid))

    def summary(self) -> PrettyTable:
        """Per-client diagnostic: anchor location, #ids, location vs filler sample counts."""
        table = PrettyTable(
            ["client_id", "location", "num_ids", "num_location", "num_filler", "num_samples"]
        )
        table.align["location"] = "l"
        for cid in range(self.num_clients):
            indices = self._client_indices[cid]
            pids = {self.train_samples[idx][0] for idx in indices}
            filler = self._client_filler.get(cid, 0)
            table.add_row(
                [
                    cid,
                    self._client_location.get(cid, "?"),
                    len(pids),
                    len(indices) - filler,
                    filler,
                    len(indices),
                ]
            )
        return table

    def print_distribution(self) -> None:
        num_locations = len(set(self._client_location.values()))
        logger.info(
            f"\n=== Federated Location Partition  |  "
            f"clients={self.num_clients}  target={self.target_num_clients}  "
            f"locations={num_locations}  min_samples={self.min_samples}  seed={self.seed} ===\n"
            "── Summary ──────────────────────────────────────────────────────\n"
            f"{self.summary()}\n"
            "── Distribution matrix (samples per site per client) ────────────\n"
            f"{self.distribution_matrix()}\n"
            "── Distribution matrix (samples per camera per client) ──────────\n"
            f"{self.distribution_matrix(extract_camera, 'camera')}"
        )
        self.write_distribution_csv("distribution_site.csv")
        self.write_distribution_csv("distribution_camera.csv", extract_camera, "camera")

    def log_partition_metrics(self, output: str = "partition") -> None:
        """Emit per-client partition stats to a dedicated metrics output (-> `<output>.csv`)."""
        for cid in range(self.num_clients):
            indices = self._client_indices[cid]
            pids = {self.train_samples[idx][0] for idx in indices}
            filler = self._client_filler.get(cid, 0)
            logger.log_metrics(
                {
                    "num_ids": len(pids),
                    "num_location": len(indices) - filler,
                    "num_filler": filler,
                    "num_samples": len(indices),
                },
                step=cid,
                output=output,
            )

    def save_partition(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "client_indices": self._client_indices,
                    "client_location": self._client_location,
                    "client_filler": self._client_filler,
                    "num_clients": self.num_clients,
                    "target_num_clients": self.target_num_clients,
                    "min_samples": self.min_samples,
                    "seed": self.seed,
                },
                f,
            )
        logger.info(f"Saved federated location partition to {path}")

    def load_partition(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._client_indices = state["client_indices"]
        self._client_location = state["client_location"]
        self._client_filler = state["client_filler"]
        self.num_clients = state["num_clients"]
        self.target_num_clients = state["target_num_clients"]
        self.min_samples = state["min_samples"]
        self.seed = state["seed"]
        logger.info(f"Loaded federated location partition from {path}")


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
            "partition.type=identity_dirichlet or partition.type=location."
        )


def build_partitioner(config, train_samples: List[Tuple]) -> _IndexPartitioner:
    """Factory: build the configured partitioner from `config.federated.partition`."""
    partition_cfg = config.federated.partition

    if partition_cfg.type == "identity_dirichlet":
        return IdentityPartitioner(
            train_samples=train_samples,
            num_clients=config.federated.num_clients,
            alpha=partition_cfg.alpha,
            seed=partition_cfg.get("seed", config.seed),
        )
    if partition_cfg.type == "location":
        return LocationPartitioner(
            train_samples=train_samples,
            num_clients=config.federated.num_clients,
            min_samples=partition_cfg.get("min_samples", 0),
            seed=partition_cfg.get("seed", config.seed),
        )
    if partition_cfg.type == "per_dataset":
        return PerDatasetPartitioner()

    raise ValueError(f"Unknown federated partition type: {partition_cfg.type}")
