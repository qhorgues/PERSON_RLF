"""Compare two federated-learning runs (offline) — one image per client + global + partition.

Usage:
    uv run plot_fl_compare.py <runA> <runB> [options]
    uv run plot_fl_compare.py 2026-07-07/08-26-24 2026-07-06/11-04-32
    uv run plot_fl_compare.py <runA> <runB> --metrics val_t2i_R1 total_loss
    uv run plot_fl_compare.py <runA> <runB> --all --no-partition

Chaque run est un dossier Hydra `outputs/<date>/<heure>/` contenant `client_*.csv`,
`metrics.csv` et `partition.csv`. Le script produit, dans le dossier de sortie :

  - client_<cid>.png : matrice de courbes des métriques du client (2 courbes = 2 runs) ;
  - metrics.png      : matrice de courbes des métriques globales serveur (2 runs) ;
  - partition.png    : barres horizontales empilées (répartition du dataset par client).

Le rendu des matrices de courbes est mutualisé avec `plot_fl_results.py` et le RealtimePlotSink
via `utils.logger.plot_panels` (aucune duplication de la logique de tracé). `read_csv` /
`to_float` sont réutilisés de `plot_fl_results.py`.
"""

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless backend: we save PNGs

from plot_fl_results import read_csv, to_float
from utils.logger.plot_panels import render_figure

# Fixed per-run styling so the two runs stay recognizable across every panel.
COLOR_A, STYLE_A = "tab:blue", "o-"
COLOR_B, STYLE_B = "tab:orange", "s-"

# Curated, readable default set of per-client metrics (skipped if absent from a CSV).
DEFAULT_CLIENT_METRICS = [
    "val_t2i_R1", "val_t2i_R5", "val_t2i_R10", "val_t2i_mAP", "val_t2i_mINP",
    "val_i2t_R1", "val_i2t_R5", "val_i2t_R10", "val_i2t_mAP", "val_i2t_mINP",
    "val_score", "total_loss",
]

# Global (server centralized-eval) metrics from metrics.csv.
GLOBAL_METRICS = ["global_R1", "global_R5", "global_R10", "global_mAP", "global_mINP", "global_loss"]

# Partition columns (partition.csv) — one horizontal bar per client, one panel per metric.
PARTITION_METRICS = ["num_samples", "num_ids", "shared_ids_pct"]

# Human-readable titles/labels; falls back to the raw column name when unmapped.
PRETTY = {
    "val_t2i_R1": "t2i Recall@1", "val_t2i_R5": "t2i Recall@5", "val_t2i_R10": "t2i Recall@10",
    "val_t2i_mAP": "t2i mAP", "val_t2i_mINP": "t2i mINP",
    "val_i2t_R1": "i2t Recall@1", "val_i2t_R5": "i2t Recall@5", "val_i2t_R10": "i2t Recall@10",
    "val_i2t_mAP": "i2t mAP", "val_i2t_mINP": "i2t mINP", "val_score": "val_score",
    "total_loss": "total loss", "nitc_loss": "nitc loss", "citc_loss": "citc loss",
    "ritc_loss": "ritc loss", "ss_loss": "ss loss", "stn_loss": "stn loss", "grad_norm": "grad norm",
    "global_R1": "Global Recall@1", "global_R5": "Global Recall@5", "global_R10": "Global Recall@10",
    "global_mAP": "Global mAP", "global_mINP": "Global mINP", "global_loss": "Global loss",
    "num_ids": "# identities", "num_samples": "# samples", "shared_ids_pct": "shared IDs (%)",
}


def pretty(name):
    return PRETTY.get(name, name)


# --------------------------------------------------------------------------- I/O helpers

def _resolve_run_dir(arg, outputs_root):
    """Return a run directory: `arg` as-is if it exists, else `<outputs_root>/<arg>`."""
    if os.path.isdir(arg):
        return arg
    candidate = os.path.join(outputs_root, arg)
    if os.path.isdir(candidate):
        return candidate
    raise SystemExit(f"Run directory not found: {arg!r} (also tried {candidate!r})")


def _short_name(run_dir):
    """`.../outputs/2026-07-07/08-26-24` -> `2026-07-07/08-26-24` (for labels/slugs)."""
    parts = os.path.normpath(run_dir).split(os.sep)
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _cols_or_none(path):
    """read_csv(path) as {column: [str, ...]}, or None if the file is missing."""
    return read_csv(path) if os.path.isfile(path) else None


def _series_from_cols(cols, metrics):
    """{metric: (xs, ys)} for each metric present with >=1 non-empty value; xs = round/step."""
    if not cols:
        return {}
    x_col = "round" if "round" in cols else "step"
    xs = to_float(cols[x_col])
    out = {}
    for m in metrics:
        if m in cols:
            ys = to_float(cols[m])
            if any(v is not None for v in ys):
                out[m] = (xs, ys)
    return out


def _discover_all_metrics(*cols_dicts):
    """Union of numeric columns across the given CSVs, in order, minus noisy duplicates."""
    names, seen = [], set()
    for cols in cols_dicts:
        if not cols:
            continue
        for name in cols:  # preserve CSV column order
            if name in seen or name in ("step", "round"):
                continue
            if name.endswith("_step") or name.endswith("_epoch"):
                continue
            seen.add(name)
            names.append(name)
    return names


def _client_ids(run_dir):
    """Sorted client ids for which a `client_<id>.csv` exists in the run dir."""
    ids = []
    for path in glob.glob(os.path.join(run_dir, "client_*.csv")):
        m = re.match(r"client_(\d+)\.csv$", os.path.basename(path))
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


# --------------------------------------------------------------------------- curve figures

def plot_curve_comparison(series_a, series_b, metrics, out_path, title, label_a, label_b, ncols):
    """One panel per metric, two curves (run A / run B), via the shared `render_figure`."""
    import matplotlib.pyplot as plt

    history, panels = {}, []
    for m in metrics:
        has = False
        if m in series_a:
            history[f"{m}::A"] = series_a[m]
            has = True
        if m in series_b:
            history[f"{m}::B"] = series_b[m]
            has = True
        if not has:
            continue
        panels.append((pretty(m), pretty(m), [
            (f"{m}::A", STYLE_A, COLOR_A, label_a),
            (f"{m}::B", STYLE_B, COLOR_B, label_b),
        ]))

    if not panels:
        print(f"  [skip] {os.path.basename(out_path)} : no metric with data in either run")
        return
    fig, _ = render_figure(history, panels=panels, out_path=out_path, title=title,
                           xlabel="round", ncols=ncols)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# --------------------------------------------------------------------------- partition figure

def _parse_partition(cols):
    """(client_ids, {metric: [values]}) from a partition.csv column dict, or None."""
    if not cols:
        return None
    x_col = "round" if "round" in cols else "step"
    clients = [int(round(v)) for v in to_float(cols[x_col]) if v is not None]
    data = {c: to_float(cols[c]) for c in PARTITION_METRICS if c in cols}
    return clients, data


def _client_bar(ax, metric, runs):
    """One horizontal bar per client (y = clients); grouped by run when there are two."""
    ids = sorted({cid for _, clients, _ in runs for cid in clients})
    index = {cid: i for i, cid in enumerate(ids)}
    nb = len(runs)
    height = 0.8 / max(nb, 1)
    start = -(nb - 1) / 2.0  # center the run-bars around each client's y position
    for k, (label, clients, data) in enumerate(runs[::-1]):  # run A first
        color = (COLOR_A, COLOR_B)[k] if k < 2 else None
        values = data.get(metric)
        if values is None:
            continue
        ys, ws = [], []
        for cid, v in zip(clients, values):
            if v is not None:
                ys.append(index[cid] + (start + k) * height)
                ws.append(v)
        ax.barh(ys, ws, height=height, color=color, label=label)
    ax.set_yticks(list(index.values()))
    ax.set_yticklabels([f"client {cid}" for cid in ids])
    ax.invert_yaxis()  # client 0 at the top
    ax.set_title(pretty(metric))
    ax.set_xlabel(metric)
    ax.set_ylabel("client")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)


def plot_partition(dir_a, dir_b, label_a, label_b, out_path):
    """partition.png : one horizontal bar per client (grouped per run), one panel per metric."""
    import matplotlib.pyplot as plt

    pa = _parse_partition(_cols_or_none(os.path.join(dir_a, "partition.csv")))
    pb = _parse_partition(_cols_or_none(os.path.join(dir_b, "partition.csv")))
    if pa is None and pb is None:
        print("  [skip] partition.png : no partition.csv in either run")
        return

    runs = []  # (label, clients, data); run B first so run A draws on top of each group
    if pb is not None:
        runs.append((label_b, pb[0], pb[1]))
    if pa is not None:
        runs.append((label_a, pa[0], pa[1]))

    present = {m for _, _, data in runs for m in data}
    metrics = [m for m in PARTITION_METRICS if m in present]
    if not metrics:
        print("  [skip] partition.png : no known partition columns")
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        _client_bar(ax, m, runs)

    fig.suptitle(f"Partition (per client) — {label_a} vs {label_b}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# --------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runA", help="first run dir (e.g. 2026-07-07/08-26-24, resolved under outputs/)")
    p.add_argument("runB", help="second run dir")
    p.add_argument("--outputs-root", default="outputs", help="base dir for relative run args")
    p.add_argument("--labels", "--titles", dest="labels", nargs=2, metavar=("TITLE_A", "TITLE_B"),
                   help="title/label for each run, shown in the suptitle + legends "
                        "(default: <date>/<time> of each run)")
    p.add_argument("--metrics", nargs="+", help="explicit client metric columns to plot")
    p.add_argument("--all", action="store_true",
                   help="plot every client metric column (excluding *_step / *_epoch duplicates)")
    p.add_argument("--clients", nargs="+", type=int, help="restrict to these client ids")
    p.add_argument("--out-dir", help="output dir (default: fl_compare/<slugA>_vs_<slugB>)")
    p.add_argument("--ncols", type=int, default=None, help="columns in the curve matrices")
    p.add_argument("--no-clients", action="store_true", help="skip the per-client images")
    p.add_argument("--no-metrics", action="store_true", help="skip metrics.png")
    p.add_argument("--no-partition", action="store_true", help="skip partition.png")
    args = p.parse_args()

    dir_a = _resolve_run_dir(args.runA, args.outputs_root)
    dir_b = _resolve_run_dir(args.runB, args.outputs_root)
    label_a = args.labels[0] if args.labels else _short_name(dir_a)
    label_b = args.labels[1] if args.labels else _short_name(dir_b)
    out_dir = args.out_dir or os.path.join(
        "fl_compare", f"{_short_name(dir_a).replace('/', '_')}_vs_{_short_name(dir_b).replace('/', '_')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Comparing:\n  A = {dir_a}  ({label_a})\n  B = {dir_b}  ({label_b})\nOutput -> {out_dir}")

    # (1) Per-client curve matrices.
    if not args.no_clients:
        ids_a, ids_b = set(_client_ids(dir_a)), set(_client_ids(dir_b))
        common = sorted(ids_a & ids_b)
        missing = ids_a ^ ids_b
        if missing:
            print(f"[warn] clients present in only one run (skipped): {sorted(missing)}")
        if args.clients:
            common = [c for c in common if c in set(args.clients)]
        for cid in common:
            cols_a = _cols_or_none(os.path.join(dir_a, f"client_{cid}.csv"))
            cols_b = _cols_or_none(os.path.join(dir_b, f"client_{cid}.csv"))
            if args.metrics:
                metrics = args.metrics
            elif args.all:
                metrics = _discover_all_metrics(cols_a, cols_b)
            else:
                metrics = DEFAULT_CLIENT_METRICS
            plot_curve_comparison(
                _series_from_cols(cols_a, metrics), _series_from_cols(cols_b, metrics), metrics,
                os.path.join(out_dir, f"client_{cid}.png"),
                title=f"Client {cid} — {label_a} vs {label_b}",
                label_a=label_a, label_b=label_b, ncols=args.ncols)

    # (2) Global (server) curve matrix.
    if not args.no_metrics:
        plot_curve_comparison(
            _series_from_cols(_cols_or_none(os.path.join(dir_a, "metrics.csv")), GLOBAL_METRICS),
            _series_from_cols(_cols_or_none(os.path.join(dir_b, "metrics.csv")), GLOBAL_METRICS),
            GLOBAL_METRICS, os.path.join(out_dir, "metrics.png"),
            title=f"Global (server) — {label_a} vs {label_b}",
            label_a=label_a, label_b=label_b, ncols=args.ncols)

    # (3) Partition stacked/grouped bars.
    if not args.no_partition:
        plot_partition(dir_a, dir_b, label_a, label_b, os.path.join(out_dir, "partition.png"))


if __name__ == "__main__":
    main()
