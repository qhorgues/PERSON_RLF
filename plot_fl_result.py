"""Plot federated-learning runs (offline) — one image per client + global + partition.

Accepts **any number of run directories** (1, 2, 3, …): a single directory plots that
run on its own, several directories overlay them for comparison.

Usage:
    uv run plot_fl_result.py <run>                       # single run
    uv run plot_fl_result.py <runA> <runB> [<runC> ...]  # compare N runs
    uv run plot_fl_result.py 2026-07-07/08-26-24 2026-07-06/11-04-32
    uv run plot_fl_result.py <runA> <runB> --metrics val_t2i_R1 total_loss
    uv run plot_fl_result.py <runA> <runB> --all --no-partition

Chaque run est un dossier Hydra `outputs/<date>/<heure>/` contenant `client_*.csv`,
`metrics.csv` et `partition.csv`. Le script produit, dans le dossier de sortie :

  - client_<cid>.png : matrice de courbes des métriques du client (une courbe par run) ;
  - metrics.png      : matrice de courbes des métriques globales serveur (une par run) ;
  - partition.png    : barres horizontales groupées (répartition du dataset par client).

Le rendu des matrices de courbes est mutualisé avec le RealtimePlotSink via
`utils.logger.plot_panels` (aucune duplication de la logique de tracé). Les couleurs par
run suivent la palette catégorielle validée (color-by-series), doublée d'un cycle de
marqueurs comme second canal d'identité.
"""

import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless backend: we save PNGs

from utils.logger.plot_panels import render_figure

# Per-run styling: validated categorical palette (light surface, CVD-safe order) +
# a marker cycle so each run is distinguishable by hue *and* shape. Indexed by run
# position, wrapping around for more runs than slots.
RUN_COLORS = [
    "#2a78d6", "#1baf7a", "#eda100", "#008300",
    "#4a3aa7", "#e34948", "#e87ba4", "#eb6834",
]
RUN_MARKERS = ["o-", "s-", "^-", "D-", "v-", "P-", "X-", "*-"]


def _run_style(i):
    """(marker/line format, color) for run `i` (cycles past the palette length)."""
    return RUN_MARKERS[i % len(RUN_MARKERS)], RUN_COLORS[i % len(RUN_COLORS)]


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


# --------------------------------------------------------------------------- CSV I/O

def read_csv(path):
    """Read the CSV, skipping comment lines (#); return a dict of columns."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    header, data = rows[0], rows[1:]
    cols = {name: [] for name in header}
    for row in data:
        for name, value in zip(header, row):
            cols[name].append(value)
    return cols


def to_float(values):
    """Convert to float, None if empty (e.g. stn_loss at round 0)."""
    out = []
    for v in values:
        v = v.strip()
        out.append(float(v) if v else None)
    return out


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

def plot_curve_comparison(series_list, labels, metrics, out_path, title, ncols):
    """One panel per metric, one curve per run, via the shared `render_figure`.

    `series_list[i]` = {metric: (xs, ys)} for run `i`; `labels[i]` its legend label.
    """
    import matplotlib.pyplot as plt

    history, panels = {}, []
    for m in metrics:
        curves = []
        for i, series in enumerate(series_list):
            if m not in series:
                continue
            key = f"{m}::{i}"
            history[key] = series[m]
            fmt, color = _run_style(i)
            curves.append((key, fmt, color, labels[i]))
        if not curves:
            continue
        panels.append((pretty(m), pretty(m), curves))

    if not panels:
        print(f"  [skip] {os.path.basename(out_path)} : no metric with data in any run")
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
    """One horizontal bar per client (y = clients); grouped by run when there are several."""
    ids = sorted({cid for _, clients, _ in runs for cid in clients})
    index = {cid: i for i, cid in enumerate(ids)}
    nb = len(runs)
    height = 0.8 / max(nb, 1)
    start = -(nb - 1) / 2.0  # center the run-bars around each client's y position
    for k, (label, clients, data) in enumerate(runs):
        _, color = _run_style(k)
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


def plot_partition(dirs, labels, out_path):
    """partition.png : one horizontal bar per client (grouped per run), one panel per metric."""
    import matplotlib.pyplot as plt

    runs = []  # (label, clients, data) for each run that has a partition.csv
    for d, label in zip(dirs, labels):
        parsed = _parse_partition(_cols_or_none(os.path.join(d, "partition.csv")))
        if parsed is not None:
            runs.append((label, parsed[0], parsed[1]))
    if not runs:
        print("  [skip] partition.png : no partition.csv in any run")
        return

    present = {m for _, _, data in runs for m in data}
    metrics = [m for m in PARTITION_METRICS if m in present]
    if not metrics:
        print("  [skip] partition.png : no known partition columns")
        return

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        _client_bar(ax, m, runs)

    fig.suptitle(f"Partition (per client) — {' vs '.join(labels)}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


# --------------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+",
                   help="one or more run dirs (e.g. 2026-07-07/08-26-24, resolved under outputs/)")
    p.add_argument("--outputs-root", default="outputs", help="base dir for relative run args")
    p.add_argument("--labels", "--titles", dest="labels", nargs="+", metavar="LABEL",
                   help="title/label for each run, shown in the suptitle + legends "
                        "(must match the number of runs; default: <date>/<time> of each run)")
    p.add_argument("--metrics", nargs="+", help="explicit client metric columns to plot")
    p.add_argument("--all", action="store_true",
                   help="plot every client metric column (excluding *_step / *_epoch duplicates)")
    p.add_argument("--clients", nargs="+", type=int, help="restrict to these client ids")
    p.add_argument("--out-dir", help="output dir (default: fl_result/<slug1>[_vs_<slug2>...])")
    p.add_argument("--ncols", type=int, default=None, help="columns in the curve matrices")
    p.add_argument("--no-clients", action="store_true", help="skip the per-client images")
    p.add_argument("--no-metrics", action="store_true", help="skip metrics.png")
    p.add_argument("--no-partition", action="store_true", help="skip partition.png")
    args = p.parse_args()

    dirs = [_resolve_run_dir(r, args.outputs_root) for r in args.runs]
    if args.labels:
        if len(args.labels) != len(dirs):
            raise SystemExit(
                f"--labels expects {len(dirs)} label(s) to match the runs, got {len(args.labels)}")
        labels = args.labels
    else:
        labels = [_short_name(d) for d in dirs]
    joined = " vs ".join(labels)

    slugs = [_short_name(d).replace("/", "_") for d in dirs]
    out_dir = args.out_dir or os.path.join("fl_result", "_vs_".join(slugs))
    os.makedirs(out_dir, exist_ok=True)
    print("Plotting:")
    for d, label in zip(dirs, labels):
        print(f"  {label}  <- {d}")
    print(f"Output -> {out_dir}")

    # (1) Per-client curve matrices.
    if not args.no_clients:
        id_sets = [set(_client_ids(d)) for d in dirs]
        common = sorted(set.intersection(*id_sets))
        missing = sorted(set().union(*id_sets) - set(common))
        if missing:
            print(f"[warn] clients not present in every run (skipped): {missing}")
        if args.clients:
            common = [c for c in common if c in set(args.clients)]
        for cid in common:
            cols_list = [_cols_or_none(os.path.join(d, f"client_{cid}.csv")) for d in dirs]
            if args.metrics:
                metrics = args.metrics
            elif args.all:
                metrics = _discover_all_metrics(*cols_list)
            else:
                metrics = DEFAULT_CLIENT_METRICS
            series_list = [_series_from_cols(cols, metrics) for cols in cols_list]
            plot_curve_comparison(
                series_list, labels, metrics,
                os.path.join(out_dir, f"client_{cid}.png"),
                title=f"Client {cid} — {joined}", ncols=args.ncols)

    # (2) Global (server) curve matrix.
    if not args.no_metrics:
        series_list = [_series_from_cols(_cols_or_none(os.path.join(d, "metrics.csv")), GLOBAL_METRICS)
                       for d in dirs]
        plot_curve_comparison(
            series_list, labels, GLOBAL_METRICS, os.path.join(out_dir, "metrics.png"),
            title=f"Global (server) — {joined}", ncols=args.ncols)

    # (3) Partition grouped bars.
    if not args.no_partition:
        plot_partition(dirs, labels, os.path.join(out_dir, "partition.png"))


if __name__ == "__main__":
    main()
