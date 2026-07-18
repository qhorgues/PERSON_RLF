"""Plot a partition distribution matrix as a horizontal *stacked* ("cumulé") bar chart.

The federated partitioner logs a ``label x client`` matrix — sample counts per physical
site (or camera) per FL client — via `IdentityPartitioner.distribution_matrix`. Runs now
also persist it next to the metrics as ``distribution_site.csv`` /
``distribution_camera.csv`` (see `federated.partition.write_distribution_csv`). This script
turns such a CSV (or the PrettyTable block still sitting in an old ``train.log``) into a
horizontal stacked bar chart, so the non-IID heterogeneity of the partition is visible at a
glance: how each client's samples split across sites.

Usage:
    # From a persisted CSV (produced by an FL run):
    uv run plot_distribution.py outputs/<date>/<time>/distribution_site.csv

    # From an old run that only has the table in its log (writes the CSV too):
    uv run plot_distribution.py --from-log outputs/2026-07-14/19-04-43/train.log --matrix site

    # Transpose (one bar per site, stacked by client) and/or normalize to 100 %:
    uv run plot_distribution.py distribution_site.csv --by site --normalize

CSV schema (same as `write_distribution_csv`): first column = the label name
(``location`` / ``camera``), then one ``client_<i>`` column per client, then a ``TOTAL``
column; the last row is a ``TOTAL`` row. Both TOTAL row and column are dropped before
plotting.

Rendering conventions (Agg backend, validated categorical palette, direct labels) are
shared with `plot_fl_result.py` — `RUN_COLORS` and `read_csv` are reused from there.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")  # headless backend: we save PNGs

from plot_fl_result import RUN_COLORS, read_csv

OTHER_GRAY = "#9aa0a6"  # neutral gray de-emphasizes the 'other' filler bucket


# --------------------------------------------------------------------------- matrix model
# A distribution matrix is (label_name, labels, clients, counts) where
#   counts[label][client] = sample count (float), TOTAL row/column excluded.


def _matrix_from_rows(header, data_rows):
    """Build (label_name, labels, clients, counts) from a header + data rows (TOTAL dropped)."""
    label_name = header[0]
    clients = [h for h in header[1:] if h != "TOTAL"]
    col_index = {name: i for i, name in enumerate(header)}
    labels, counts = [], {}
    for row in data_rows:
        label = row[0]
        if label == "TOTAL":
            continue
        labels.append(label)
        counts[label] = {c: float(row[col_index[c]]) for c in clients}
    return label_name, labels, clients, counts


def read_matrix_csv(path):
    """Read a distribution CSV (via the shared comment-skipping `read_csv`) into the model."""
    cols = read_csv(path)
    header = list(cols.keys())
    n = len(next(iter(cols.values())))
    data_rows = [[cols[h][i] for h in header] for i in range(n)]
    return _matrix_from_rows(header, data_rows)


def parse_matrix_from_log(log_path, matrix):
    """Extract a distribution matrix from the PrettyTable block in a `train.log`.

    `matrix` is ``site`` or ``camera`` and selects which of the two logged tables
    (``samples per <matrix> per client``) to read.
    """
    marker = f"samples per {matrix} per client"
    with open(log_path) as f:
        lines = f.read().splitlines()

    start = next((i + 1 for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        raise SystemExit(f"No '{marker}' table found in {log_path!r}")

    rows = []
    for line in lines[start:]:
        s = line.strip()
        if s.startswith("+"):  # PrettyTable border
            continue
        if s.startswith("|"):  # PrettyTable data/header row
            rows.append([c.strip() for c in s.strip("|").split("|")])
            continue
        if rows:  # first non-table line after the block => table is done
            break
    if len(rows) < 2:
        raise SystemExit(f"Could not parse a '{marker}' table from {log_path!r}")
    return _matrix_from_rows(rows[0], rows[1:])


def write_matrix_csv(path, label_name, labels, clients, counts):
    """Persist a matrix to CSV in the `write_distribution_csv` schema (TOTAL row/col added)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label_name] + clients + ["TOTAL"])
        totals = {c: 0.0 for c in clients}
        for label in labels:
            row_vals = [counts[label][c] for c in clients]
            for c, v in zip(clients, row_vals):
                totals[c] += v
            writer.writerow([label] + [_fmt(v) for v in row_vals] + [_fmt(sum(row_vals))])
        writer.writerow(
            ["TOTAL"] + [_fmt(totals[c]) for c in clients] + [_fmt(sum(totals.values()))]
        )
    print(f"  wrote CSV -> {path}")


def _fmt(v):
    """Render a count as an int when it is whole (counts always are), else as float."""
    return int(v) if float(v).is_integer() else v


# --------------------------------------------------------------------------- rendering

def _client_label(col):
    """`client_3` -> `client 3` for the axis tick."""
    return col.replace("_", " ")


def _segment_color(seg, i, seg_is_client):
    """Categorical color for stacked segment `seg` (i-th); 'other' site => neutral gray."""
    if not seg_is_client and str(seg).lower() == "other":
        return OTHER_GRAY
    return RUN_COLORS[i % len(RUN_COLORS)]


def plot_stacked(label_name, labels, clients, counts, by, normalize, title, out_path):
    """Render the horizontal stacked bar chart.

    `by="client"`: one bar per client, stacked by site (default — shows each client's site
    composition / heterogeneity). `by="site"`: transpose (one bar per site, stacked by
    client). `normalize`: 100 %-stacked shares instead of raw counts.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if by == "client":
        bars, segments = clients, labels          # bar=client, stacked by site
        bar_ticks = [_client_label(c) for c in clients]
        y_axis, seg_kind = "client", label_name
        val = lambda bar, seg: counts[seg][bar]
        seg_is_client = False
    else:  # by == "site"
        bars, segments = labels, clients          # bar=site, stacked by client
        bar_ticks = labels
        y_axis, seg_kind = label_name, "client"
        val = lambda bar, seg: counts[bar][seg]
        seg_is_client = True

    y = np.arange(len(bars))
    totals = np.array([sum(val(b, s) for s in segments) for b in bars], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, max(3.0, 0.45 * len(bars) + 1.6)))
    left = np.zeros(len(bars))
    for i, seg in enumerate(segments):
        widths = np.array([val(b, seg) for b in bars], dtype=float)
        if normalize:
            widths = np.divide(widths, totals, out=np.zeros_like(widths), where=totals > 0) * 100.0
        seg_label = _client_label(seg) if seg_is_client else str(seg)
        ax.barh(y, widths, left=left, height=0.8, color=_segment_color(seg, i, seg_is_client),
                label=seg_label, edgecolor="white", linewidth=0.7)
        left = left + widths

    ax.set_yticks(y)
    ax.set_yticklabels(bar_ticks)
    ax.invert_yaxis()  # first bar (client 0 / first site) on top
    ax.set_ylabel(y_axis)
    ax.set_xlabel("share (%)" if normalize else "samples")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)

    if normalize:
        ax.set_xlim(0, 100)
    else:
        # one direct label per bar (its total) — never a number on every segment
        for yi, tot in zip(y, totals):
            ax.text(tot, yi, f" {int(tot)}", va="center", ha="left", fontsize=8, color="#444")
        ax.margins(x=0.08)

    ax.legend(title=seg_kind, bbox_to_anchor=(1.01, 1.0), loc="upper left",
              frameon=False, fontsize=8, title_fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# --------------------------------------------------------------------------- CLI

def _default_out(base_path, by, normalize):
    stem = os.path.splitext(base_path)[0]
    return f"{stem}_by_{by}" + ("_norm" if normalize else "") + ".png"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="distribution CSV to plot (omit when using --from-log)")
    p.add_argument("--from-log", metavar="TRAIN_LOG",
                   help="parse the distribution table out of a train.log and write a CSV first")
    p.add_argument("--matrix", choices=["site", "camera"], default="site",
                   help="which logged table to read with --from-log (default: site)")
    p.add_argument("--csv-out", help="CSV path to write with --from-log "
                                     "(default: distribution_<matrix>.csv beside the log)")
    p.add_argument("--by", choices=["client", "site"], default="client",
                   help="one bar per client (stacked by site, default) or per site (stacked by client)")
    p.add_argument("--normalize", action="store_true", help="100%%-stacked shares instead of raw counts")
    p.add_argument("--title", help="chart title (default: auto)")
    p.add_argument("--out", help="output PNG (default: <input>_by_<by>[_norm].png)")
    args = p.parse_args()

    if bool(args.input) == bool(args.from_log):
        raise SystemExit("Provide exactly one source: a CSV positional OR --from-log.")

    if args.from_log:
        label_name, labels, clients, counts = parse_matrix_from_log(args.from_log, args.matrix)
        csv_out = args.csv_out or os.path.join(
            os.path.dirname(args.from_log) or ".", f"distribution_{args.matrix}.csv")
        write_matrix_csv(csv_out, label_name, labels, clients, counts)
        source = csv_out
    else:
        label_name, labels, clients, counts = read_matrix_csv(args.input)
        source = args.input

    out_path = args.out or _default_out(source, args.by, args.normalize)
    # `--by client` stacks by the label dimension (location/camera); `--by site` stacks by client.
    stacked_by, bar_dim = (label_name, "client") if args.by == "client" else ("client", label_name)
    title = args.title or (
        f"Sample distribution — {stacked_by} per {bar_dim}"
        + (" (normalized)" if args.normalize else "")
    )
    plot_stacked(label_name, labels, clients, counts, args.by, args.normalize, title, out_path)


if __name__ == "__main__":
    main()
