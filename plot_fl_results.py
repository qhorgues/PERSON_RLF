"""Plot a matrix of charts (pyplot) from results_fl_stn_summary.csv (offline).

Usage:
    uv run plot_fl_results.py
    uv run plot_fl_results.py --csv results_fl_stn_summary.csv --out fl_results.png --show

Produces a 2x3 grid (Recall@K, mAP/mINP, global loss, L_STN, ΔR1 vs init,
cumulative time). Le rendu est mutualisé avec le RealtimePlotSink via
`utils.logger.plot_panels` (aucune duplication de la logique de tracé).
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # headless backend: we save a PNG

from utils.logger.plot_panels import render_figure
from fl_plot_spec import KEY_MAP as _COLUMN_MAP, PANELS, shade_stage1


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results_fl_stn_summary.csv")
    parser.add_argument("--out", default="fl_results.png")
    parser.add_argument("--show", action="store_true", help="display the figure in addition to saving it")
    args = parser.parse_args()

    cols = read_csv(args.csv)
    # abscisse : `round` (ancien CSV) ou `step` (CsvSink, où step = server_round)
    x_col = "round" if "round" in cols else "step"
    rounds = to_float(cols[x_col])

    history = {}
    for column, key in _COLUMN_MAP.items():
        if column in cols:
            history[key] = (rounds, to_float(cols[column]))

    render_figure(
        history,
        panels=PANELS,
        shade=shade_stage1,
        out_path=args.out,
        title="Federated TBPS-SigLIP — FedProx (µ=0.01) · STN · 8 clients · VN3K_VI",
    )
    print(f"Figure saved -> {args.out}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
