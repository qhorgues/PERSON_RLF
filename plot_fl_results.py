"""Plot a matrix of charts (pyplot) from results_fl_stn_summary.csv.

Usage:
    uv run plot_fl_results.py
    uv run plot_fl_results.py --csv results_fl_stn_summary.csv --out fl_results.png --show

Produces a 2x3 grid:
  R@K   |  mAP/mINP    |  global loss
  L_STN |  ΔR1 vs init |  cumulative time (single-GPU)
The Stage 1 region (STN trainable / ReID frozen, rounds 0-5) is shaded because
retrieval metrics stay flat there (the aggregated backbone does not move).
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")  # headless backend: we save a PNG
import matplotlib.pyplot as plt

STAGE1_END = 5  # rounds [0, 5) = Stage 1 (see _apply_two_stage_policy)


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


def filter_pairs(xs, ys):
    """Keep (x, y) pairs whose y is not None."""
    return zip(*[(x, y) for x, y in zip(xs, ys) if y is not None]) if any(
        y is not None for y in ys
    ) else ([], [])


def shade_stage1(ax):
    ax.axvspan(0, STAGE1_END, color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.axvline(STAGE1_END, color="0.4", ls="--", lw=1, zorder=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results_fl_stn_summary.csv")
    parser.add_argument("--out", default="fl_results.png")
    parser.add_argument("--show", action="store_true", help="display the figure in addition to saving it")
    args = parser.parse_args()

    cols = read_csv(args.csv)
    rounds = to_float(cols["round"])
    r1 = to_float(cols["global_R1"])
    r5 = to_float(cols["global_R5"])
    r10 = to_float(cols["global_R10"])
    mAP = to_float(cols["global_mAP"])
    mINP = to_float(cols["global_mINP"])
    loss = to_float(cols["global_loss"])
    stn = to_float(cols["stn_loss"])
    dR1 = to_float(cols["delta_R1_vs_init"])
    elapsed = to_float(cols["elapsed_min"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Federated TBPS-SigLIP — FedProx (µ=0.01) · STN · 8 clients · VN3K_VI",
        fontsize=13, fontweight="bold",
    )

    # (0,0) Recall@K
    ax = axes[0, 0]
    shade_stage1(ax)
    ax.plot(rounds, r1, "o-", ms=3, label="R@1")
    ax.plot(rounds, r5, "s-", ms=3, label="R@5")
    ax.plot(rounds, r10, "^-", ms=3, label="R@10")
    ax.set_title("Global Recall@K (t2i)")
    ax.set_xlabel("round"); ax.set_ylabel("Recall (%)")
    ax.legend(); ax.grid(alpha=0.3)

    # (0,1) mAP / mINP
    ax = axes[0, 1]
    shade_stage1(ax)
    ax.plot(rounds, mAP, "o-", ms=3, color="tab:green", label="mAP")
    ax.plot(rounds, mINP, "s-", ms=3, color="tab:olive", label="mINP")
    ax.set_title("Global mAP / mINP")
    ax.set_xlabel("round"); ax.set_ylabel("score (%)")
    ax.legend(); ax.grid(alpha=0.3)

    # (0,2) global loss
    ax = axes[0, 2]
    shade_stage1(ax)
    ax.plot(rounds, loss, "o-", ms=3, color="tab:red")
    ax.set_title("Global loss (centralized eval)")
    ax.set_xlabel("round"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3)

    # (1,0) L_STN
    ax = axes[1, 0]
    shade_stage1(ax)
    xs, ys = filter_pairs(rounds, stn)
    ax.plot(list(xs), list(ys), "o-", ms=3, color="tab:purple")
    ax.set_title("STN loss (partial↔holistic alignment)")
    ax.set_xlabel("round"); ax.set_ylabel("L_STN")
    ax.grid(alpha=0.3)

    # (1,1) ΔR1 vs init
    ax = axes[1, 1]
    shade_stage1(ax)
    ax.plot(rounds, dR1, "o-", ms=3, color="tab:blue")
    ax.set_title("R@1 gain vs init")
    ax.set_xlabel("round"); ax.set_ylabel("ΔR@1 (pts)")
    ax.grid(alpha=0.3)

    # (1,2) cumulative time
    ax = axes[1, 2]
    shade_stage1(ax)
    ax.plot(rounds, elapsed, "o-", ms=3, color="tab:gray")
    ax.set_title("Cumulative time (single-GPU simulation)")
    ax.set_xlabel("round"); ax.set_ylabel("minutes")
    ax.grid(alpha=0.3)

    # shared annotation for the shaded region
    axes[0, 0].text(
        0.1, 0.95, "Stage 1\n(ReID frozen)", transform=axes[0, 0].transAxes,
        fontsize=8, va="top", color="0.3",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=150)
    print(f"Figure saved -> {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
