"""FL/STN plot spec for TBPS-SigLIP — project-specific (NOT part of the logger core).

Single source of truth for the federated-learning result panels, consumed by
`utils/logger/sinks/plot.py` via `sinks.plot.options.spec: fl_plot_spec` (real-time).

Kept out of `utils/logger/` so the logging package stays generic and droppable
into any project. `utils/logger/plot_panels.render_figure` consumes `PANELS` and
`shade_stage1` from here; `RealtimePlotSink` also uses `KEY_MAP` to map logged
metric names to the canonical plot keys used by `PANELS`.
"""

from __future__ import annotations

STAGE1_END = 5  # rounds [0, 5) = Stage 1 (see _apply_two_stage_policy)

# logged metric name -> canonical plot key
KEY_MAP = {
    "global_R1": "r1", "global_R5": "r5", "global_R10": "r10",
    "global_mAP": "mAP", "global_mINP": "mINP",
    "global_loss": "loss", "stn_loss": "stn",
    "delta_R1_vs_init": "dR1", "elapsed_min": "elapsed",
    # tolerance for non-prefixed names
    "R1": "r1", "R5": "r5", "R10": "r10", "mAP": "mAP", "mINP": "mINP",
}

# (title, ylabel, [(key, style, color|None, label|None)])
PANELS = [
    ("Global Recall@K (t2i)", "Recall (%)", [
        ("r1", "o-", None, "R@1"), ("r5", "s-", None, "R@5"), ("r10", "^-", None, "R@10"),
    ]),
    ("Global mAP / mINP", "score (%)", [
        ("mAP", "o-", "tab:green", "mAP"), ("mINP", "s-", "tab:olive", "mINP"),
    ]),
    ("Global loss (centralized eval)", "loss", [
        ("loss", "o-", "tab:red", None),
    ]),
    ("STN loss (partial<->holistic)", "L_STN", [
        ("stn", "o-", "tab:purple", None),
    ]),
    ("R@1 gain vs init", "dR@1 (pts)", [
        ("dR1", "o-", "tab:blue", None),
    ]),
    ("Cumulative time (single-GPU)", "minutes", [
        ("elapsed", "o-", "tab:gray", None),
    ]),
]


def shade_stage1(ax, stage1_end: int = STAGE1_END) -> None:
    """Highlight the FL Stage-1 rounds on an axis."""
    ax.axvspan(0, stage1_end, color="0.85", alpha=0.6, lw=0, zorder=0)
    ax.axvline(stage1_end, color="0.4", ls="--", lw=1, zorder=1)
