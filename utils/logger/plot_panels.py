"""Generic, project-agnostic rendering of a grid of metric panels.

Used by:
  - `utils/logger/sinks/plot.py` (RealtimePlotSink, real-time)
  - any offline script that supplies its own panel spec (e.g. `plot_fl_results.py`,
    which imports the FL spec from `fl_plot_spec.py`)

`history` = dict {canonical_key: (xs, ys)}. A *panel spec* describes the grid;
when `panels is None`, one panel is auto-created per key in `history`, so the
renderer works out-of-the-box for arbitrary metrics with zero config.

Panel spec — both formats accepted:
  tuple: (title, ylabel, [series, ...])
  dict:  {"title": ..., "ylabel": ..., "series": [series, ...]}
Series — both formats accepted:
  tuple: (key, style, color, label)   # color / label may be None
  dict:  {"key": ..., "style": "o-", "color": None, "label": None}
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _clean(xs, ys) -> Tuple[list, list]:
    """Keep the (x, y) pairs whose y is not None."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ([], [])


def _norm_series(entry: Any) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Normalize a series entry (tuple or dict) to (key, style, color, label)."""
    if isinstance(entry, dict):
        return (entry["key"], entry.get("style") or "o-", entry.get("color"), entry.get("label"))
    entry = list(entry) + [None] * (4 - len(entry))
    key, style, color, label = entry[:4]
    return (key, style or "o-", color, label)


def _norm_panel(panel: Any) -> Tuple[str, str, List[tuple]]:
    """Normalize a panel (tuple or dict) to (title, ylabel, [series...])."""
    if isinstance(panel, dict):
        title, ylabel, series = panel.get("title", ""), panel.get("ylabel", ""), panel.get("series", [])
    else:
        title, ylabel, series = panel
    return (title, ylabel, [_norm_series(s) for s in series])


def _auto_panels(history: Dict[str, Tuple[list, list]]) -> List[tuple]:
    """One panel per history key (used when no spec is given)."""
    return [(key, key, [(key, "o-", None, None)]) for key in history]


def _flat_axes(axes) -> list:
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, (list, tuple)):
        return list(axes)
    return [axes]


def render_figure(
    history: Dict[str, Tuple[list, list]],
    panels: Optional[Sequence[Any]] = None,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "round",
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    shade: Optional[Callable[[Any], None]] = None,
    colormap: Optional[str] = None,
    fig=None,
    axes=None,
):
    """Plot a grid of panels. Reuses (fig, axes) if provided (real-time mode).

    `shade`  : optional callable applied to each axis (e.g. highlight a region).
    `colormap`: name of a matplotlib colormap used to color series with no explicit
                color; when None, matplotlib's default color cycle is used.
    """
    import matplotlib.pyplot as plt

    spec = [_norm_panel(p) for p in (panels if panels is not None else _auto_panels(history))]
    n = len(spec)
    if n == 0:
        return fig, axes

    cols = int(ncols) if ncols else min(3, n)
    rows = math.ceil(n / cols)
    if figsize is None:
        figsize = (5.33 * cols, 4.5 * rows)  # ~ (16, 9) for a 3x2 grid
    cmap = plt.get_cmap(colormap) if colormap else None

    # Recreate the grid if none was passed or if the panel count changed.
    created = fig is None or axes is None or len(_flat_axes(axes)) != rows * cols
    if created:
        fig, axes = plt.subplots(rows, cols, figsize=tuple(figsize))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    for idx, ax in enumerate(_flat_axes(axes)):
        ax.clear()
        if idx >= n:
            ax.axis("off")  # unused cell in a non-full grid
            continue
        panel_title, ylabel, series = spec[idx]
        if shade is not None:
            try:
                shade(ax)
            except Exception:
                pass
        plotted = False
        for si, (key, style, color, label) in enumerate(series):
            if key not in history:
                continue
            xs, ys = _clean(*history[key])
            if not xs:
                continue
            kwargs = {"ms": 3}
            if color:
                kwargs["color"] = color
            elif cmap is not None:
                kwargs["color"] = cmap(si / (len(series) - 1) if len(series) > 1 else 0.0)
            if label:
                kwargs["label"] = label
            ax.plot(xs, ys, style, **kwargs)
            plotted = True
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if plotted and any(lbl for _, _, _, lbl in series):
            ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else (0, 0, 1, 1))
    if out_path:
        fig.savefig(out_path, dpi=dpi)
    return fig, axes
