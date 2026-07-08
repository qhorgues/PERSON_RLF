"""RealtimePlotSink: metrics -> matplotlib figure updated live (generic).

Accumulates the metrics history and re-renders the panel grid (via `plot_panels`)
every `every_n` calls. Headless-friendly: the figure is saved as PNG on each
update (= "real-time" via an evolving file).

Everything project-specific is supplied through `options` (from config), so the
sink itself carries no domain knowledge:
  - `spec`     : dotted path to a module exposing `PANELS` / `KEY_MAP` /
                 `shade_stage1` (e.g. "fl_plot_spec"); each is optional.
  - `panels`   : inline panel spec (overrides `spec.PANELS`).
  - `key_map`  : inline {logged_name: canonical_key} (overrides `spec.KEY_MAP`);
                 when absent, metric names are used as canonical keys (identity).
  - pyplot     : `backend`, `style`, `rcParams` (applied in `setup`),
                 `dpi`, `figsize`, `ncols`, `xlabel`, `colormap`, `title`
                 (passed to `render_figure`).
  - shading    : `shade_until` (int) or `shade_span` [a, b] — a generic vertical
                 span; ignored if `spec.shade_stage1` is provided.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..interface import CRITICAL, LoggerInterface

_NEVER = CRITICAL + 10


class RealtimePlotSink(LoggerInterface):
    def __init__(self, out_path: str, every_n: int = 1, title: Optional[str] = None, options: Optional[dict] = None):
        super().__init__(level=_NEVER, options=options)
        self._out_path = out_path
        self._every_n = max(1, int(every_n))
        self._title = self._opt("title", title)
        self._hist: Dict[str, Tuple[List, List]] = {}
        self._calls = 0
        self._fig = None
        self._axes = None

        spec = self._load_spec(self._opt("spec"))
        self._key_map = self._opt("key_map") or getattr(spec, "KEY_MAP", None)
        self._panels = self._opt("panels") or getattr(spec, "PANELS", None)
        self._shade = self._resolve_shade(spec)

    # --- option resolution ---------------------------------------------
    @staticmethod
    def _load_spec(dotted: Optional[str]):
        if not dotted:
            return None
        try:
            from importlib import import_module

            return import_module(dotted)
        except Exception:
            return None

    def _resolve_shade(self, spec):
        shade = getattr(spec, "shade_stage1", None)
        if shade is not None:
            return shade
        until = self._opt("shade_until")
        span = self._opt("shade_span")
        if until is not None:
            u = float(until)
            return lambda ax: (
                ax.axvspan(0, u, color="0.85", alpha=0.6, lw=0, zorder=0),
                ax.axvline(u, color="0.4", ls="--", lw=1, zorder=1),
            )
        if span is not None and len(span) == 2:
            a, b = float(span[0]), float(span[1])
            return lambda ax: ax.axvspan(a, b, color="0.85", alpha=0.6, lw=0, zorder=0)
        return None

    # --- lifecycle ------------------------------------------------------
    def setup(self) -> None:
        import matplotlib

        backend = self._opt("backend")
        if backend:
            try:
                matplotlib.use(backend)
            except Exception:
                pass
        elif not matplotlib.get_backend() or matplotlib.get_backend().lower() == "agg":
            matplotlib.use("Agg")  # headless fallback if no display

        import matplotlib.pyplot as plt

        style = self._opt("style")
        if style:
            try:
                plt.style.use(style)
            except Exception:
                pass
        rc = self._opt("rcParams")
        if rc:
            try:
                plt.rcParams.update(dict(rc))
            except Exception:
                pass

    def log_metrics(
        self, metrics: dict, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        # Only the global (untagged) stream is plotted; per-output (e.g. per-client)
        # metrics stay in their CSVs to avoid a figure-per-output explosion.
        if output is not None:
            return
        x = step if step is not None else self._calls
        for name, value in metrics.items():
            key = self._key_map.get(name) if self._key_map else name
            if key is None:
                continue
            xs, ys = self._hist.setdefault(key, ([], []))
            try:
                xs.append(x)
                ys.append(float(value))
            except (TypeError, ValueError):
                xs.pop()
        self._calls += 1
        if self._calls % self._every_n == 0:
            self._render()

    def _render(self) -> None:
        if not self._hist:
            return
        from ..plot_panels import render_figure

        self._fig, self._axes = render_figure(
            self._hist,
            panels=self._panels,
            out_path=self._out_path,
            title=self._title,
            xlabel=self._opt("xlabel", "round"),
            ncols=self._opt("ncols"),
            figsize=self._opt("figsize"),
            dpi=self._opt("dpi", 150),
            shade=self._shade,
            colormap=self._opt("colormap"),
            fig=self._fig,
            axes=self._axes,
        )

    def close(self) -> None:
        self._render()
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
