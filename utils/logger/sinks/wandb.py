"""WandbSink: metrics / tables / images / watch -> Weights & Biases.

Owns the W&B run (init in `setup`, finish in `close`). Fixes the flaws of the old
path: `project`/`entity`/`group`/`mode` read from the config, `mode` validated
(`online|offline|disabled` — the old invalid `'disable'`).
"""

from __future__ import annotations

from typing import Any, Optional

from ..interface import CRITICAL, LoggerInterface

_NEVER = CRITICAL + 10  # never receives a text message
_VALID_MODES = {"online", "offline", "disabled"}


def _key(output: Optional[str], key: str) -> str:
    """Namespace a metric/artifact key by its output (W&B groups on `/`)."""
    return f"{output}/{key}" if output else key


class WandbSink(LoggerInterface):
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        mode: str = "online",
        watch_mode: str = "all",
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        config: Optional[dict] = None,
        options: Optional[dict] = None,
    ):
        super().__init__(level=_NEVER, options=options)
        self._project = project
        self._entity = entity
        self._group = group
        self._mode = mode if mode in _VALID_MODES else "online"
        self._watch_mode = watch_mode
        self._name = name
        self._save_dir = save_dir
        self._config = config
        self._wandb = None
        self._run = None

    def setup(self) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=self._project,
            entity=self._entity,
            group=self._group,
            mode=self._mode,
            name=self._name,
            dir=self._save_dir,
            config=self._config,
            reinit=True,
        )

    def log_metrics(
        self, metrics: dict, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        if self._run is not None:
            self._wandb.log(
                {_key(output, k): v for k, v in metrics.items()}, step=step
            )

    def log_hyperparams(self, params: dict) -> None:
        if self._run is not None:
            self._run.config.update(dict(params), allow_val_change=True)

    def log_table(
        self, key: str, columns: Any, data: Any, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        if self._run is not None:
            table = self._wandb.Table(columns=list(columns), data=data)
            self._wandb.log({_key(output, key): table}, step=step)

    def log_image(
        self, key: str, image: Any, step: Optional[int] = None, output: Optional[str] = None
    ) -> None:
        if self._run is not None:
            self._wandb.log({_key(output, key): self._wandb.Image(image)}, step=step)

    def watch(self, model: Any, log: str = "all") -> None:
        if self._run is not None:
            self._wandb.watch(model, log=log or self._watch_mode)

    def close(self) -> None:
        if self._run is not None:
            self._wandb.finish()
            self._run = None
