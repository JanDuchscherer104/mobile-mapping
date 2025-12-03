"""Typed command-line interface for exercising Mobile Mapping modules.

The CLI is intentionally thin: it wires `pydantic-settings`' CLI helpers to the
existing Pydantic config objects so teammates can inspect or persist configs and
spin up placeholder runtime components without writing imperative argparse code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from mob_map.data import DataModuleConfig
from mob_map.utils.console import Console

ModeLiteral = Literal["show-data", "show-sample", "plot-frame", "plot-undistort"]
ModeHandler = Callable[["CLISettings"], None]

__all__ = ["CLISettings", "main"]


class CLISettings(BaseSettings):
    """Declarative CLI options parsed via `pydantic-settings`."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="MOB_MAP_",
        extra="forbid",
        cli_shortcuts={
            "data.dataset": ["dataset"],
            "data.sequence_loader.dataset_root": ["dataset-root"],
            "data.sequence_loader.sequence": ["sequence"],
            "data.stereo_iterator.require_disparity": ["require-disparity"],
        },
    )

    mode: ModeLiteral = Field(
        default="show-data",
        description="Selects which subsystem to exercise (e.g. show-data).",
        validation_alias=AliasChoices("mode", "m"),
    )
    sample_index: int = Field(
        default=0,
        ge=0,
        description="Frame index inspected by the show-sample mode.",
        validation_alias=AliasChoices("sample-index", "i"),
    )
    quiet: bool = Field(
        default=False,
        description="Silence informational console logs (still prints requested artefacts).",
        validation_alias=AliasChoices("quiet", "q"),
    )
    debug: bool = Field(
        default=False,
        description="Enable verbose debug logging for troubleshooting.",
        validation_alias=AliasChoices("debug", "d"),
    )

    def cli_cmd(self) -> None:  # pragma: no cover - exercised via CLI entrypoint
        """Dispatch execution for the chosen mode."""

        run_cli(self)


def run_cli(settings: CLISettings) -> None:
    """Execute the requested CLI mode using the parsed settings."""

    handler = MODE_HANDLERS.get(settings.mode)
    if handler is None:  # pragma: no cover - guarded by Literal typing
        msg = f"Unsupported mode '{settings.mode}'"
        raise ValueError(msg)
    handler(settings)


def _run_show_data(settings: CLISettings) -> None:
    console = _build_console(settings, "data")
    console.log("Initialising DataModule placeholder ...")
    data_module = DataModuleConfig().setup_target()
    data_module.summary()


def _run_show_sample(settings: CLISettings) -> None:
    console = _build_console(settings, "sample")
    dataset = DataModuleConfig().setup_target()
    sample = _fetch_sample(dataset, settings.sample_index, console)
    if sample is None:
        return
    images = sample.images
    shapes = {
        "left": images.left.shape if images.left is not None else None,
        "right": images.right.shape if images.right is not None else None,
        "disparity": images.disparity.shape if images.disparity is not None else None,
    }
    console.log(
        "Frame {idx}: left={left} right={right} disparity={disp}".format(
            idx=sample.index,
            left=shapes.get("left"),
            right=shapes.get("right"),
            disp=shapes.get("disparity"),
        )
    )
    console.log(f"Paths: {sample.paths.left.name} ↔ {sample.paths.right.name}")
    if sample.paths.disparity is not None:
        console.log(f"Disparity: {sample.paths.disparity.name}")
    if sample.pose is not None:
        console.log(f"Ground-truth source: {sample.pose.source}")
        console.plog(sample.pose.matrix.tolist())
    else:
        console.warn("No ground-truth pose recorded for this index.")


def _run_plot_frame(settings: CLISettings, undistort: bool = False) -> None:
    console = _build_console(settings, "plot")
    dataset = DataModuleConfig().setup_target()
    sample = _fetch_sample(dataset, settings.sample_index, console)
    if sample is None:
        return
    from matplotlib import pyplot as plt

    color_images = dataset.loader.load_frame(sample.index, color=True, load_disparity=True)
    panels = _prepare_plot_panels(color_images)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, panel in zip(axes, panels, strict=False):
        if undistort:
            title, image, cmap = panel
            if image is not None:
                from mob_map.image_processing.undistort import undistort_image

                image = undistort_image(image, sample.intrinsics)
            _render_panel(ax, title, image, cmap)
        else:
            _render_panel(ax, *panel)
    fig.suptitle(
        f"Frame {sample.index} - {sample.paths.left.name} (dataset {dataset.loader.config.sequence})"
        + (" (undistorted)" if undistort else "")
    )
    plt.tight_layout()
    plt.show()


def _build_console(settings: CLISettings, *parts: str) -> Console:
    return Console.with_prefix("mob-map", "cli", *parts).set_verbose(not settings.quiet).set_debug(settings.debug)


def _fetch_sample(dataset, index: int, console: Console):
    if index >= len(dataset):
        console.error(f"Requested sample {index} exceeds dataset length ({len(dataset)} frames).")
        return None
    return dataset[index]


def _prepare_plot_panels(images: dict[str, object]) -> list[tuple[str, object | None, str | None]]:
    return [
        ("Left", images.get("left"), None),
        ("Right", images.get("right"), None),
        ("Disparity", images.get("disparity"), "inferno"),
    ]


def _render_panel(ax, title: str, image, cmap: str | None) -> None:
    ax.set_title(title)
    if image is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "missing", ha="center", va="center")
        return
    array = image
    if hasattr(array, "ndim") and array.ndim == 3:
        array = array[..., ::-1]
        ax.imshow(array)
    else:
        ax.imshow(array, cmap=cmap or "gray")
    ax.axis("off")


MODE_HANDLERS: dict[ModeLiteral, ModeHandler] = {
    "show-data": _run_show_data,
    "show-sample": _run_show_sample,
    "plot-frame": _run_plot_frame,
    "plot-undistort": lambda settings: _run_plot_frame(settings, undistort=True),
}


def main(cli_args: Sequence[str] | None = None) -> CLISettings:
    """Entrypoint executed by `python -m mob_map.cli` or console scripts."""

    args_list = list(cli_args) if cli_args is not None else None
    return CliApp.run(CLISettings, cli_args=args_list)


if __name__ == "__main__":  # pragma: no cover - script invocation
    main()
