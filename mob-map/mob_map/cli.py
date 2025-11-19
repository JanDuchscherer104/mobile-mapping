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

ModeLiteral = Literal["show-data", "show-sample"]
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
    index = settings.sample_index
    if index >= len(dataset):
        console.error(
            f"Requested sample {index} exceeds dataset length ({len(dataset)} frames)."
        )
        return

    sample = dataset[index]
    images = sample.images
    shapes = {
        name: (value.shape if value is not None else None)
        for name, value in images.items()
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


def _build_console(settings: CLISettings, *parts: str) -> Console:
    return Console.with_prefix("mob-map", "cli", *parts).set_verbose(not settings.quiet).set_debug(settings.debug)


MODE_HANDLERS: dict[ModeLiteral, ModeHandler] = {
    "show-data": _run_show_data,
    "show-sample": _run_show_sample,
}


def main(cli_args: Sequence[str] | None = None) -> CLISettings:
    """Entrypoint executed by `python -m mob_map.cli` or console scripts."""

    args_list = list(cli_args) if cli_args is not None else None
    return CliApp.run(CLISettings, cli_args=args_list)


if __name__ == "__main__":  # pragma: no cover - script invocation
    main()
