"""LowVRAM command-line interface."""

import json
from pathlib import Path
from typing import Annotated

import typer

from lowvram.collectors import collect_system
from lowvram.validators import DataValidationError, validate_file

app = typer.Typer(
    name="lowvram",
    help="Inspect local hardware and validate LowVRAM data contracts.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """LowVRAM commands."""


@app.command("system")
def system_command() -> None:
    """Collect local hardware information and print structured JSON."""
    hardware = collect_system()
    typer.echo(json.dumps(hardware.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a LowVRAM JSON document."""
    try:
        validate_file(file)
    except DataValidationError as exc:
        typer.echo("INVALID")
        typer.echo("")
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo("VALID")


if __name__ == "__main__":
    app()
