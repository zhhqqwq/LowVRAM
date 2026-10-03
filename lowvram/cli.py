"""LowVRAM P0 command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from lowvram.validators import DataValidationError, validate_file

app = typer.Typer(
    name="lowvram",
    help="Validate LowVRAM data contracts. P0 intentionally contains no benchmark runner.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """LowVRAM P0 validation commands."""


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
