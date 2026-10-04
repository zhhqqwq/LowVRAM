"""LowVRAM command-line interface."""

import json
from pathlib import Path
from typing import Annotated

import typer

from lowvram.collectors import collect_system
from lowvram.doctor import run_doctor
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


@app.command("doctor")
def doctor_command(
    llama_cli: Annotated[
        str | None,
        typer.Option("--llama-cli", help="Explicit llama.cpp executable path."),
    ] = None,
    model: Annotated[
        Path | None,
        typer.Option("--model", help="Optional model file to validate without loading it."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Output directory to test. Defaults to the current directory.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print only machine-readable JSON."),
    ] = False,
) -> None:
    """Check whether the local environment is ready for LowVRAM benchmarking."""
    result = run_doctor(
        llama_cli=llama_cli,
        model_path=model,
        output_dir=output_dir,
    )

    if json_output:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        for check in result.checks:
            label = check.status.value.upper()
            typer.echo(f"{label:5} {check.name.value}: {check.message}")
        typer.echo("")
        typer.echo(result.summary)
        if not result.ready:
            typer.echo(
                "Blocking checks: "
                + ", ".join(check.value for check in result.blocking_checks)
            )

    if not result.ready:
        raise typer.Exit(code=1)


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
