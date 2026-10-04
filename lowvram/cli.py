"""LowVRAM command-line interface."""

import json
from pathlib import Path
from typing import Annotated

import typer

from lowvram.collectors import collect_system
from lowvram.doctor import run_doctor
from lowvram.dry_run import build_benchmark_dry_run
from lowvram.models.preparation import BenchmarkPreparationRequest
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


@app.command("benchmark")
def benchmark_command(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview the exact benchmark command without running it."),
    ] = False,
    model: Annotated[
        Path,
        typer.Option("--model", help="Local GGUF model path."),
    ] = Path("model.gguf"),
    llama_cli: Annotated[
        str | None,
        typer.Option("--llama-cli", help="Explicit llama.cpp executable path."),
    ] = None,
    context: Annotated[
        int,
        typer.Option("--context", min=1, help="Context length."),
    ] = 4096,
    threads: Annotated[
        int,
        typer.Option("--threads", min=1, help="CPU thread count."),
    ] = 8,
    gpu_layers: Annotated[
        int,
        typer.Option("--gpu-layers", min=0, help="Layers offloaded to GPU."),
    ] = 0,
    batch: Annotated[
        int,
        typer.Option("--batch", min=1, help="llama.cpp batch size."),
    ] = 512,
    temperature: Annotated[
        float,
        typer.Option("--temperature", min=0.0, help="Sampling temperature."),
    ] = 0.0,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Deterministic benchmark seed."),
    ] = 42,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print only machine-readable JSON."),
    ] = False,
) -> None:
    """Preview a benchmark command without executing the model."""
    if not dry_run:
        typer.echo("Benchmark execution CLI is not implemented in P1-13; use --dry-run.")
        raise typer.Exit(code=2)

    result = build_benchmark_dry_run(
        BenchmarkPreparationRequest(
            model_path=str(model),
            llama_cli=llama_cli,
            context_length=context,
            threads=threads,
            gpu_layers=gpu_layers,
            batch_size=batch,
            temperature=temperature,
            seed=seed,
            prompt_version="v1",
        )
    )

    if json_output:
        typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        typer.echo("DRY RUN READY" if result.ready else "DRY RUN BLOCKED")
        typer.echo(f"Runtime: {result.runtime.name} {result.runtime.version or 'unknown'}")
        typer.echo(f"Model: {result.model}")
        typer.echo(
            "Configuration: "
            f"context={result.configuration.context_length}, "
            f"threads={result.configuration.threads}, "
            f"gpu_layers={result.configuration.gpu_layers}, "
            f"batch={result.configuration.batch_size}, "
            f"temperature={result.configuration.temperature}, "
            f"seed={result.configuration.seed}"
        )
        typer.echo(f"Prompt version: {result.prompt_version}")
        if result.command is not None:
            typer.echo(
                "Command argv: "
                + json.dumps(list(result.command.argv), ensure_ascii=False)
            )
        if not result.ready:
            typer.echo(f"Error: {result.error_type}: {result.error_message}")

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
