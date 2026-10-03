"""Benchmark-specific validator compatibility entry point."""

from pathlib import Path

from lowvram.validators.core import validate_file


def validate_benchmark(path: Path) -> None:
    """Validate one benchmark JSON file."""
    validate_file(path, "benchmark")
