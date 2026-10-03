"""Hardware-specific validator compatibility entry point."""

from pathlib import Path

from lowvram.validators.core import validate_file


def validate_hardware(path: Path) -> None:
    """Validate one hardware JSON file."""
    validate_file(path, "hardware")
