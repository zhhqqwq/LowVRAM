"""Typer CLI tests."""

from pathlib import Path

from typer.testing import CliRunner

from lowvram.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def test_help_starts() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "validate" in result.stdout


def test_validate_valid_file() -> None:
    path = FIXTURES / "valid" / "01_dense_single_gpu_success.json"
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0
    assert result.stdout.strip() == "VALID"


def test_validate_invalid_file_has_field_path() -> None:
    path = FIXTURES / "invalid" / "01_negative_vram.json"
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "INVALID" in result.stdout
    assert "hardware.gpu.0.vram_total_mb" in result.stdout
