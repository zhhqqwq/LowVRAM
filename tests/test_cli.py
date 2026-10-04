"""Typer CLI tests."""

import json
from pathlib import Path

from typer.testing import CliRunner

import lowvram.cli
from lowvram.cli import app
from lowvram.models.hardware import HardwareInfo

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def test_help_starts() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "system" in result.stdout
    assert "validate" in result.stdout


def test_system_outputs_all_p1_01_fields_as_structured_json(monkeypatch) -> None:
    hardware = HardwareInfo.model_validate(
        {
            "cpu": {
                "name": "Example CPU",
                "architecture": "x86_64",
                "physical_cores": 8,
                "logical_cores": 16,
            },
            "gpu": [
                {
                    "vendor": "nvidia",
                    "name": "NVIDIA GeForce RTX 4060",
                    "vram_total_mb": 8192,
                    "driver_version": "555.42",
                    "cuda_version": "12.5",
                }
            ],
            "ram_total_mb": 32768,
            "os": {"name": "Linux", "version": "6.8", "architecture": "x86_64"},
            "driver": {
                "nvidia_driver_version": "555.42",
                "cuda_version": "12.5",
            },
            "python_version": "3.11.9",
        }
    )
    monkeypatch.setattr(lowvram.cli, "collect_system", lambda: hardware)

    result = runner.invoke(app, ["system"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    assert payload["os"] == {
        "name": "Linux",
        "version": "6.8",
        "architecture": "x86_64",
    }
    assert payload["cpu"] == {
        "name": "Example CPU",
        "architecture": "x86_64",
        "physical_cores": 8,
        "logical_cores": 16,
    }
    assert payload["ram_total_mb"] == 32768

    assert len(payload["gpu"]) == 1
    assert payload["gpu"][0]["vendor"] == "nvidia"
    assert payload["gpu"][0]["name"] == "NVIDIA GeForce RTX 4060"
    assert payload["gpu"][0]["vram_total_mb"] == 8192
    assert payload["gpu"][0]["driver_version"] == "555.42"
    assert payload["gpu"][0]["cuda_version"] == "12.5"

    assert payload["driver"]["nvidia_driver_version"] == "555.42"
    assert payload["driver"]["cuda_version"] == "12.5"
    assert payload["python_version"] == "3.11.9"


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
