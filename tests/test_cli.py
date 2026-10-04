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


def _doctor_result(*, ready: bool) -> "DoctorResult":
    from lowvram.models.doctor import (
        DoctorCheck,
        DoctorCheckName,
        DoctorCheckStatus,
        DoctorResult,
    )

    status = DoctorCheckStatus.PASS if ready else DoctorCheckStatus.BLOCK
    check = DoctorCheck(
        name=DoctorCheckName.LLAMA_CPP,
        status=status,
        message="ready" if ready else "llama.cpp missing",
    )
    return DoctorResult(
        ready=ready,
        summary="READY" if ready else "BLOCKED",
        checks=(check,),
        blocking_checks=() if ready else (DoctorCheckName.LLAMA_CPP,),
    )


def test_help_lists_doctor_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout


def test_doctor_human_output_ready_and_exit_zero(monkeypatch) -> None:
    monkeypatch.setattr(lowvram.cli, "run_doctor", lambda **kwargs: _doctor_result(ready=True))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "PASS" in result.stdout
    assert result.stdout.strip().endswith("READY")


def test_doctor_json_output_is_machine_readable(monkeypatch) -> None:
    monkeypatch.setattr(lowvram.cli, "run_doctor", lambda **kwargs: _doctor_result(ready=True))

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "p1.12.0"
    assert payload["ready"] is True
    assert payload["summary"] == "READY"


def test_doctor_blocked_returns_exit_one_after_output(monkeypatch) -> None:
    monkeypatch.setattr(lowvram.cli, "run_doctor", lambda **kwargs: _doctor_result(ready=False))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "BLOCKED" in result.stdout
    assert "Blocking checks: llama_cpp" in result.stdout


def test_doctor_forwards_paths_without_typer_prevalidation(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_run_doctor(**kwargs):
        captured.update(kwargs)
        return _doctor_result(ready=True)

    monkeypatch.setattr(lowvram.cli, "run_doctor", fake_run_doctor)
    model = tmp_path / "missing.gguf"
    output_dir = tmp_path / "missing-output"

    result = runner.invoke(
        app,
        [
            "doctor",
            "--llama-cli",
            "/custom/llama-cli",
            "--model",
            str(model),
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "llama_cli": "/custom/llama-cli",
        "model_path": model,
        "output_dir": output_dir,
    }
