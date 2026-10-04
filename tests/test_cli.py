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


def _doctor_result(*, ready: bool):
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


def _dry_run_result(*, ready: bool):
    from lowvram.models.benchmark import ErrorType, RuntimeInfo
    from lowvram.models.command import LlamaCppCommand
    from lowvram.models.dry_run import (
        BenchmarkDryRunConfiguration,
        BenchmarkDryRunResult,
    )

    configuration = BenchmarkDryRunConfiguration(
        context_length=4096,
        threads=8,
        gpu_layers=0,
        batch_size=512,
        temperature=0.0,
        seed=42,
    )
    if ready:
        command = LlamaCppCommand(
            executable="/opt/llama/llama-cli",
            arguments=("--model", "model.gguf"),
            argv=("/opt/llama/llama-cli", "--model", "model.gguf"),
        )
        return BenchmarkDryRunResult(
            ready=True,
            runtime=RuntimeInfo(name="llama.cpp", version="b9001"),
            model="model.gguf",
            configuration=configuration,
            prompt_version="v1",
            command=command,
        )
    return BenchmarkDryRunResult(
        ready=False,
        runtime=RuntimeInfo(name="llama.cpp", version=None),
        model="missing.gguf",
        configuration=configuration,
        prompt_version="v1",
        error_type=ErrorType.MODEL_NOT_FOUND,
        error_message="model file not found",
    )


def test_help_lists_benchmark_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "benchmark" in result.stdout


def test_benchmark_without_dry_run_refuses_execution() -> None:
    result = runner.invoke(app, ["benchmark"])

    assert result.exit_code == 2
    assert "use --dry-run" in result.stdout


def test_benchmark_dry_run_human_output(monkeypatch) -> None:
    monkeypatch.setattr(
        lowvram.cli,
        "build_benchmark_dry_run",
        lambda request: _dry_run_result(ready=True),
    )

    result = runner.invoke(app, ["benchmark", "--dry-run"])

    assert result.exit_code == 0
    assert "DRY RUN READY" in result.stdout
    assert "Runtime: llama.cpp b9001" in result.stdout
    assert "context=4096" in result.stdout
    assert "Prompt version: v1" in result.stdout
    assert "Command argv:" in result.stdout


def test_benchmark_dry_run_json_output(monkeypatch) -> None:
    captured = {}

    def fake_build(request):
        captured["request"] = request
        return _dry_run_result(ready=True)

    monkeypatch.setattr(lowvram.cli, "build_benchmark_dry_run", fake_build)

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--dry-run",
            "--model",
            "example.gguf",
            "--llama-cli",
            "/custom/llama-cli",
            "--context",
            "8192",
            "--threads",
            "12",
            "--gpu-layers",
            "24",
            "--batch",
            "256",
            "--temperature",
            "0.25",
            "--seed",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "p1.13.0"
    assert payload["ready"] is True
    request = captured["request"]
    assert request.model_path == "example.gguf"
    assert request.llama_cli == "/custom/llama-cli"
    assert request.context_length == 8192
    assert request.threads == 12
    assert request.gpu_layers == 24
    assert request.batch_size == 256
    assert request.temperature == 0.25
    assert request.seed == 7
    assert request.prompt_version == "v1"


def test_benchmark_dry_run_blocked_returns_exit_one(monkeypatch) -> None:
    monkeypatch.setattr(
        lowvram.cli,
        "build_benchmark_dry_run",
        lambda request: _dry_run_result(ready=False),
    )

    result = runner.invoke(app, ["benchmark", "--dry-run", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ready"] is False
    assert payload["error_type"] == "model_not_found"
