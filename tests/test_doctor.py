"""P1-12 Doctor Command tests."""

import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import lowvram.doctor as doctor_module
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.doctor import (
    DoctorCheck,
    DoctorCheckName,
    DoctorCheckStatus,
    DoctorResult,
)
from lowvram.models.hardware import HardwareInfo
from lowvram.models.nvidia import NvidiaGPUStatus, NvidiaSnapshot


def _hardware(ram_total_mb: int = 32768) -> HardwareInfo:
    return HardwareInfo.model_validate(
        {
            "cpu": {
                "name": "Example CPU",
                "architecture": "x86_64",
                "physical_cores": 8,
                "logical_cores": 16,
            },
            "gpu": [],
            "ram_total_mb": ram_total_mb,
            "os": {"name": "Linux", "version": "6.8", "architecture": "x86_64"},
            "driver": {},
            "python_version": "3.11.9",
        }
    )


def _verified_detection() -> LlamaCppDetectionResult:
    return LlamaCppDetectionResult(
        found=True,
        executable="/opt/llama/llama-cli",
        source="explicit_path",
        candidate_name="llama-cli",
        runnable=True,
        version="b9001",
        version_command=["/opt/llama/llama-cli", "--version"],
        verified_eligible=True,
        message="detected",
    )


def _patch_ready_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(doctor_module, "detect_llama_cpp", lambda path: _verified_detection())
    monkeypatch.setattr(
        doctor_module,
        "collect_nvidia_snapshot",
        lambda: NvidiaSnapshot(),
    )
    monkeypatch.setattr(doctor_module, "collect_system", _hardware)


def test_ready_cpu_only_machine_is_not_blocked_by_missing_nvidia(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch)

    result = doctor_module.run_doctor(output_dir=tmp_path)

    assert result.ready is True
    assert result.summary == "READY"
    by_name = {check.name: check for check in result.checks}
    assert by_name[DoctorCheckName.LLAMA_CPP].status == DoctorCheckStatus.PASS
    assert by_name[DoctorCheckName.NVIDIA].status == DoctorCheckStatus.WARN
    assert by_name[DoctorCheckName.CUDA].status == DoctorCheckStatus.WARN
    assert by_name[DoctorCheckName.PERMISSIONS].status == DoctorCheckStatus.PASS
    assert by_name[DoctorCheckName.DISK].status == DoctorCheckStatus.PASS
    assert by_name[DoctorCheckName.RAM].details["ram_total_mb"] == 32768


def test_python_below_minimum_is_blocking(monkeypatch, tmp_path: Path) -> None:
    _patch_ready_dependencies(monkeypatch)
    monkeypatch.setattr(
        doctor_module.sys,
        "version_info",
        SimpleNamespace(major=3, minor=10),
    )
    monkeypatch.setattr(doctor_module.platform, "python_version", lambda: "3.10.14")

    result = doctor_module.run_doctor(output_dir=tmp_path)

    by_name = {check.name: check for check in result.checks}
    assert by_name[DoctorCheckName.PYTHON].status == DoctorCheckStatus.BLOCK
    assert result.ready is False


def test_nvidia_probe_failure_is_distinct_from_no_gpu(monkeypatch, tmp_path: Path) -> None:
    _patch_ready_dependencies(monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "collect_nvidia_snapshot",
        lambda: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )

    result = doctor_module.run_doctor(output_dir=tmp_path)
    by_name = {check.name: check for check in result.checks}

    assert by_name[DoctorCheckName.NVIDIA].status == DoctorCheckStatus.WARN
    assert "could not be determined" in by_name[DoctorCheckName.NVIDIA].message
    assert by_name[DoctorCheckName.NVIDIA].details["probe_error_type"] == "RuntimeError"
    assert by_name[DoctorCheckName.CUDA].status == DoctorCheckStatus.WARN


def test_nvidia_and_cuda_are_reported_when_available(monkeypatch, tmp_path: Path) -> None:
    _patch_ready_dependencies(monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "collect_nvidia_snapshot",
        lambda: NvidiaSnapshot(
            gpus=[
                NvidiaGPUStatus(
                    index=0,
                    name="NVIDIA GeForce RTX 4060",
                    vram_total_mb=8192,
                    vram_used_mb=1024,
                    gpu_utilization_percent=10,
                )
            ],
            driver_version="555.42",
            cuda_version="12.5",
        ),
    )

    result = doctor_module.run_doctor(output_dir=tmp_path)
    by_name = {check.name: check for check in result.checks}

    assert by_name[DoctorCheckName.NVIDIA].status == DoctorCheckStatus.PASS
    assert by_name[DoctorCheckName.NVIDIA].details["gpu_count"] == 1
    assert by_name[DoctorCheckName.CUDA].status == DoctorCheckStatus.PASS
    assert by_name[DoctorCheckName.CUDA].details["cuda_version"] == "12.5"


def test_unrecognized_llama_version_blocks_verified_readiness(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "detect_llama_cpp",
        lambda path: LlamaCppDetectionResult(
            found=True,
            executable="/opt/llama/llama-cli",
            source="explicit_path",
            candidate_name="llama-cli",
            runnable=True,
            version=None,
            verified_eligible=False,
            message="version output was not recognized",
        ),
    )

    result = doctor_module.run_doctor(output_dir=tmp_path)

    assert result.ready is False
    assert DoctorCheckName.LLAMA_CPP in result.blocking_checks


def test_missing_model_blocks_when_model_was_explicitly_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch)

    result = doctor_module.run_doctor(
        model_path=tmp_path / "missing.gguf",
        output_dir=tmp_path,
    )

    assert result.ready is False
    check = next(check for check in result.checks if check.name == DoctorCheckName.MODEL)
    assert check.status == DoctorCheckStatus.BLOCK


def test_readable_nonempty_model_passes_without_loading_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF-data")

    result = doctor_module.run_doctor(model_path=model, output_dir=tmp_path)

    check = next(check for check in result.checks if check.name == DoctorCheckName.MODEL)
    assert check.status == DoctorCheckStatus.PASS
    assert check.details["size_bytes"] == len(b"GGUF-data")


def test_unwritable_output_directory_blocks_permissions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch)

    def fail_mkstemp(*args, **kwargs):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(doctor_module.tempfile, "mkstemp", fail_mkstemp)

    result = doctor_module.run_doctor(output_dir=tmp_path)

    assert result.ready is False
    by_name = {check.name: check for check in result.checks}
    assert by_name[DoctorCheckName.PERMISSIONS].status == DoctorCheckStatus.BLOCK
    assert by_name[DoctorCheckName.DISK].status == DoctorCheckStatus.PASS


def test_missing_output_directory_blocks_permissions_and_disk(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_ready_dependencies(monkeypatch)

    result = doctor_module.run_doctor(output_dir=tmp_path / "missing")

    assert result.ready is False
    assert result.blocking_checks == (
        DoctorCheckName.PERMISSIONS,
        DoctorCheckName.DISK,
    )


def test_disk_usage_failure_is_blocking(monkeypatch, tmp_path: Path) -> None:
    _patch_ready_dependencies(monkeypatch)
    monkeypatch.setattr(
        doctor_module.shutil,
        "disk_usage",
        lambda path: (_ for _ in ()).throw(OSError("disk query failed")),
    )

    result = doctor_module.run_doctor(output_dir=tmp_path)

    by_name = {check.name: check for check in result.checks}
    assert by_name[DoctorCheckName.DISK].status == DoctorCheckStatus.BLOCK
    assert result.ready is False


def test_ram_collection_failure_is_blocking(monkeypatch, tmp_path: Path) -> None:
    _patch_ready_dependencies(monkeypatch)
    monkeypatch.setattr(
        doctor_module,
        "collect_system",
        lambda: (_ for _ in ()).throw(RuntimeError("RAM probe failed")),
    )

    result = doctor_module.run_doctor(output_dir=tmp_path)

    check = next(check for check in result.checks if check.name == DoctorCheckName.RAM)
    assert check.status == DoctorCheckStatus.BLOCK
    assert result.ready is False


def test_doctor_does_not_leave_permission_probe_file(monkeypatch, tmp_path: Path) -> None:
    _patch_ready_dependencies(monkeypatch)

    result = doctor_module.run_doctor(output_dir=tmp_path)

    assert result.ready is True
    assert list(tmp_path.iterdir()) == []


def test_doctor_result_rejects_inconsistent_ready_state() -> None:
    check = DoctorCheck(
        name=DoctorCheckName.RAM,
        status=DoctorCheckStatus.BLOCK,
        message="RAM unavailable",
    )
    with pytest.raises(ValidationError, match="blocking_checks"):
        DoctorResult(
            ready=True,
            summary="READY",
            checks=(check,),
            blocking_checks=(),
        )


def test_model_readability_uses_permission_check(monkeypatch, tmp_path: Path) -> None:
    _patch_ready_dependencies(monkeypatch)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    real_access = os.access

    def fake_access(path, mode):
        if Path(path) == model and mode == os.R_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(doctor_module.os, "access", fake_access)

    result = doctor_module.run_doctor(model_path=model, output_dir=tmp_path)

    check = next(check for check in result.checks if check.name == DoctorCheckName.MODEL)
    assert check.status == DoctorCheckStatus.BLOCK
