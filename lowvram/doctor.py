"""P1-12 environment readiness diagnostics."""

import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

from lowvram.collectors import collect_nvidia_snapshot, collect_system
from lowvram.models.doctor import (
    DoctorCheck,
    DoctorCheckName,
    DoctorCheckStatus,
    DoctorResult,
)
from lowvram.models.hardware import HardwareInfo
from lowvram.models.nvidia import NvidiaSnapshot
from lowvram.runtime.detector import detect_llama_cpp

_MINIMUM_PYTHON = (3, 11)


def run_doctor(
    *,
    llama_cli: str | None = None,
    model_path: Path | None = None,
    output_dir: Path | None = None,
) -> DoctorResult:
    """Run non-model-executing environment checks and return structured readiness."""
    target_output_dir = output_dir or Path.cwd()

    checks: list[DoctorCheck] = [
        _check_python(),
        _check_llama_cpp(llama_cli),
    ]

    snapshot, nvidia_probe_error = _safe_nvidia_snapshot()
    checks.extend(
        (
            _check_nvidia(snapshot, nvidia_probe_error),
            _check_cuda(snapshot, nvidia_probe_error),
        )
    )

    checks.append(
        _check_model(model_path)
        if model_path is not None
        else DoctorCheck(
            name=DoctorCheckName.MODEL,
            status=DoctorCheckStatus.WARN,
            message="model-specific checks skipped because --model was not provided",
        )
    )

    permissions_check, disk_check = _check_output_directory(target_output_dir)
    checks.extend((permissions_check, disk_check))

    hardware = _safe_system_snapshot()
    checks.append(_check_ram(hardware))

    blocking_checks = tuple(
        check.name for check in checks if check.status == DoctorCheckStatus.BLOCK
    )
    ready = not blocking_checks
    return DoctorResult(
        ready=ready,
        summary="READY" if ready else "BLOCKED",
        checks=tuple(checks),
        blocking_checks=blocking_checks,
    )


def _check_python() -> DoctorCheck:
    version = platform.python_version()
    current = (sys.version_info.major, sys.version_info.minor)
    if current < _MINIMUM_PYTHON:
        return DoctorCheck(
            name=DoctorCheckName.PYTHON,
            status=DoctorCheckStatus.BLOCK,
            message="Python 3.11 or newer is required",
            details={"version": version, "minimum": "3.11"},
        )
    return DoctorCheck(
        name=DoctorCheckName.PYTHON,
        status=DoctorCheckStatus.PASS,
        message="Python version satisfies the LowVRAM requirement",
        details={"version": version, "minimum": "3.11"},
    )


def _check_llama_cpp(explicit_path: str | None) -> DoctorCheck:
    detection = detect_llama_cpp(explicit_path)
    details: dict[str, str | int | float | bool | None] = {
        "found": detection.found,
        "runnable": detection.runnable,
        "verified_eligible": detection.verified_eligible,
        "version": detection.version,
        "source": detection.source,
        "candidate_name": detection.candidate_name,
    }
    if detection.verified_eligible:
        return DoctorCheck(
            name=DoctorCheckName.LLAMA_CPP,
            status=DoctorCheckStatus.PASS,
            message="llama.cpp executable and version are ready",
            details=details,
        )

    if not detection.found:
        message = detection.message or "llama.cpp executable was not found"
    elif not detection.runnable:
        message = detection.message or "llama.cpp executable is not runnable"
    else:
        message = (
            detection.message
            or "llama.cpp version is not recognized; verified benchmarks are blocked"
        )
    return DoctorCheck(
        name=DoctorCheckName.LLAMA_CPP,
        status=DoctorCheckStatus.BLOCK,
        message=message,
        details=details,
    )


def _safe_nvidia_snapshot() -> tuple[NvidiaSnapshot | None, str | None]:
    try:
        return collect_nvidia_snapshot(), None
    except Exception as exc:
        return None, type(exc).__name__


def _check_nvidia(
    snapshot: NvidiaSnapshot | None,
    probe_error: str | None,
) -> DoctorCheck:
    if snapshot is None:
        return DoctorCheck(
            name=DoctorCheckName.NVIDIA,
            status=DoctorCheckStatus.WARN,
            message="NVIDIA availability could not be determined; CPU-only remains available",
            details={"probe_error_type": probe_error},
        )
    if not snapshot.gpus:
        return DoctorCheck(
            name=DoctorCheckName.NVIDIA,
            status=DoctorCheckStatus.WARN,
            message="no NVIDIA GPU detected; CPU-only benchmarking remains available",
            details={"gpu_count": 0, "driver_version": snapshot.driver_version},
        )
    return DoctorCheck(
        name=DoctorCheckName.NVIDIA,
        status=DoctorCheckStatus.PASS,
        message="NVIDIA GPU support is available",
        details={
            "gpu_count": len(snapshot.gpus),
            "driver_version": snapshot.driver_version,
        },
    )


def _check_cuda(
    snapshot: NvidiaSnapshot | None,
    probe_error: str | None,
) -> DoctorCheck:
    if snapshot is None:
        return DoctorCheck(
            name=DoctorCheckName.CUDA,
            status=DoctorCheckStatus.WARN,
            message="CUDA availability could not be determined from the NVIDIA probe",
            details={"probe_error_type": probe_error},
        )
    if not snapshot.gpus:
        return DoctorCheck(
            name=DoctorCheckName.CUDA,
            status=DoctorCheckStatus.WARN,
            message="CUDA availability is not required for CPU-only benchmarking",
            details={"cuda_version": snapshot.cuda_version},
        )
    if snapshot.cuda_version is None:
        return DoctorCheck(
            name=DoctorCheckName.CUDA,
            status=DoctorCheckStatus.WARN,
            message=(
                "NVIDIA GPU detected but nvidia-smi did not report CUDA compatibility; "
                "CPU-only benchmarking remains available"
            ),
            details={"cuda_version": None},
        )
    return DoctorCheck(
        name=DoctorCheckName.CUDA,
        status=DoctorCheckStatus.PASS,
        message="CUDA compatibility is reported by nvidia-smi",
        details={"cuda_version": snapshot.cuda_version},
    )


def _check_model(model_path: Path) -> DoctorCheck:
    if not model_path.is_file():
        return DoctorCheck(
            name=DoctorCheckName.MODEL,
            status=DoctorCheckStatus.BLOCK,
            message="specified model path is not a regular file",
        )
    if not os.access(model_path, os.R_OK):
        return DoctorCheck(
            name=DoctorCheckName.MODEL,
            status=DoctorCheckStatus.BLOCK,
            message="specified model file is not readable",
        )
    try:
        size_bytes = model_path.stat().st_size
    except OSError as exc:
        return DoctorCheck(
            name=DoctorCheckName.MODEL,
            status=DoctorCheckStatus.BLOCK,
            message=f"specified model file metadata cannot be read: {exc}",
        )
    if size_bytes <= 0:
        return DoctorCheck(
            name=DoctorCheckName.MODEL,
            status=DoctorCheckStatus.BLOCK,
            message="specified model file is empty",
            details={"size_bytes": size_bytes},
        )
    return DoctorCheck(
        name=DoctorCheckName.MODEL,
        status=DoctorCheckStatus.PASS,
        message="specified model file exists and is readable",
        details={
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        },
    )


def _check_output_directory(output_dir: Path) -> tuple[DoctorCheck, DoctorCheck]:
    if not output_dir.is_dir():
        blocked = DoctorCheck(
            name=DoctorCheckName.PERMISSIONS,
            status=DoctorCheckStatus.BLOCK,
            message="output directory does not exist or is not a directory",
        )
        disk = DoctorCheck(
            name=DoctorCheckName.DISK,
            status=DoctorCheckStatus.BLOCK,
            message="output disk cannot be checked without an existing output directory",
        )
        return blocked, disk

    write_error: OSError | None = None
    temporary_path: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".lowvram-doctor-",
            dir=output_dir,
        )
        os.close(descriptor)
        temporary_path = Path(raw_path)
    except OSError as exc:
        write_error = exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass

    if write_error is None:
        permissions = DoctorCheck(
            name=DoctorCheckName.PERMISSIONS,
            status=DoctorCheckStatus.PASS,
            message="output directory passed a create-and-delete write test",
        )
    else:
        permissions = DoctorCheck(
            name=DoctorCheckName.PERMISSIONS,
            status=DoctorCheckStatus.BLOCK,
            message=f"output directory is not writable: {write_error}",
        )

    try:
        usage = shutil.disk_usage(output_dir)
    except OSError as exc:
        disk = DoctorCheck(
            name=DoctorCheckName.DISK,
            status=DoctorCheckStatus.BLOCK,
            message=f"output disk usage cannot be read: {exc}",
        )
        return permissions, disk

    free_mb = usage.free // (1024 * 1024)
    if usage.free <= 0:
        disk = DoctorCheck(
            name=DoctorCheckName.DISK,
            status=DoctorCheckStatus.BLOCK,
            message="output filesystem reports no free disk space",
            details={"free_mb": free_mb},
        )
    else:
        disk = DoctorCheck(
            name=DoctorCheckName.DISK,
            status=DoctorCheckStatus.PASS,
            message="output filesystem reports available disk space",
            details={
                "free_mb": free_mb,
                "total_mb": usage.total // (1024 * 1024),
            },
        )
    return permissions, disk


def _safe_system_snapshot() -> HardwareInfo | None:
    try:
        return collect_system()
    except Exception:
        return None


def _check_ram(hardware: HardwareInfo | None) -> DoctorCheck:
    if hardware is None:
        return DoctorCheck(
            name=DoctorCheckName.RAM,
            status=DoctorCheckStatus.BLOCK,
            message="system RAM could not be detected",
        )
    return DoctorCheck(
        name=DoctorCheckName.RAM,
        status=DoctorCheckStatus.PASS,
        message="system RAM was detected",
        details={"ram_total_mb": hardware.ram_total_mb},
    )
