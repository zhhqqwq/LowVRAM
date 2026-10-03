"""Best-effort NVIDIA GPU collection through nvidia-smi."""

import re
import subprocess

from pydantic import ValidationError

from lowvram.models.hardware import DriverInfo, GPUInfo
from lowvram.models.nvidia import NvidiaGPUStatus, NvidiaSnapshot

_QUERY_ARGS = [
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
    "--format=csv,noheader,nounits",
]
_CUDA_PATTERN = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)
_UNAVAILABLE_VALUES = {"", "n/a", "[not supported]", "not supported"}


def _run_nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run nvidia-smi without making its absence fatal to collection."""
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None
    return result


def _parse_required_nonnegative_int(value: str) -> int:
    """Parse an integer-valued nvidia-smi field that must be available."""
    parsed = float(value.strip())
    if parsed < 0 or not parsed.is_integer():
        raise ValueError("expected a non-negative integer")
    return int(parsed)


def _parse_optional_nonnegative_int(value: str) -> int | None:
    """Parse an optional integer-valued metric, preserving unsupported values as null."""
    normalized = value.strip()
    if normalized.lower() in _UNAVAILABLE_VALUES:
        return None
    return _parse_required_nonnegative_int(normalized)


def _parse_optional_percentage(value: str) -> float | None:
    """Parse an optional percentage metric from nvidia-smi nounits output."""
    normalized = value.strip()
    if normalized.lower() in _UNAVAILABLE_VALUES:
        return None
    parsed = float(normalized)
    if not 0 <= parsed <= 100:
        raise ValueError("percentage must be between 0 and 100")
    return parsed


def _parse_gpu_rows(output: str) -> tuple[list[NvidiaGPUStatus], str | None]:
    """Parse one-shot NVIDIA state rows and the shared driver version."""
    gpus: list[NvidiaGPUStatus] = []
    driver_version: str | None = None

    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue

        index_text, name, total_text, used_text, utilization_text, row_driver = fields
        try:
            gpu = NvidiaGPUStatus(
                index=_parse_required_nonnegative_int(index_text),
                name=name,
                vram_total_mb=_parse_required_nonnegative_int(total_text),
                vram_used_mb=_parse_optional_nonnegative_int(used_text),
                gpu_utilization_percent=_parse_optional_percentage(utilization_text),
            )
        except (ValueError, ValidationError):
            continue

        gpus.append(gpu)
        if driver_version is None and row_driver:
            driver_version = row_driver

    gpus.sort(key=lambda gpu: gpu.index)
    return gpus, driver_version


def _detect_cuda_version() -> str | None:
    """Read the CUDA compatibility version reported by nvidia-smi."""
    result = _run_nvidia_smi([])
    if result is None:
        return None

    match = _CUDA_PATTERN.search(result.stdout)
    return match.group(1) if match else None


def collect_nvidia_snapshot() -> NvidiaSnapshot:
    """Capture one NVIDIA state snapshot without starting a sampling loop."""
    query = _run_nvidia_smi(_QUERY_ARGS)
    if query is None:
        return NvidiaSnapshot()

    gpus, driver_version = _parse_gpu_rows(query.stdout)
    return NvidiaSnapshot(
        gpus=gpus,
        driver_version=driver_version,
        cuda_version=_detect_cuda_version(),
    )


def collect_nvidia() -> tuple[list[GPUInfo], DriverInfo]:
    """Project a one-shot NVIDIA snapshot into the static HardwareInfo contract."""
    snapshot = collect_nvidia_snapshot()
    gpus = [
        GPUInfo(
            vendor="nvidia",
            name=gpu.name,
            vram_total_mb=gpu.vram_total_mb,
            driver_version=snapshot.driver_version,
            cuda_version=snapshot.cuda_version,
        )
        for gpu in snapshot.gpus
    ]
    return gpus, DriverInfo(
        nvidia_driver_version=snapshot.driver_version,
        cuda_version=snapshot.cuda_version,
    )
