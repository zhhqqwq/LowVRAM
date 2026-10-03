"""Best-effort NVIDIA GPU collection through nvidia-smi."""

import re
import subprocess

from lowvram.models.hardware import DriverInfo, GPUInfo

_QUERY_ARGS = [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]
_CUDA_PATTERN = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)


def _run_nvidia_smi(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run nvidia-smi without making its absence fatal to system collection."""
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


def _parse_gpu_rows(output: str, cuda_version: str | None) -> list[GPUInfo]:
    """Parse nvidia-smi CSV rows into strict GPU records."""
    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue

        name, memory_text, driver_version = fields
        try:
            vram_total_mb = int(float(memory_text))
        except ValueError:
            continue

        if not name or vram_total_mb < 0:
            continue

        gpus.append(
            GPUInfo(
                vendor="nvidia",
                name=name,
                vram_total_mb=vram_total_mb,
                driver_version=driver_version or None,
                cuda_version=cuda_version,
            )
        )
    return gpus


def _detect_cuda_version() -> str | None:
    """Read the CUDA compatibility version reported by nvidia-smi."""
    result = _run_nvidia_smi([])
    if result is None:
        return None

    match = _CUDA_PATTERN.search(result.stdout)
    return match.group(1) if match else None


def collect_nvidia() -> tuple[list[GPUInfo], DriverInfo]:
    """Collect NVIDIA GPUs plus machine-level driver/CUDA information."""
    query = _run_nvidia_smi(_QUERY_ARGS)
    if query is None:
        return [], DriverInfo()

    cuda_version = _detect_cuda_version()
    gpus = _parse_gpu_rows(query.stdout, cuda_version)
    driver_version = next(
        (gpu.driver_version for gpu in gpus if gpu.driver_version is not None),
        None,
    )
    return gpus, DriverInfo(
        nvidia_driver_version=driver_version,
        cuda_version=cuda_version,
    )
