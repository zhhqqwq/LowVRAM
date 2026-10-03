"""CPU collection helpers."""

import os
import platform
from pathlib import Path

import psutil

from lowvram.models.hardware import CPUInfo


def _linux_cpu_name() -> str | None:
    """Read the first Linux CPU model name when /proc/cpuinfo is available."""
    cpuinfo = Path("/proc/cpuinfo")
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"model name", "hardware", "processor"}:
            candidate = value.strip()
            if candidate:
                return candidate
    return None


def _cpu_name() -> str:
    """Return a stable human-readable CPU name without exposing host identity."""
    if platform.system() == "Linux":
        linux_name = _linux_cpu_name()
        if linux_name:
            return linux_name

    windows_name = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if windows_name:
        return windows_name

    processor = platform.processor().strip()
    if processor:
        return processor

    machine = platform.machine().strip()
    return machine or "unknown CPU"


def collect_cpu() -> CPUInfo:
    """Collect CPU identity and core counts."""
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    return CPUInfo(
        name=_cpu_name(),
        architecture=platform.machine() or None,
        physical_cores=physical if physical and physical > 0 else None,
        logical_cores=logical if logical and logical > 0 else None,
    )
