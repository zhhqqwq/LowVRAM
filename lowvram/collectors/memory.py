"""System-memory collection helpers."""

import psutil


def collect_ram_total_mb() -> int:
    """Return total physical RAM in MiB-compatible MB units used by LowVRAM."""
    total_bytes = int(psutil.virtual_memory().total)
    return max(1, total_bytes // (1024 * 1024))
