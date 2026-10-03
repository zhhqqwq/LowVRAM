"""Top-level system collector."""

import platform

from lowvram.collectors.cpu import collect_cpu
from lowvram.collectors.memory import collect_ram_total_mb
from lowvram.collectors.nvidia import collect_nvidia
from lowvram.models.hardware import HardwareInfo, OSInfo


def collect_system() -> HardwareInfo:
    """Collect the P1-01 hardware snapshot used by `lowvram system`."""
    gpus, driver = collect_nvidia()
    return HardwareInfo(
        cpu=collect_cpu(),
        gpu=gpus,
        ram_total_mb=collect_ram_total_mb(),
        os=OSInfo(
            name=platform.system() or "unknown",
            version=platform.release() or None,
            architecture=platform.machine() or None,
        ),
        driver=driver,
        python_version=platform.python_version(),
    )
