"""Hardware collectors used by LowVRAM P1."""

from lowvram.collectors.nvidia import collect_nvidia_snapshot
from lowvram.collectors.system import collect_system

__all__ = ["collect_nvidia_snapshot", "collect_system"]
