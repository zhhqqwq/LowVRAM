"""Hardware data contract."""

from typing import Literal

from pydantic import Field

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel


class CPUInfo(StrictModel):
    """CPU identity and topology."""

    name: str = Field(min_length=1)
    architecture: str | None = None
    physical_cores: PositiveInt | None = None
    logical_cores: PositiveInt | None = None


class GPUInfo(StrictModel):
    """NVIDIA GPU information supported by the P0 contract."""

    vendor: Literal["nvidia"]
    name: str = Field(min_length=1)
    vram_total_mb: NonNegativeInt
    driver_version: str | None = None
    cuda_version: str | None = None


class OSInfo(StrictModel):
    """Operating-system identity."""

    name: str = Field(min_length=1)
    version: str | None = None
    architecture: str | None = None


class DriverInfo(StrictModel):
    """Machine-level accelerator driver information."""

    nvidia_driver_version: str | None = None
    cuda_version: str | None = None


class HardwareInfo(StrictModel):
    """Portable hardware description using MB for all memory quantities."""

    cpu: CPUInfo
    gpu: list[GPUInfo] = Field(default_factory=list)
    ram_total_mb: PositiveInt
    os: OSInfo
    driver: DriverInfo = Field(default_factory=DriverInfo)
    python_version: str | None = None
