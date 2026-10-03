"""P1 NVIDIA snapshot models."""

from typing import Annotated

from pydantic import Field, model_validator

from lowvram.models.base import NonNegativeInt, StrictModel

Percentage = Annotated[float, Field(ge=0, le=100)]


class NvidiaGPUStatus(StrictModel):
    """One NVIDIA GPU sample captured at a single point in time."""

    index: NonNegativeInt
    name: str = Field(min_length=1)
    vram_total_mb: NonNegativeInt
    vram_used_mb: NonNegativeInt | None = None
    gpu_utilization_percent: Percentage | None = None

    @model_validator(mode="after")
    def validate_memory_usage(self) -> "NvidiaGPUStatus":
        """Reject samples whose used VRAM exceeds the reported total."""
        if self.vram_used_mb is not None and self.vram_used_mb > self.vram_total_mb:
            raise ValueError("vram_used_mb must be <= vram_total_mb")
        return self


class NvidiaSnapshot(StrictModel):
    """A one-shot NVIDIA state snapshot, not a time-series monitor."""

    gpus: list[NvidiaGPUStatus] = Field(default_factory=list)
    driver_version: str | None = None
    cuda_version: str | None = None
