"""P1 VRAM monitoring result models."""

from typing import Annotated

from pydantic import Field, model_validator

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel

PositiveFloat = Annotated[float, Field(gt=0)]


class VramGPUUsage(StrictModel):
    """One GPU VRAM measurement captured at a single point in time."""

    index: NonNegativeInt
    name: str = Field(min_length=1)
    vram_total_mb: NonNegativeInt
    vram_used_mb: NonNegativeInt

    @model_validator(mode="after")
    def validate_usage(self) -> "VramGPUUsage":
        """Reject measurements whose used VRAM exceeds total VRAM."""
        if self.vram_used_mb > self.vram_total_mb:
            raise ValueError("vram_used_mb must be <= vram_total_mb")
        return self


class VramSample(StrictModel):
    """One multi-GPU VRAM sample."""

    gpus: list[VramGPUUsage] = Field(default_factory=list)
    total_vram_used_mb: NonNegativeInt

    @model_validator(mode="after")
    def validate_total(self) -> "VramSample":
        """Require unique GPU indexes and an exact aggregate total."""
        indexes = [gpu.index for gpu in self.gpus]
        if len(indexes) != len(set(indexes)):
            raise ValueError("GPU indexes must be unique")
        if self.total_vram_used_mb != sum(gpu.vram_used_mb for gpu in self.gpus):
            raise ValueError("total_vram_used_mb must equal the sum of GPU usage")
        return self


class VramGPUResult(StrictModel):
    """Aggregated VRAM measurements for one GPU."""

    index: NonNegativeInt
    name: str = Field(min_length=1)
    vram_total_mb: NonNegativeInt
    baseline_vram_mb: NonNegativeInt
    current_vram_mb: NonNegativeInt
    peak_vram_mb: NonNegativeInt
    delta_vram_mb: NonNegativeInt

    @model_validator(mode="after")
    def validate_aggregates(self) -> "VramGPUResult":
        """Keep per-GPU baseline/current/peak/delta values consistent."""
        if self.peak_vram_mb < self.baseline_vram_mb:
            raise ValueError("peak_vram_mb must be >= baseline_vram_mb")
        if self.peak_vram_mb < self.current_vram_mb:
            raise ValueError("peak_vram_mb must be >= current_vram_mb")
        if self.delta_vram_mb != self.peak_vram_mb - self.baseline_vram_mb:
            raise ValueError("delta_vram_mb must equal peak_vram_mb - baseline_vram_mb")
        if self.peak_vram_mb > self.vram_total_mb:
            raise ValueError("peak_vram_mb must be <= vram_total_mb")
        return self


class VramMonitorResult(StrictModel):
    """Aggregated multi-GPU VRAM measurements for one monitor lifecycle."""

    gpus: list[VramGPUResult] = Field(default_factory=list)
    baseline_vram_mb: NonNegativeInt
    current_vram_mb: NonNegativeInt
    peak_vram_mb: NonNegativeInt
    delta_vram_mb: NonNegativeInt
    sample_count: PositiveInt
    sample_interval_seconds: PositiveFloat

    @model_validator(mode="after")
    def validate_aggregates(self) -> "VramMonitorResult":
        """Validate aggregate values without summing per-GPU peaks across time."""
        if self.peak_vram_mb < self.baseline_vram_mb:
            raise ValueError("peak_vram_mb must be >= baseline_vram_mb")
        if self.peak_vram_mb < self.current_vram_mb:
            raise ValueError("peak_vram_mb must be >= current_vram_mb")
        if self.delta_vram_mb != self.peak_vram_mb - self.baseline_vram_mb:
            raise ValueError("delta_vram_mb must equal peak_vram_mb - baseline_vram_mb")

        baseline_sum = sum(gpu.baseline_vram_mb for gpu in self.gpus)
        current_sum = sum(gpu.current_vram_mb for gpu in self.gpus)
        if self.baseline_vram_mb != baseline_sum:
            raise ValueError("baseline_vram_mb must equal the per-GPU baseline sum")
        if self.current_vram_mb != current_sum:
            raise ValueError("current_vram_mb must equal the per-GPU current sum")
        return self
