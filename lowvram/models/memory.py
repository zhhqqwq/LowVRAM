"""P1 RAM monitoring result models."""

from typing import Annotated

from pydantic import Field, model_validator

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel

PositiveFloat = Annotated[float, Field(gt=0)]


class RamSample(StrictModel):
    """One RAM sample for the monitored process tree and whole system."""

    process_ram_mb: NonNegativeInt
    system_ram_mb: NonNegativeInt


class RamMonitorResult(StrictModel):
    """Aggregated RAM measurements for one monitor lifecycle."""

    baseline_process_ram_mb: NonNegativeInt
    current_process_ram_mb: NonNegativeInt
    peak_process_ram_mb: NonNegativeInt
    delta_process_ram_mb: NonNegativeInt

    baseline_system_ram_mb: NonNegativeInt
    current_system_ram_mb: NonNegativeInt
    peak_system_ram_mb: NonNegativeInt
    delta_system_ram_mb: NonNegativeInt

    sample_count: PositiveInt
    sample_interval_seconds: PositiveFloat

    @model_validator(mode="after")
    def validate_aggregates(self) -> "RamMonitorResult":
        """Keep baseline/current/peak/delta relationships internally consistent."""
        if self.peak_process_ram_mb < self.baseline_process_ram_mb:
            raise ValueError("peak_process_ram_mb must be >= baseline_process_ram_mb")
        if self.peak_process_ram_mb < self.current_process_ram_mb:
            raise ValueError("peak_process_ram_mb must be >= current_process_ram_mb")
        if self.delta_process_ram_mb != (
            self.peak_process_ram_mb - self.baseline_process_ram_mb
        ):
            raise ValueError(
                "delta_process_ram_mb must equal peak_process_ram_mb - baseline_process_ram_mb"
            )

        if self.peak_system_ram_mb < self.baseline_system_ram_mb:
            raise ValueError("peak_system_ram_mb must be >= baseline_system_ram_mb")
        if self.peak_system_ram_mb < self.current_system_ram_mb:
            raise ValueError("peak_system_ram_mb must be >= current_system_ram_mb")
        if self.delta_system_ram_mb != self.peak_system_ram_mb - self.baseline_system_ram_mb:
            raise ValueError(
                "delta_system_ram_mb must equal peak_system_ram_mb - baseline_system_ram_mb"
            )
        return self
