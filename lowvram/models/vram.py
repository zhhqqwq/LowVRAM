"""P1 VRAM monitoring result models."""

from typing import Annotated

from pydantic import Field, model_validator

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel

PositiveFloat = Annotated[float, Field(gt=0)]


class VramGPUUsage(StrictModel):
    """One GPU system/process VRAM measurement captured at one point in time."""

    index: NonNegativeInt
    name: str = Field(min_length=1)
    vram_total_mb: NonNegativeInt
    vram_used_mb: NonNegativeInt
    process_vram_used_mb: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_usage(self) -> "VramGPUUsage":
        """Reject measurements that exceed the device's reported total VRAM."""
        if self.vram_used_mb > self.vram_total_mb:
            raise ValueError("vram_used_mb must be <= vram_total_mb")
        if (
            self.process_vram_used_mb is not None
            and self.process_vram_used_mb > self.vram_total_mb
        ):
            raise ValueError("process_vram_used_mb must be <= vram_total_mb")
        return self


class VramSample(StrictModel):
    """One multi-GPU system/process VRAM sample."""

    gpus: list[VramGPUUsage] = Field(default_factory=list)
    total_vram_used_mb: NonNegativeInt
    process_vram_supported: bool = False
    total_process_vram_used_mb: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_total(self) -> "VramSample":
        """Require unique GPU indexes and exact aggregate totals."""
        indexes = [gpu.index for gpu in self.gpus]
        if len(indexes) != len(set(indexes)):
            raise ValueError("GPU indexes must be unique")

        if self.total_vram_used_mb != sum(gpu.vram_used_mb for gpu in self.gpus):
            raise ValueError("total_vram_used_mb must equal the sum of GPU usage")

        process_values = [gpu.process_vram_used_mb for gpu in self.gpus]
        if self.process_vram_supported:
            if self.total_process_vram_used_mb is None:
                raise ValueError(
                    "supported process VRAM requires total_process_vram_used_mb"
                )
            if any(value is None for value in process_values):
                raise ValueError(
                    "supported process VRAM requires a value for every measurable GPU"
                )
            expected = sum(value for value in process_values if value is not None)
            if self.total_process_vram_used_mb != expected:
                raise ValueError(
                    "total_process_vram_used_mb must equal the per-GPU process sum"
                )
        else:
            if self.total_process_vram_used_mb is not None:
                raise ValueError(
                    "unsupported process VRAM cannot include total_process_vram_used_mb"
                )
            if any(value is not None for value in process_values):
                raise ValueError(
                    "unsupported process VRAM cannot include per-GPU process values"
                )
        return self


class VramGPUResult(StrictModel):
    """Aggregated system/process VRAM measurements for one GPU."""

    index: NonNegativeInt
    name: str = Field(min_length=1)
    vram_total_mb: NonNegativeInt

    baseline_vram_mb: NonNegativeInt
    current_vram_mb: NonNegativeInt
    peak_vram_mb: NonNegativeInt
    delta_vram_mb: NonNegativeInt

    process_vram_supported: bool = False
    baseline_process_vram_mb: NonNegativeInt | None = None
    current_process_vram_mb: NonNegativeInt | None = None
    peak_process_vram_mb: NonNegativeInt | None = None
    delta_process_vram_mb: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_aggregates(self) -> "VramGPUResult":
        """Keep per-GPU system and process aggregate values consistent."""
        if self.peak_vram_mb < self.baseline_vram_mb:
            raise ValueError("peak_vram_mb must be >= baseline_vram_mb")
        if self.peak_vram_mb < self.current_vram_mb:
            raise ValueError("peak_vram_mb must be >= current_vram_mb")
        if self.delta_vram_mb != self.peak_vram_mb - self.baseline_vram_mb:
            raise ValueError("delta_vram_mb must equal peak_vram_mb - baseline_vram_mb")
        if self.peak_vram_mb > self.vram_total_mb:
            raise ValueError("peak_vram_mb must be <= vram_total_mb")

        process_values = (
            self.baseline_process_vram_mb,
            self.current_process_vram_mb,
            self.peak_process_vram_mb,
            self.delta_process_vram_mb,
        )
        if not self.process_vram_supported:
            if any(value is not None for value in process_values):
                raise ValueError(
                    "unsupported process VRAM cannot include process aggregate values"
                )
            return self

        baseline = self.baseline_process_vram_mb
        current = self.current_process_vram_mb
        peak = self.peak_process_vram_mb
        delta = self.delta_process_vram_mb
        if baseline is None or current is None or peak is None or delta is None:
            raise ValueError("supported process VRAM requires all process aggregate values")
        if peak < baseline:
            raise ValueError("peak_process_vram_mb must be >= baseline_process_vram_mb")
        if peak < current:
            raise ValueError("peak_process_vram_mb must be >= current_process_vram_mb")
        if delta != peak - baseline:
            raise ValueError(
                "delta_process_vram_mb must equal peak_process_vram_mb "
                "- baseline_process_vram_mb"
            )
        if peak > self.vram_total_mb:
            raise ValueError("peak_process_vram_mb must be <= vram_total_mb")
        return self


class VramMonitorResult(StrictModel):
    """Aggregated multi-GPU system/process VRAM for one monitor lifecycle."""

    gpus: list[VramGPUResult] = Field(default_factory=list)

    baseline_vram_mb: NonNegativeInt
    current_vram_mb: NonNegativeInt
    peak_vram_mb: NonNegativeInt
    delta_vram_mb: NonNegativeInt

    process_vram_supported: bool = False
    baseline_process_vram_mb: NonNegativeInt | None = None
    current_process_vram_mb: NonNegativeInt | None = None
    peak_process_vram_mb: NonNegativeInt | None = None
    delta_process_vram_mb: NonNegativeInt | None = None

    sample_count: PositiveInt
    sample_interval_seconds: PositiveFloat

    @model_validator(mode="after")
    def validate_aggregates(self) -> "VramMonitorResult":
        """Validate aggregate values without summing independent peaks across time."""
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

        process_values = (
            self.baseline_process_vram_mb,
            self.current_process_vram_mb,
            self.peak_process_vram_mb,
            self.delta_process_vram_mb,
        )
        if not self.process_vram_supported:
            if any(value is not None for value in process_values):
                raise ValueError(
                    "unsupported process VRAM cannot include process aggregate values"
                )
            if any(gpu.process_vram_supported for gpu in self.gpus):
                raise ValueError(
                    "unsupported aggregate process VRAM requires unsupported GPU results"
                )
            return self

        baseline_process = self.baseline_process_vram_mb
        current_process = self.current_process_vram_mb
        peak_process = self.peak_process_vram_mb
        delta_process = self.delta_process_vram_mb
        if (
            baseline_process is None
            or current_process is None
            or peak_process is None
            or delta_process is None
        ):
            raise ValueError("supported process VRAM requires all process aggregate values")

        if not all(gpu.process_vram_supported for gpu in self.gpus):
            raise ValueError("supported aggregate process VRAM requires supported GPU results")

        per_gpu_baseline = sum(
            gpu.baseline_process_vram_mb or 0 for gpu in self.gpus
        )
        per_gpu_current = sum(
            gpu.current_process_vram_mb or 0 for gpu in self.gpus
        )
        if baseline_process != per_gpu_baseline:
            raise ValueError(
                "baseline_process_vram_mb must equal the per-GPU process baseline sum"
            )
        if current_process != per_gpu_current:
            raise ValueError(
                "current_process_vram_mb must equal the per-GPU process current sum"
            )
        if peak_process < baseline_process:
            raise ValueError(
                "peak_process_vram_mb must be >= baseline_process_vram_mb"
            )
        if peak_process < current_process:
            raise ValueError("peak_process_vram_mb must be >= current_process_vram_mb")
        if delta_process != peak_process - baseline_process:
            raise ValueError(
                "delta_process_vram_mb must equal peak_process_vram_mb "
                "- baseline_process_vram_mb"
            )
        return self
