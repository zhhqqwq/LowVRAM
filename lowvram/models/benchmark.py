"""Benchmark-run data contract."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from lowvram.models.base import NonNegativeFloat, NonNegativeInt, StrictModel
from lowvram.models.hardware import HardwareInfo
from lowvram.models.model import ModelInfo
from lowvram.models.prompt import PromptVersion
from lowvram.models.recipe import Recipe


class RuntimeInfo(StrictModel):
    """Runtime identity recorded with a benchmark."""

    name: Literal["llama.cpp"]
    version: str | None = None


class MemoryMetrics(StrictModel):
    """Peak memory measurements, expressed only in MB."""

    peak_vram_mb: NonNegativeInt | None = None
    peak_ram_mb: NonNegativeInt | None = None


class PerformanceMetrics(StrictModel):
    """Timing and throughput metrics with fixed units."""

    load_time_seconds: NonNegativeFloat | None = None
    prompt_tokens_per_second: NonNegativeFloat | None = None
    generation_tokens_per_second: NonNegativeFloat | None = None


class ErrorType(StrEnum):
    """Stable P0 failure categories; later phases may add runner-specific handling."""

    RUNTIME_NOT_FOUND = "runtime_not_found"
    MODEL_NOT_FOUND = "model_not_found"
    MODEL_LOAD_FAILED = "model_load_failed"
    OUT_OF_MEMORY = "out_of_memory"
    PROCESS_CRASH = "process_crash"
    TIMEOUT = "timeout"
    PARSE_FAILED = "parse_failed"
    UNKNOWN = "unknown"


class BenchmarkResult(StrictModel):
    """Outcome of a benchmark attempt."""

    success: bool
    error_type: ErrorType | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_error_state(self) -> "BenchmarkResult":
        """Keep success and failure payloads internally consistent."""
        if self.success and (self.error_type is not None or self.error_message is not None):
            raise ValueError("successful results cannot include error fields")
        if not self.success and self.error_type is None:
            raise ValueError("failed results require error_type")
        return self


class VerificationInfo(StrictModel):
    """Provenance status for a benchmark record."""

    status: Literal["unverified", "verified"] = "unverified"
    generated_by: Literal["lowvram"] = "lowvram"
    evidence: list[str] = Field(default_factory=list)
    notes: str | None = None


class BenchmarkRun(StrictModel):
    """Self-contained benchmark record supporting both success and failure runs."""

    schema_version: Literal["1.1.0"] = "1.1.0"
    run_id: str = Field(min_length=1)
    timestamp: datetime
    hardware: HardwareInfo
    model: ModelInfo
    runtime: RuntimeInfo
    prompt_version: PromptVersion
    configuration: Recipe
    memory: MemoryMetrics
    performance: PerformanceMetrics
    result: BenchmarkResult
    verification: VerificationInfo = Field(default_factory=VerificationInfo)

    @model_validator(mode="after")
    def validate_cross_field_contract(self) -> "BenchmarkRun":
        """Require observable success metrics and consistent model/runtime references."""
        if self.configuration.model_id != self.model.id:
            raise ValueError("configuration.model_id must match model.id")
        if self.configuration.runtime != self.runtime.name:
            raise ValueError("configuration.runtime must match runtime.name")

        if self.result.success:
            required_metrics = {
                "memory.peak_vram_mb": self.memory.peak_vram_mb,
                "memory.peak_ram_mb": self.memory.peak_ram_mb,
                "performance.prompt_tokens_per_second": self.performance.prompt_tokens_per_second,
                "performance.generation_tokens_per_second": (
                    self.performance.generation_tokens_per_second
                ),
            }
            missing = [name for name, value in required_metrics.items() if value is None]
            if missing:
                raise ValueError("successful runs require metrics: " + ", ".join(missing))
        return self
