"""P1-10 Benchmark Orchestrator request and record models."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import StrictModel
from lowvram.models.benchmark import BenchmarkResult
from lowvram.models.command import LlamaCppCommand
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.hardware import HardwareInfo
from lowvram.models.memory import RamMonitorResult
from lowvram.models.model import ModelInfo
from lowvram.models.output_parser import LlamaCppTimingMetrics
from lowvram.models.prompt import PromptVersion
from lowvram.models.recipe import Recipe
from lowvram.models.runtime import RuntimeExecutionResult
from lowvram.models.vram import VramMonitorResult

PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FiniteNonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class BenchmarkOrchestratorRequest(StrictModel):
    """Inputs needed to run one P1-10 llama.cpp benchmark attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(min_length=1)
    model: ModelInfo
    configuration: Recipe
    llama_cli: str | None = Field(default=None, min_length=1)
    prompt_version: PromptVersion = "v1"
    temperature: FiniteNonNegativeFloat = 0.0
    seed: int = 42
    timeout_seconds: PositiveFiniteFloat = 300.0

    @model_validator(mode="after")
    def validate_model_reference(self) -> "BenchmarkOrchestratorRequest":
        """Require the P0 recipe to target the same model metadata."""
        if self.configuration.model_id != self.model.id:
            raise ValueError("configuration.model_id must match model.id")
        return self


class BenchmarkOrchestrationRecord(StrictModel):
    """Serializable evidence record produced by one P1-10 orchestration attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["p1.10.0"] = "p1.10.0"
    run_id: str = Field(min_length=1)
    timestamp: datetime
    hardware: HardwareInfo | None = None
    model: ModelInfo
    configuration: Recipe

    detection: LlamaCppDetectionResult | None = None
    prompt_version: PromptVersion
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    command: LlamaCppCommand | None = None

    execution: RuntimeExecutionResult | None = None
    ram: RamMonitorResult | None = None
    vram: VramMonitorResult | None = None
    performance: LlamaCppTimingMetrics | None = None

    result: BenchmarkResult

    @model_validator(mode="after")
    def validate_success_evidence(self) -> "BenchmarkOrchestrationRecord":
        """Require complete evidence for successful orchestration records."""
        if not self.result.success:
            return self

        required = {
            "hardware": self.hardware,
            "detection": self.detection,
            "prompt_sha256": self.prompt_sha256,
            "command": self.command,
            "execution": self.execution,
            "ram": self.ram,
            "vram": self.vram,
            "performance": self.performance,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "successful orchestration requires evidence: " + ", ".join(missing)
            )

        if self.detection is not None and not self.detection.runnable:
            raise ValueError("successful orchestration requires a runnable runtime")
        if self.execution is not None and not self.execution.success:
            raise ValueError("successful orchestration requires successful execution")
        return self
