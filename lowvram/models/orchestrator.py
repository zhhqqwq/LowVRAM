"""P1 Benchmark Orchestrator request and evidence-record models."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import StrictModel
from lowvram.models.benchmark import BenchmarkResult
from lowvram.models.command import LlamaCppCommand
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.failure import FailureClassificationResult
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
    """Inputs needed to run one llama.cpp benchmark attempt."""

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
    """Serializable P1 evidence record for one orchestration attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["p1.11.0"] = "p1.11.0"
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

    failure_classification: FailureClassificationResult | None = None
    result: BenchmarkResult

    @model_validator(mode="after")
    def validate_evidence(self) -> "BenchmarkOrchestrationRecord":
        """Require success evidence or a matching structured failure classification."""
        if not self.result.success:
            if self.failure_classification is None:
                raise ValueError("failed orchestration requires failure_classification")
            if self.result.error_type != self.failure_classification.error_type:
                raise ValueError(
                    "result.error_type must match failure_classification.error_type"
                )
            if self.result.error_message != self.failure_classification.message:
                raise ValueError(
                    "result.error_message must match failure_classification.message"
                )
            return self

        if self.failure_classification is not None:
            raise ValueError("successful orchestration cannot include failure_classification")

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
