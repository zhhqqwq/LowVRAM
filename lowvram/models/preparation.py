"""P1-13 shared benchmark preparation models."""

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel
from lowvram.models.benchmark import ErrorType
from lowvram.models.command import LlamaCppCommand
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.failure import FailureStage
from lowvram.models.prompt import PromptVersion

FiniteNonNegativeFloat = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]


class BenchmarkPreparationRequest(StrictModel):
    """Inputs needed before any benchmark model process may be spawned."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_path: str = Field(min_length=1)
    llama_cli: str | None = Field(default=None, min_length=1)
    context_length: PositiveInt
    threads: PositiveInt
    gpu_layers: NonNegativeInt
    batch_size: PositiveInt
    temperature: FiniteNonNegativeFloat
    seed: int
    prompt_version: PromptVersion = "v1"
    extra_args: tuple[str, ...] = Field(default_factory=tuple)


class BenchmarkPreparationResult(StrictModel):
    """Structured output of the shared P1 benchmark pre-spawn path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ready: bool
    detection: LlamaCppDetectionResult | None = None
    prompt_version: PromptVersion
    prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    command: LlamaCppCommand | None = None
    failure_stage: FailureStage | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "BenchmarkPreparationResult":
        """Require complete command evidence for ready results and errors for blockers."""
        if self.ready:
            if self.detection is None or not self.detection.verified_eligible:
                raise ValueError("ready preparation requires verified runtime detection")
            if self.prompt_sha256 is None or self.command is None:
                raise ValueError("ready preparation requires prompt hash and command")
            if any(
                value is not None
                for value in (self.failure_stage, self.error_type, self.error_message)
            ):
                raise ValueError("ready preparation cannot include failure fields")
            return self

        if self.failure_stage is None or self.error_type is None or not self.error_message:
            raise ValueError("blocked preparation requires stage, error_type, and message")
        if self.command is not None:
            raise ValueError("blocked preparation cannot include an executable command")
        return self
