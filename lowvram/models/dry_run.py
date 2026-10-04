"""P1-13 Benchmark Dry Run models."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel
from lowvram.models.benchmark import ErrorType, RuntimeInfo
from lowvram.models.command import LlamaCppCommand
from lowvram.models.prompt import PromptVersion

FiniteNonNegativeFloat = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]


class BenchmarkDryRunConfiguration(StrictModel):
    """Normalized configuration displayed by benchmark --dry-run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_length: PositiveInt
    threads: PositiveInt
    gpu_layers: NonNegativeInt
    batch_size: PositiveInt
    temperature: FiniteNonNegativeFloat
    seed: int


class BenchmarkDryRunResult(StrictModel):
    """Machine-readable P1-13 preview of one benchmark command."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["p1.13.0"] = "p1.13.0"
    ready: bool
    runtime: RuntimeInfo
    model: str = Field(min_length=1)
    configuration: BenchmarkDryRunConfiguration
    prompt_version: PromptVersion
    command: LlamaCppCommand | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> "BenchmarkDryRunResult":
        """Keep READY/BLOCKED dry-run output internally consistent."""
        if self.ready:
            if self.runtime.version is None:
                raise ValueError("ready dry-run requires recognized runtime version")
            if self.command is None:
                raise ValueError("ready dry-run requires exact command")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("ready dry-run cannot include error fields")
            return self

        if self.command is not None:
            raise ValueError("blocked dry-run cannot include exact command")
        if self.error_type is None or not self.error_message:
            raise ValueError("blocked dry-run requires error_type and error_message")
        return self
