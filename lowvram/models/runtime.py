"""P1 runtime execution models."""

from typing import Annotated

from pydantic import Field, model_validator

from lowvram.models.base import NonNegativeFloat, StrictModel
from lowvram.models.benchmark import ErrorType

PositiveFloat = Annotated[float, Field(gt=0)]


class RuntimeExecutionRequest(StrictModel):
    """One request to execute a specific runtime executable."""

    executable: str = Field(min_length=1)
    arguments: list[str] = Field(default_factory=list)
    timeout_seconds: PositiveFloat = 300.0


class RuntimeExecutionResult(StrictModel):
    """Structured outcome of one external runtime process."""

    runtime_name: str = Field(min_length=1)
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: NonNegativeFloat
    error_type: ErrorType | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "RuntimeExecutionResult":
        """Keep success, exit code, and error classification consistent."""
        if self.success:
            if self.exit_code != 0:
                raise ValueError("successful runtime execution requires exit_code=0")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful runtime execution cannot include error fields")
            return self

        if self.error_type is None:
            raise ValueError("failed runtime execution requires error_type")
        if self.error_type in {ErrorType.RUNTIME_NOT_FOUND, ErrorType.TIMEOUT}:
            if self.exit_code is not None:
                raise ValueError(
                    "runtime_not_found and timeout results must not report an exit code"
                )
        elif self.error_type == ErrorType.PROCESS_CRASH and self.exit_code is None:
            raise ValueError("process_crash results require an exit code")
        return self
