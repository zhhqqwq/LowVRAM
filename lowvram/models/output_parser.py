"""P1 llama.cpp timing parser models."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import StrictModel
from lowvram.models.benchmark import ErrorType

PositiveFiniteFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class LlamaCppTimingMetrics(StrictModel):
    """Timing metrics parsed directly from one supported llama.cpp timing format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timing_format: Literal["detailed", "compact"] = "detailed"
    load_time_seconds: PositiveFiniteFloat | None = None
    prompt_eval_time_seconds: PositiveFiniteFloat | None = None
    prompt_tokens_per_second: PositiveFiniteFloat
    eval_time_seconds: PositiveFiniteFloat | None = None
    generation_tokens_per_second: PositiveFiniteFloat

    @model_validator(mode="after")
    def validate_timing_format(self) -> "LlamaCppTimingMetrics":
        """Do not fabricate duration fields that a timing format does not expose."""
        if self.timing_format == "detailed":
            if self.prompt_eval_time_seconds is None or self.eval_time_seconds is None:
                raise ValueError(
                    "detailed timing format requires prompt_eval_time_seconds "
                    "and eval_time_seconds"
                )
            return self

        if self.prompt_eval_time_seconds is not None or self.eval_time_seconds is not None:
            raise ValueError(
                "compact timing format cannot include unreported prompt/eval durations"
            )
        return self


class LlamaCppOutputParseResult(StrictModel):
    """Structured success/failure result for one llama.cpp output parse."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    metrics: LlamaCppTimingMetrics | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> "LlamaCppOutputParseResult":
        """Keep parser success and parse_failed states internally consistent."""
        if self.success:
            if self.metrics is None:
                raise ValueError("successful parse requires metrics")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful parse cannot include error fields")
            return self

        if self.metrics is not None:
            raise ValueError("failed parse cannot include metrics")
        if self.error_type != ErrorType.PARSE_FAILED:
            raise ValueError("failed parse requires parse_failed")
        if not self.error_message:
            raise ValueError("failed parse requires error_message")
        return self
