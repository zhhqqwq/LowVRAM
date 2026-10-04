"""P1-11 structured failure-classification models."""

from enum import StrEnum

from pydantic import ConfigDict, Field

from lowvram.models.base import StrictModel
from lowvram.models.benchmark import ErrorType
from lowvram.models.runtime import RuntimeExecutionResult


class FailureStage(StrEnum):
    """Lifecycle stage at which a benchmark attempt failed."""

    PREFLIGHT = "preflight"
    RUNTIME_START = "runtime_start"
    RUNTIME_EXECUTION = "runtime_execution"
    OUTPUT_PARSE = "output_parse"
    INTERNAL = "internal"


class FailureEvidenceSource(StrEnum):
    """Non-sensitive source category used by a classification rule."""

    STRUCTURED = "structured"
    STDERR = "stderr"
    STDOUT = "stdout"
    MESSAGE = "message"


class FailureClassificationRequest(StrictModel):
    """Evidence supplied to the P1-11 classifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: FailureStage
    model_path_exists: bool | None = None
    execution: RuntimeExecutionResult | None = None
    upstream_error_type: ErrorType | None = None
    message: str | None = None


class FailureClassificationResult(StrictModel):
    """Stable error type plus the rule and source categories that justified it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    error_type: ErrorType
    rule: str = Field(min_length=1)
    message: str = Field(min_length=1)
    evidence_sources: tuple[FailureEvidenceSource, ...] = Field(default_factory=tuple)
