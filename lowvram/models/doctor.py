"""P1-12 Doctor Command models."""

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import StrictModel


class DoctorCheckStatus(StrEnum):
    """Severity of one environment readiness check."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


class DoctorCheckName(StrEnum):
    """Stable identifiers for Doctor checks."""

    PYTHON = "python"
    LLAMA_CPP = "llama_cpp"
    NVIDIA = "nvidia"
    CUDA = "cuda"
    PERMISSIONS = "permissions"
    DISK = "disk"
    RAM = "ram"
    MODEL = "model"


DoctorDetailValue = str | int | float | bool | None


class DoctorCheck(StrictModel):
    """One structured Doctor check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: DoctorCheckName
    status: DoctorCheckStatus
    message: str = Field(min_length=1)
    details: dict[str, DoctorDetailValue] = Field(default_factory=dict)


class DoctorResult(StrictModel):
    """Complete machine-readable Doctor result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["p1.12.0"] = "p1.12.0"
    ready: bool
    summary: Literal["READY", "BLOCKED"]
    checks: tuple[DoctorCheck, ...]
    blocking_checks: tuple[DoctorCheckName, ...]

    @model_validator(mode="after")
    def validate_summary(self) -> "DoctorResult":
        """Require summary/readiness/blocking list to agree with check statuses."""
        expected_blockers = tuple(
            check.name
            for check in self.checks
            if check.status == DoctorCheckStatus.BLOCK
        )
        if self.blocking_checks != expected_blockers:
            raise ValueError("blocking_checks must match block-status checks in order")

        expected_ready = not expected_blockers
        if self.ready != expected_ready:
            raise ValueError("ready must be true exactly when no blocking checks exist")
        expected_summary = "READY" if expected_ready else "BLOCKED"
        if self.summary != expected_summary:
            raise ValueError("summary must match readiness")
        return self
