"""P1 runtime detection models."""

from typing import Literal

from pydantic import Field, model_validator

from lowvram.models.base import StrictModel
from lowvram.models.benchmark import ErrorType


class LlamaCppDetectionResult(StrictModel):
    """Structured result of one llama.cpp discovery attempt."""

    found: bool
    executable: str | None = None
    source: Literal["explicit_path", "path"] | None = None
    candidate_name: str | None = None
    runnable: bool = False
    version: str | None = None
    version_command: list[str] | None = None
    verified_eligible: bool = False
    probe_error_type: ErrorType | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_detection_state(self) -> "LlamaCppDetectionResult":
        """Keep discovery, probe, and verification state internally consistent."""
        if not self.found:
            if any(
                value is not None
                for value in (
                    self.executable,
                    self.source,
                    self.candidate_name,
                    self.version,
                    self.version_command,
                )
            ):
                raise ValueError("not-found results cannot include executable details")
            if self.runnable or self.verified_eligible:
                raise ValueError("not-found results cannot be runnable or verified")
            if self.probe_error_type != ErrorType.RUNTIME_NOT_FOUND:
                raise ValueError("not-found results require runtime_not_found")
            return self

        if self.executable is None or self.source is None or self.candidate_name is None:
            raise ValueError("found results require executable, source, and candidate_name")

        if self.version is not None and self.version_command is None:
            raise ValueError("recognized versions require version_command")

        expected_verified = self.runnable and self.version is not None
        if self.verified_eligible != expected_verified:
            raise ValueError(
                "verified_eligible must equal runnable and recognized version state"
            )
        return self
