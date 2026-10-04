"""P1 standard benchmark prompt models."""

import hashlib
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import StrictModel

PromptVersion = Literal["v1"]


class BenchmarkPrompt(StrictModel):
    """One immutable, versioned benchmark prompt loaded from trusted bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_version: PromptVersion
    content: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content_hash(self) -> "BenchmarkPrompt":
        """Require the recorded digest to match the exact UTF-8 prompt content."""
        digest = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if digest != self.sha256:
            raise ValueError("sha256 does not match benchmark prompt content")
        return self
