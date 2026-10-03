"""Model metadata data contract."""

from typing import Literal

from pydantic import Field, model_validator

from lowvram.models.base import PositiveInt, StrictModel


class ModelInfo(StrictModel):
    """Metadata needed to identify a supported local LLM artifact."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    source: str = Field(min_length=1)
    architecture: Literal["dense", "moe", "unknown"]
    parameter_count: PositiveInt
    active_parameter_count: PositiveInt | None = None
    model_type: Literal["llm"] = "llm"
    format: Literal["gguf"] = "gguf"
    quantization: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_active_parameters(self) -> "ModelInfo":
        """Require active parameter count for MoE and keep it within total parameters."""
        if self.architecture == "moe" and self.active_parameter_count is None:
            raise ValueError("MoE models require active_parameter_count")
        if (
            self.active_parameter_count is not None
            and self.active_parameter_count > self.parameter_count
        ):
            raise ValueError("active_parameter_count must be <= parameter_count")
        return self
