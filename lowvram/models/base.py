"""Shared model configuration and constrained scalar aliases."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields instead of silently ignoring them."""

    model_config = ConfigDict(extra="forbid")
