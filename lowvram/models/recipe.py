"""Runtime recipe data contract."""

from typing import Literal

from pydantic import Field

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel


class Recipe(StrictModel):
    """A reproducible llama.cpp configuration without executing it."""

    model_id: str = Field(min_length=1)
    runtime: Literal["llama.cpp"]
    context_length: PositiveInt
    gpu_layers: NonNegativeInt
    kv_cache_type: str | None = None
    threads: PositiveInt
    batch_size: PositiveInt
    extra_args: list[str] = Field(default_factory=list)
