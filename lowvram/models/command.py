"""P1 llama.cpp command-building models."""

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from lowvram.models.base import NonNegativeInt, PositiveInt, StrictModel

FiniteNonNegativeFloat = Annotated[
    float,
    Field(ge=0, allow_inf_nan=False),
]

_MANAGED_LLAMA_CPP_FLAGS = (
    "-m",
    "--model",
    "-c",
    "--ctx-size",
    "-t",
    "--threads",
    "-ngl",
    "--gpu-layers",
    "--n-gpu-layers",
    "-b",
    "--batch-size",
    "--temp",
    "--temperature",
    "-s",
    "--seed",
)


def _managed_extra_arg(argument: str) -> str | None:
    """Return the managed flag targeted by one extra argv item, if any."""
    for flag in _MANAGED_LLAMA_CPP_FLAGS:
        if argument == flag or argument.startswith(f"{flag}="):
            return flag
    return None


class LlamaCppCommandRequest(StrictModel):
    """Inputs required to build one deterministic llama.cpp command."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    executable: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    context_length: PositiveInt
    threads: PositiveInt
    gpu_layers: NonNegativeInt
    batch_size: PositiveInt
    temperature: FiniteNonNegativeFloat
    seed: int
    extra_args: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_command_input(self) -> "LlamaCppCommandRequest":
        """Reject invalid OS argv values and managed-parameter overrides."""
        if "\x00" in self.executable:
            raise ValueError("executable cannot contain a NUL byte")
        if "\x00" in self.model_path:
            raise ValueError("model_path cannot contain a NUL byte")

        for argument in self.extra_args:
            if "\x00" in argument:
                raise ValueError("extra_args cannot contain a NUL byte")
            managed_flag = _managed_extra_arg(argument)
            if managed_flag is not None:
                raise ValueError(
                    f"extra_args cannot override managed llama.cpp flag: {managed_flag}"
                )
        return self


class LlamaCppCommand(StrictModel):
    """Immutable exact argv generated for one llama.cpp invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executable: str = Field(min_length=1)
    arguments: tuple[str, ...]
    argv: tuple[str, ...]

    @model_validator(mode="after")
    def validate_argv(self) -> "LlamaCppCommand":
        """Require the recorded full command to match executable plus arguments."""
        expected = (self.executable, *self.arguments)
        if self.argv != expected:
            raise ValueError("argv must equal (executable, *arguments)")
        return self
