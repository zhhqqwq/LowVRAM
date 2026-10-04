"""P1 llama.cpp command-building models."""

from pydantic import Field, model_validator

from lowvram.models.base import NonNegativeFloat, NonNegativeInt, PositiveInt, StrictModel

_MANAGED_LLAMA_CPP_FLAGS = frozenset(
    {
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
    }
)


def _managed_extra_arg(argument: str) -> str | None:
    """Return the managed flag targeted by one extra argv item, if any."""
    for flag in _MANAGED_LLAMA_CPP_FLAGS:
        if argument == flag or argument.startswith(f"{flag}="):
            return flag
    return None


class LlamaCppCommandRequest(StrictModel):
    """Inputs required to build one deterministic llama.cpp command."""

    executable: str = Field(min_length=1)
    model_path: str = Field(min_length=1)
    context_length: PositiveInt
    threads: PositiveInt
    gpu_layers: NonNegativeInt
    batch_size: PositiveInt
    temperature: NonNegativeFloat
    seed: int
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_extra_args(self) -> "LlamaCppCommandRequest":
        """Reject extra args that would override parameters owned by this model."""
        for argument in self.extra_args:
            managed_flag = _managed_extra_arg(argument)
            if managed_flag is not None:
                raise ValueError(
                    f"extra_args cannot override managed llama.cpp flag: {managed_flag}"
                )
        return self


class LlamaCppCommand(StrictModel):
    """Exact argv generated for one llama.cpp invocation."""

    executable: str = Field(min_length=1)
    arguments: list[str]
    argv: list[str]

    @model_validator(mode="after")
    def validate_argv(self) -> "LlamaCppCommand":
        """Require the recorded full command to match executable plus arguments."""
        expected = [self.executable, *self.arguments]
        if self.argv != expected:
            raise ValueError("argv must equal [executable, *arguments]")
        return self
