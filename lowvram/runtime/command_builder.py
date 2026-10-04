"""P1-07 deterministic llama.cpp command construction."""

from lowvram.models.command import LlamaCppCommand, LlamaCppCommandRequest
from lowvram.models.runtime import RuntimeExecutionRequest


def build_llama_cpp_command(request: LlamaCppCommandRequest) -> LlamaCppCommand:
    """Map normalized P1 command inputs to deterministic llama.cpp argv."""
    arguments = (
        "--model",
        request.model_path,
        "--ctx-size",
        str(request.context_length),
        "--threads",
        str(request.threads),
        "--gpu-layers",
        str(request.gpu_layers),
        "--batch-size",
        str(request.batch_size),
        "--temp",
        str(request.temperature),
        "--seed",
        str(request.seed),
        *request.extra_args,
    )
    return LlamaCppCommand(
        executable=request.executable,
        arguments=arguments,
        argv=(request.executable, *arguments),
    )


def to_runtime_execution_request(
    command: LlamaCppCommand,
    *,
    timeout_seconds: float = 300.0,
) -> RuntimeExecutionRequest:
    """Convert the immutable command record into the P1-05 execution request."""
    return RuntimeExecutionRequest(
        executable=command.executable,
        arguments=list(command.arguments),
        timeout_seconds=timeout_seconds,
    )
