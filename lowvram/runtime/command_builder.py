"""P1-07 deterministic llama.cpp command construction."""

from lowvram.models.command import LlamaCppCommand, LlamaCppCommandRequest


def build_llama_cpp_command(request: LlamaCppCommandRequest) -> LlamaCppCommand:
    """Map normalized P1 command inputs to deterministic llama.cpp argv."""
    arguments = [
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
    ]
    return LlamaCppCommand(
        executable=request.executable,
        arguments=arguments,
        argv=[request.executable, *arguments],
    )
