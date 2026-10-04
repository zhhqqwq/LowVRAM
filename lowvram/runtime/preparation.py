"""P1-13 shared pre-spawn benchmark preparation."""

from pathlib import Path

from pydantic import ValidationError

from lowvram.models.benchmark import ErrorType
from lowvram.models.command import LlamaCppCommandRequest
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.failure import FailureStage
from lowvram.models.preparation import (
    BenchmarkPreparationRequest,
    BenchmarkPreparationResult,
)
from lowvram.prompts import BenchmarkPromptError, load_benchmark_prompt
from lowvram.runtime.adapter import RuntimeAdapter
from lowvram.runtime.command_builder import build_llama_cpp_command
from lowvram.runtime.detector import detect_llama_cpp
from lowvram.runtime.llama_cpp import LlamaCppRuntimeAdapter

_PROMPT_OVERRIDE_FLAGS = ("-p", "--prompt", "-f", "--file", "--prompt-file")


def prepare_llama_cpp_benchmark(
    request: BenchmarkPreparationRequest,
    *,
    adapter: RuntimeAdapter | None = None,
) -> BenchmarkPreparationResult:
    """Prepare the exact benchmark command without spawning a model process."""
    if not Path(request.model_path).is_file():
        return _blocked(
            request,
            stage=FailureStage.PREFLIGHT,
            error_type=ErrorType.MODEL_NOT_FOUND,
            message=f"model file not found: {request.model_path}",
        )

    runtime_adapter = adapter or LlamaCppRuntimeAdapter()
    detection = detect_llama_cpp(request.llama_cli, adapter=runtime_adapter)

    if not detection.found:
        return _blocked(
            request,
            detection=detection,
            stage=FailureStage.RUNTIME_START,
            error_type=detection.probe_error_type or ErrorType.RUNTIME_NOT_FOUND,
            message=detection.message or "llama.cpp runtime was not found",
        )

    if not detection.runnable:
        return _blocked(
            request,
            detection=detection,
            stage=FailureStage.RUNTIME_START,
            error_type=detection.probe_error_type or ErrorType.UNKNOWN,
            message=detection.message or "llama.cpp runtime is not runnable",
        )

    if not detection.verified_eligible:
        return _blocked(
            request,
            detection=detection,
            stage=FailureStage.RUNTIME_START,
            error_type=ErrorType.UNKNOWN,
            message=(
                detection.message
                or "llama.cpp version is not recognized; verified benchmarks are blocked"
            ),
        )

    executable = detection.executable
    if executable is None:
        return _blocked(
            request,
            detection=detection,
            stage=FailureStage.INTERNAL,
            error_type=ErrorType.UNKNOWN,
            message="llama.cpp detection returned no executable path",
        )

    try:
        prompt = load_benchmark_prompt(request.prompt_version)
    except BenchmarkPromptError as exc:
        return _blocked(
            request,
            detection=detection,
            stage=FailureStage.INTERNAL,
            error_type=ErrorType.UNKNOWN,
            message=str(exc),
        )

    prompt_override = _find_prompt_override(request.extra_args)
    if prompt_override is not None:
        return _blocked(
            request,
            detection=detection,
            stage=FailureStage.PREFLIGHT,
            error_type=ErrorType.UNKNOWN,
            message=(
                "configuration.extra_args cannot override Standard Prompt: "
                f"{prompt_override}"
            ),
        )

    try:
        command = build_llama_cpp_command(
            LlamaCppCommandRequest(
                executable=executable,
                model_path=request.model_path,
                context_length=request.context_length,
                threads=request.threads,
                gpu_layers=request.gpu_layers,
                batch_size=request.batch_size,
                temperature=request.temperature,
                seed=request.seed,
                extra_args=(
                    *request.extra_args,
                    "--prompt",
                    prompt.content,
                ),
            )
        )
    except ValidationError as exc:
        return _blocked(
            request,
            detection=detection,
            stage=FailureStage.INTERNAL,
            error_type=ErrorType.UNKNOWN,
            message=f"command construction failed: {exc}",
        )

    return BenchmarkPreparationResult(
        ready=True,
        detection=detection,
        prompt_version=request.prompt_version,
        prompt_sha256=prompt.sha256,
        command=command,
    )


def _blocked(
    request: BenchmarkPreparationRequest,
    *,
    stage: FailureStage,
    error_type: ErrorType,
    message: str,
    detection: LlamaCppDetectionResult | None = None,
) -> BenchmarkPreparationResult:
    return BenchmarkPreparationResult(
        ready=False,
        detection=detection,
        prompt_version=request.prompt_version,
        failure_stage=stage,
        error_type=error_type,
        error_message=message,
    )


def _find_prompt_override(arguments: tuple[str, ...]) -> str | None:
    for argument in arguments:
        for flag in _PROMPT_OVERRIDE_FLAGS:
            if argument == flag or argument.startswith(f"{flag}="):
                return flag
    return None
