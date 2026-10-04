"""P1-13 non-executing benchmark preview."""

from lowvram.models.benchmark import ErrorType, RuntimeInfo
from lowvram.models.dry_run import (
    BenchmarkDryRunConfiguration,
    BenchmarkDryRunResult,
)
from lowvram.models.preparation import BenchmarkPreparationRequest
from lowvram.runtime.adapter import RuntimeAdapter
from lowvram.runtime.preparation import prepare_llama_cpp_benchmark


def build_benchmark_dry_run(
    request: BenchmarkPreparationRequest,
    *,
    adapter: RuntimeAdapter | None = None,
) -> BenchmarkDryRunResult:
    """Build the same pre-spawn command used by a real run, without spawning it."""
    prepared = prepare_llama_cpp_benchmark(request, adapter=adapter)
    runtime_version = (
        prepared.detection.version if prepared.detection is not None else None
    )
    configuration = BenchmarkDryRunConfiguration(
        context_length=request.context_length,
        threads=request.threads,
        gpu_layers=request.gpu_layers,
        batch_size=request.batch_size,
        temperature=request.temperature,
        seed=request.seed,
    )

    if prepared.ready:
        return BenchmarkDryRunResult(
            ready=True,
            runtime=RuntimeInfo(name="llama.cpp", version=runtime_version),
            model=request.model_path,
            configuration=configuration,
            prompt_version=request.prompt_version,
            command=prepared.command,
        )

    return BenchmarkDryRunResult(
        ready=False,
        runtime=RuntimeInfo(name="llama.cpp", version=runtime_version),
        model=request.model_path,
        configuration=configuration,
        prompt_version=request.prompt_version,
        error_type=prepared.error_type or ErrorType.UNKNOWN,
        error_message=prepared.error_message or "benchmark preparation failed",
    )
