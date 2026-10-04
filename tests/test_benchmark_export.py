"""P1-15 BenchmarkRun export tests."""

from datetime import UTC, datetime

import pytest

from lowvram.benchmark_export import BenchmarkExportError, build_benchmark_run
from lowvram.models.benchmark import BenchmarkResult, ErrorType
from lowvram.models.command import LlamaCppCommand
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.failure import FailureClassificationResult, FailureEvidenceSource
from lowvram.models.hardware import CPUInfo, DriverInfo, HardwareInfo, OSInfo
from lowvram.models.memory import RamMonitorResult
from lowvram.models.model import ModelInfo
from lowvram.models.orchestrator import BenchmarkOrchestrationRecord
from lowvram.models.output_parser import LlamaCppTimingMetrics
from lowvram.models.recipe import Recipe
from lowvram.models.runtime import RuntimeExecutionResult
from lowvram.models.vram import VramMonitorResult


def _hardware() -> HardwareInfo:
    return HardwareInfo(
        cpu=CPUInfo(
            name="Example CPU",
            architecture="x86_64",
            physical_cores=4,
            logical_cores=4,
        ),
        gpu=[],
        ram_total_mb=8192,
        os=OSInfo(name="Linux", version="6.8", architecture="x86_64"),
        driver=DriverInfo(),
        python_version="3.11.9",
    )


def _model() -> ModelInfo:
    return ModelInfo(
        id="seed-model",
        name="Seed Model",
        source="https://example.invalid/seed-model",
        architecture="dense",
        parameter_count=135_000_000,
        quantization="Q4_K_M",
    )


def _recipe() -> Recipe:
    return Recipe(
        model_id="seed-model",
        runtime="llama.cpp",
        context_length=1024,
        gpu_layers=0,
        threads=4,
        batch_size=128,
        extra_args=["--n-predict", "64"],
    )


def _detection() -> LlamaCppDetectionResult:
    return LlamaCppDetectionResult(
        found=True,
        executable="/private/llama-cli",
        source="explicit_path",
        candidate_name="llama-cli",
        runnable=True,
        version="b10336",
        version_command=["/private/llama-cli", "--version"],
        verified_eligible=True,
        message="detected",
    )


def _ram() -> RamMonitorResult:
    return RamMonitorResult(
        baseline_process_ram_mb=10,
        current_process_ram_mb=20,
        peak_process_ram_mb=256,
        delta_process_ram_mb=246,
        baseline_system_ram_mb=1000,
        current_system_ram_mb=1100,
        peak_system_ram_mb=1300,
        delta_system_ram_mb=300,
        sample_count=4,
        sample_interval_seconds=0.1,
    )


def _vram() -> VramMonitorResult:
    return VramMonitorResult(
        gpus=[],
        baseline_vram_mb=0,
        current_vram_mb=0,
        peak_vram_mb=0,
        delta_vram_mb=0,
        process_vram_supported=False,
        sample_count=1,
        sample_interval_seconds=0.1,
    )


def _success_record(
    *,
    load_time_seconds: float | None = 0.25,
) -> BenchmarkOrchestrationRecord:
    command = LlamaCppCommand(
        executable="/private/llama-cli",
        arguments=("--model", "/private/model.gguf", "--ctx-size", "1024"),
        argv=(
            "/private/llama-cli",
            "--model",
            "/private/model.gguf",
            "--ctx-size",
            "1024",
        ),
    )
    execution = RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=True,
        exit_code=0,
        stdout="generated",
        stderr="timings",
        duration_seconds=1.0,
    )
    return BenchmarkOrchestrationRecord(
        run_id="seed-success",
        timestamp=datetime(2026, 10, 5, tzinfo=UTC),
        hardware=_hardware(),
        model=_model(),
        configuration=_recipe(),
        detection=_detection(),
        prompt_version="v1",
        prompt_sha256="a" * 64,
        command=command,
        execution=execution,
        ram=_ram(),
        vram=_vram(),
        performance=LlamaCppTimingMetrics(
            load_time_seconds=load_time_seconds,
            prompt_eval_time_seconds=0.1,
            prompt_tokens_per_second=100.0,
            eval_time_seconds=0.5,
            generation_tokens_per_second=50.0,
        ),
        result=BenchmarkResult(success=True),
    )


def test_success_export_uses_direct_load_time_and_process_memory() -> None:
    benchmark = build_benchmark_run(_success_record())

    assert benchmark.schema_version == "1.1.0"
    assert benchmark.prompt_version == "v1"
    assert benchmark.memory.peak_ram_mb == 256
    assert benchmark.memory.peak_vram_mb == 0
    assert benchmark.performance.load_time_seconds == 0.25
    assert benchmark.performance.prompt_tokens_per_second == 100.0
    assert benchmark.performance.generation_tokens_per_second == 50.0
    assert benchmark.verification.status == "verified"


def test_export_omits_private_runtime_and_model_paths() -> None:
    payload = build_benchmark_run(_success_record()).model_dump_json()

    assert "/private/llama-cli" not in payload
    assert "/private/model.gguf" not in payload


def test_success_export_preserves_missing_direct_load_time_as_null() -> None:
    benchmark = build_benchmark_run(_success_record(load_time_seconds=None))

    assert benchmark.performance.load_time_seconds is None
    assert benchmark.verification.notes is not None
    assert "did not directly expose model load time" in benchmark.verification.notes


def test_execution_failure_can_export_null_performance() -> None:
    record = _success_record().model_copy(
        update={
            "performance": None,
            "result": BenchmarkResult(
                success=False,
                error_type=ErrorType.TIMEOUT,
                error_message="runtime exceeded timeout",
            ),
            "failure_classification": FailureClassificationResult(
                error_type=ErrorType.TIMEOUT,
                rule="timeout.structured",
                message="runtime exceeded timeout",
                evidence_sources=(FailureEvidenceSource.STRUCTURED,),
            ),
        }
    )

    benchmark = build_benchmark_run(record)

    assert benchmark.result.success is False
    assert benchmark.result.error_type == ErrorType.TIMEOUT
    assert benchmark.performance.load_time_seconds is None
    assert benchmark.verification.status == "verified"
