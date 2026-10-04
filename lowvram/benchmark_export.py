"""Convert P1 orchestration evidence into the public BenchmarkRun contract."""

from pydantic import ValidationError

from lowvram.models.benchmark import (
    BenchmarkRun,
    MemoryMetrics,
    PerformanceMetrics,
    RuntimeInfo,
    VerificationInfo,
)
from lowvram.models.orchestrator import BenchmarkOrchestrationRecord


class BenchmarkExportError(ValueError):
    """Raised when orchestration evidence is insufficient for a BenchmarkRun."""


def build_benchmark_run(record: BenchmarkOrchestrationRecord) -> BenchmarkRun:
    """Build a path-free BenchmarkRun from one orchestration evidence record."""
    hardware = record.hardware
    if hardware is None:
        raise BenchmarkExportError("benchmark export requires captured hardware")

    detection = record.detection
    runtime_version = detection.version if detection is not None else None
    peak_ram_mb = record.ram.peak_process_ram_mb if record.ram is not None else None
    peak_vram_mb = _peak_vram_mb(record)

    performance = record.performance
    load_time_seconds = performance.load_time_seconds if performance is not None else None
    prompt_tokens_per_second = (
        performance.prompt_tokens_per_second if performance is not None else None
    )
    generation_tokens_per_second = (
        performance.generation_tokens_per_second if performance is not None else None
    )

    if record.result.success:
        if peak_ram_mb is None:
            raise BenchmarkExportError(
                "successful benchmark export requires process-tree peak RAM"
            )
        if peak_vram_mb is None:
            raise BenchmarkExportError(
                "successful benchmark export requires attributable peak VRAM"
            )

    try:
        return BenchmarkRun(
            run_id=record.run_id,
            timestamp=record.timestamp,
            hardware=hardware,
            model=record.model,
            runtime=RuntimeInfo(name="llama.cpp", version=runtime_version),
            prompt_version=record.prompt_version,
            configuration=record.configuration,
            memory=MemoryMetrics(peak_vram_mb=peak_vram_mb, peak_ram_mb=peak_ram_mb),
            performance=PerformanceMetrics(
                load_time_seconds=load_time_seconds,
                prompt_tokens_per_second=prompt_tokens_per_second,
                generation_tokens_per_second=generation_tokens_per_second,
            ),
            result=record.result,
            verification=_verification_info(record),
        )
    except ValidationError as exc:
        raise BenchmarkExportError(f"benchmark export validation failed: {exc}") from exc


def _peak_vram_mb(record: BenchmarkOrchestrationRecord) -> int | None:
    hardware = record.hardware
    vram = record.vram
    if hardware is None or vram is None:
        return None
    if not hardware.gpu:
        return 0
    if vram.process_vram_supported:
        return vram.peak_process_vram_mb
    return None


def _verification_info(record: BenchmarkOrchestrationRecord) -> VerificationInfo:
    detection = record.detection
    verified = (
        detection is not None
        and detection.verified_eligible
        and detection.version is not None
        and record.prompt_sha256 is not None
        and record.command is not None
    )
    evidence: list[str] = [f"orchestration:{record.schema_version}"]
    if record.prompt_sha256 is not None:
        evidence.append(f"prompt:{record.prompt_version}:{record.prompt_sha256}")
    if detection is not None and detection.version is not None:
        evidence.append(f"runtime:llama.cpp:{detection.version}")
    if record.performance is not None:
        evidence.append(f"timing_format:{record.performance.timing_format}")
    if record.failure_classification is not None:
        evidence.append(f"failure_rule:{record.failure_classification.rule}")

    notes = (
        "Local runtime paths and raw logs remain in private run artifacts; "
        "this BenchmarkRun intentionally omits them."
    )
    if record.performance is not None and record.performance.load_time_seconds is None:
        notes += (
            " load_time_seconds is null because this llama.cpp timing format did not "
            "directly expose model load time; total process duration was not substituted."
        )

    return VerificationInfo(
        status="verified" if verified else "unverified",
        evidence=evidence,
        notes=notes,
    )
