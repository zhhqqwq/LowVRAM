"""P1 benchmark orchestration with P1-11 failure classification."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from jsonschema import Draft202012Validator

from lowvram.collectors import RamMonitor, VramMonitor, collect_system
from lowvram.models.benchmark import BenchmarkResult, ErrorType
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.failure import (
    FailureClassificationRequest,
    FailureClassificationResult,
    FailureStage,
)
from lowvram.models.hardware import HardwareInfo
from lowvram.models.memory import RamMonitorResult
from lowvram.models.orchestrator import (
    BenchmarkOrchestrationRecord,
    BenchmarkOrchestratorRequest,
)
from lowvram.models.output_parser import LlamaCppTimingMetrics
from lowvram.models.preparation import BenchmarkPreparationRequest
from lowvram.models.runtime import RuntimeExecutionResult
from lowvram.models.vram import VramMonitorResult
from lowvram.runtime.adapter import RuntimeSpawnError, SubprocessRuntimeAdapter
from lowvram.runtime.command_builder import to_runtime_execution_request
from lowvram.runtime.failure_classifier import classify_failure
from lowvram.runtime.llama_cpp import LlamaCppRuntimeAdapter
from lowvram.runtime.output_parser import parse_llama_cpp_output
from lowvram.runtime.preparation import prepare_llama_cpp_benchmark


class BenchmarkOrchestrator:
    """Compose P1 collectors/runtime components into one benchmark attempt."""

    def __init__(self, adapter: SubprocessRuntimeAdapter | None = None) -> None:
        self._adapter = adapter or LlamaCppRuntimeAdapter()

    def run(
        self,
        request: BenchmarkOrchestratorRequest,
        *,
        output_path: Path | None = None,
    ) -> BenchmarkOrchestrationRecord:
        """Run one benchmark attempt and optionally save its validated evidence JSON."""
        run_id = str(uuid4())
        timestamp = datetime.now(UTC)

        hardware: HardwareInfo | None = None
        detection: LlamaCppDetectionResult | None = None
        prompt_sha256: str | None = None
        command = None
        execution: RuntimeExecutionResult | None = None
        ram_result: RamMonitorResult | None = None
        vram_result: VramMonitorResult | None = None
        performance: LlamaCppTimingMetrics | None = None

        def finish(
            result: BenchmarkResult,
            classification: FailureClassificationResult | None = None,
        ) -> BenchmarkOrchestrationRecord:
            record = BenchmarkOrchestrationRecord(
                run_id=run_id,
                timestamp=timestamp,
                hardware=hardware,
                model=request.model,
                configuration=request.configuration,
                detection=detection,
                prompt_version=request.prompt_version,
                prompt_sha256=prompt_sha256,
                command=command,
                execution=execution,
                ram=ram_result,
                vram=vram_result,
                performance=performance,
                failure_classification=classification,
                result=result,
            )
            if output_path is not None:
                save_benchmark_orchestration_record(record, output_path)
            return record

        def fail(evidence: FailureClassificationRequest) -> BenchmarkOrchestrationRecord:
            classification = classify_failure(evidence)
            return finish(
                BenchmarkResult(
                    success=False,
                    error_type=classification.error_type,
                    error_message=classification.message,
                ),
                classification,
            )

        try:
            hardware = collect_system()
        except Exception as exc:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message=f"system collection failed: {exc}",
                )
            )

        preparation = prepare_llama_cpp_benchmark(
            BenchmarkPreparationRequest(
                model_path=request.model_path,
                llama_cli=request.llama_cli,
                context_length=request.configuration.context_length,
                threads=request.configuration.threads,
                gpu_layers=request.configuration.gpu_layers,
                batch_size=request.configuration.batch_size,
                temperature=request.temperature,
                seed=request.seed,
                prompt_version=request.prompt_version,
                extra_args=tuple(request.configuration.extra_args),
            ),
            adapter=self._adapter,
        )
        detection = preparation.detection
        prompt_sha256 = preparation.prompt_sha256
        command = preparation.command

        if not preparation.ready:
            return fail(
                FailureClassificationRequest(
                    stage=preparation.failure_stage or FailureStage.INTERNAL,
                    upstream_error_type=preparation.error_type or ErrorType.UNKNOWN,
                    message=preparation.error_message or "benchmark preparation failed",
                )
            )

        if command is None:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message="benchmark preparation returned no command",
                )
            )

        execution_request = to_runtime_execution_request(
            command,
            timeout_seconds=request.timeout_seconds,
        )

        try:
            session = self._adapter.spawn(execution_request)
        except RuntimeSpawnError as exc:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.RUNTIME_START,
                    upstream_error_type=exc.error_type,
                    message=exc.message,
                )
            )

        ram_monitor = RamMonitor(session.pid)
        vram_monitor = VramMonitor(session.pid)
        ram_started = False
        vram_started = False

        try:
            ram_monitor.start()
            ram_started = True
            vram_monitor.start()
            vram_started = True
        except Exception as exc:
            session.cancel()
            ram_result = _stop_ram_monitor(ram_monitor) if ram_started else None
            vram_result = _stop_vram_monitor(vram_monitor) if vram_started else None
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message=f"memory monitor startup failed: {exc}",
                )
            )

        try:
            execution = session.wait()
        except Exception as exc:
            session.cancel()
            ram_result = _stop_ram_monitor(ram_monitor)
            vram_result = _stop_vram_monitor(vram_monitor)
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message=f"runtime session wait failed: {exc}",
                )
            )

        try:
            ram_result = ram_monitor.stop()
            vram_result = vram_monitor.stop()
        except Exception as exc:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message=f"memory monitor stop failed: {exc}",
                )
            )

        if not execution.success:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.RUNTIME_EXECUTION,
                    execution=execution,
                )
            )

        parsed = parse_llama_cpp_output(
            stdout=execution.stdout,
            stderr=execution.stderr,
        )
        if not parsed.success or parsed.metrics is None:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.OUTPUT_PARSE,
                    execution=execution,
                    upstream_error_type=parsed.error_type or ErrorType.PARSE_FAILED,
                    message=parsed.error_message or "llama.cpp timing parse failed",
                )
            )

        performance = parsed.metrics
        return finish(BenchmarkResult(success=True))

    @property
    def adapter(self) -> SubprocessRuntimeAdapter:
        """Expose the configured adapter for integration and diagnostics."""
        return self._adapter


def _stop_ram_monitor(monitor: RamMonitor) -> RamMonitorResult | None:
    try:
        return monitor.stop()
    except Exception:
        return None


def _stop_vram_monitor(monitor: VramMonitor) -> VramMonitorResult | None:
    try:
        return monitor.stop()
    except Exception:
        return None


def save_benchmark_orchestration_record(
    record: BenchmarkOrchestrationRecord,
    output_path: Path,
) -> None:
    """Schema-validate serialized P1 JSON before writing it to disk."""
    payload = record.model_dump(mode="json")
    schema = BenchmarkOrchestrationRecord.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
