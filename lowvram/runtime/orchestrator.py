"""P1 benchmark orchestration with P1-11 failure classification."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from lowvram.collectors import RamMonitor, VramMonitor, collect_system
from lowvram.models.benchmark import BenchmarkResult, ErrorType
from lowvram.models.command import LlamaCppCommandRequest
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
from lowvram.models.runtime import RuntimeExecutionResult
from lowvram.models.vram import VramMonitorResult
from lowvram.prompts import BenchmarkPromptError, load_benchmark_prompt
from lowvram.runtime.adapter import RuntimeSpawnError, SubprocessRuntimeAdapter
from lowvram.runtime.command_builder import (
    build_llama_cpp_command,
    to_runtime_execution_request,
)
from lowvram.runtime.detector import detect_llama_cpp
from lowvram.runtime.failure_classifier import classify_failure
from lowvram.runtime.llama_cpp import LlamaCppRuntimeAdapter
from lowvram.runtime.output_parser import parse_llama_cpp_output

_PROMPT_OVERRIDE_FLAGS = ("-p", "--prompt", "-f", "--file", "--prompt-file")


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

        model_path = Path(request.model_path)
        if not model_path.is_file():
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.PREFLIGHT,
                    model_path_exists=False,
                    message=f"model file not found: {request.model_path}",
                )
            )

        detection = detect_llama_cpp(
            request.llama_cli,
            adapter=self._adapter,
        )
        if not detection.found:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.RUNTIME_START,
                    upstream_error_type=(
                        detection.probe_error_type or ErrorType.RUNTIME_NOT_FOUND
                    ),
                    message=detection.message or "llama.cpp runtime was not found",
                )
            )
        if not detection.runnable:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.RUNTIME_START,
                    upstream_error_type=detection.probe_error_type or ErrorType.UNKNOWN,
                    message=detection.message or "llama.cpp runtime is not runnable",
                )
            )
        executable = detection.executable
        if executable is None:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message="llama.cpp detection returned no executable path",
                )
            )

        try:
            prompt = load_benchmark_prompt(request.prompt_version)
        except BenchmarkPromptError as exc:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message=str(exc),
                )
            )
        prompt_sha256 = prompt.sha256

        prompt_override = _find_prompt_override(request.configuration.extra_args)
        if prompt_override is not None:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.PREFLIGHT,
                    message=(
                        "configuration.extra_args cannot override Standard Prompt: "
                        f"{prompt_override}"
                    ),
                )
            )

        try:
            command_request = LlamaCppCommandRequest(
                executable=executable,
                model_path=request.model_path,
                context_length=request.configuration.context_length,
                threads=request.configuration.threads,
                gpu_layers=request.configuration.gpu_layers,
                batch_size=request.configuration.batch_size,
                temperature=request.temperature,
                seed=request.seed,
                extra_args=(
                    *request.configuration.extra_args,
                    "--prompt",
                    prompt.content,
                ),
            )
            command = build_llama_cpp_command(command_request)
            execution_request = to_runtime_execution_request(
                command,
                timeout_seconds=request.timeout_seconds,
            )
        except ValidationError as exc:
            return fail(
                FailureClassificationRequest(
                    stage=FailureStage.INTERNAL,
                    message=f"command construction failed: {exc}",
                )
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


def _find_prompt_override(arguments: list[str]) -> str | None:
    """Reject caller-provided prompt flags so P1-08 remains the workload source."""
    for argument in arguments:
        for flag in _PROMPT_OVERRIDE_FLAGS:
            if argument == flag or argument.startswith(f"{flag}="):
                return flag
    return None


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
