"""P1-10 Benchmark Orchestrator integration tests."""

import json
from pathlib import Path

import pytest

import lowvram.runtime.orchestrator as orchestrator_module
from lowvram.models.benchmark import ErrorType
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.hardware import HardwareInfo
from lowvram.models.memory import RamMonitorResult, RamSample
from lowvram.models.model import ModelInfo
from lowvram.models.orchestrator import (
    BenchmarkOrchestrationRecord,
    BenchmarkOrchestratorRequest,
)
from lowvram.models.recipe import Recipe
from lowvram.models.runtime import RuntimeExecutionResult
from lowvram.models.vram import VramMonitorResult, VramSample
from lowvram.runtime import BenchmarkOrchestrator
from lowvram.runtime.adapter import RuntimeSpawnError


def _hardware() -> HardwareInfo:
    return HardwareInfo.model_validate(
        {
            "cpu": {
                "name": "Example CPU",
                "architecture": "x86_64",
                "physical_cores": 8,
                "logical_cores": 16,
            },
            "gpu": [],
            "ram_total_mb": 32768,
            "os": {"name": "Linux", "version": "6.8", "architecture": "x86_64"},
            "driver": {},
            "python_version": "3.11.9",
        }
    )


def _model() -> ModelInfo:
    return ModelInfo(
        id="example-7b-q4",
        name="Example 7B Q4",
        source="https://example.invalid/model",
        architecture="dense",
        parameter_count=7_000_000_000,
        format="gguf",
        quantization="Q4_K_M",
    )


def _recipe(*, extra_args: list[str] | None = None) -> Recipe:
    return Recipe(
        model_id="example-7b-q4",
        runtime="llama.cpp",
        context_length=4096,
        gpu_layers=0,
        threads=8,
        batch_size=512,
        extra_args=extra_args or [],
    )


def _request(model_path: Path, *, extra_args: list[str] | None = None) -> BenchmarkOrchestratorRequest:
    return BenchmarkOrchestratorRequest(
        model_path=str(model_path),
        model=_model(),
        configuration=_recipe(extra_args=extra_args),
        llama_cli="/opt/llama/llama-cli",
        temperature=0.0,
        seed=42,
        timeout_seconds=30,
    )


def _detection() -> LlamaCppDetectionResult:
    return LlamaCppDetectionResult(
        found=True,
        executable="/opt/llama/llama-cli",
        source="explicit_path",
        candidate_name="llama-cli",
        runnable=True,
        version="b9001",
        version_command=["/opt/llama/llama-cli", "--version"],
        verified_eligible=True,
        message="detected",
    )


def _execution_success() -> RuntimeExecutionResult:
    return RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=True,
        exit_code=0,
        stdout="generated text\n",
        stderr=(
            "llama_perf_context_print: prompt eval time = 1000.00 ms / 20 tokens "
            "(50.00 ms per token, 20.00 tokens per second)\n"
            "llama_perf_context_print: eval time = 2000.00 ms / 40 runs "
            "(50.00 ms per token, 20.00 tokens per second)\n"
        ),
        duration_seconds=3.0,
    )


def _ram_result() -> RamMonitorResult:
    return RamMonitorResult(
        baseline_process_ram_mb=100,
        current_process_ram_mb=120,
        peak_process_ram_mb=150,
        delta_process_ram_mb=50,
        baseline_system_ram_mb=1000,
        current_system_ram_mb=1100,
        peak_system_ram_mb=1200,
        delta_system_ram_mb=200,
        sample_count=3,
        sample_interval_seconds=0.1,
    )


def _vram_result() -> VramMonitorResult:
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


class FakeSession:
    pid = 4321

    def __init__(self, events: list[str], execution: RuntimeExecutionResult) -> None:
        self._events = events
        self._execution = execution
        self.cancelled = False

    def wait(self) -> RuntimeExecutionResult:
        self._events.append("wait")
        return self._execution

    def cancel(self) -> None:
        self.cancelled = True
        self._events.append("cancel")


class FakeAdapter:
    runtime_name = "llama.cpp"

    def __init__(self, events: list[str], execution: RuntimeExecutionResult) -> None:
        self.events = events
        self.execution = execution
        self.spawn_request = None
        self.session = FakeSession(events, execution)

    def build_command(self, request):
        return [request.executable, *request.arguments]

    def execute(self, request):
        raise AssertionError("detector is monkeypatched in orchestrator integration tests")

    def spawn(self, request):
        self.events.append("spawn")
        self.spawn_request = request
        return self.session


class FakeRamMonitor:
    def __init__(self, pid: int, events: list[str]) -> None:
        assert pid == 4321
        self.events = events

    def start(self) -> RamSample:
        self.events.append("ram_start")
        return RamSample(process_ram_mb=100, system_ram_mb=1000)

    def stop(self) -> RamMonitorResult:
        self.events.append("ram_stop")
        return _ram_result()


class FakeVramMonitor:
    def __init__(self, pid: int, events: list[str]) -> None:
        assert pid == 4321
        self.events = events

    def start(self) -> VramSample:
        self.events.append("vram_start")
        return VramSample(gpus=[], total_vram_used_mb=0)

    def stop(self) -> VramMonitorResult:
        self.events.append("vram_stop")
        return _vram_result()


def _patch_success_dependencies(monkeypatch, events: list[str]) -> None:
    monkeypatch.setattr(orchestrator_module, "collect_system", _hardware)
    monkeypatch.setattr(
        orchestrator_module,
        "detect_llama_cpp",
        lambda explicit_path, adapter: _detection(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "RamMonitor",
        lambda pid: FakeRamMonitor(pid, events),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "VramMonitor",
        lambda pid: FakeVramMonitor(pid, events),
    )


def test_orchestrator_composes_pid_monitors_parser_and_saved_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    output_path = tmp_path / "run" / "benchmark.json"
    events: list[str] = []
    adapter = FakeAdapter(events, _execution_success())
    _patch_success_dependencies(monkeypatch, events)

    record = BenchmarkOrchestrator(adapter=adapter).run(
        _request(model_path),
        output_path=output_path,
    )

    assert record.result.success is True
    assert events == ["spawn", "ram_start", "vram_start", "wait", "ram_stop", "vram_stop"]
    assert record.hardware == _hardware()
    assert record.detection == _detection()
    assert record.prompt_version == "v1"
    assert record.prompt_sha256 is not None
    assert record.command is not None
    assert record.execution == _execution_success()
    assert record.ram == _ram_result()
    assert record.vram == _vram_result()
    assert record.performance is not None
    assert record.performance.prompt_eval_time_seconds == 1.0
    assert record.performance.generation_tokens_per_second == 20.0

    assert adapter.spawn_request is not None
    assert record.command is not None
    assert tuple(adapter.build_command(adapter.spawn_request)) == record.command.argv
    prompt_index = record.command.argv.index("--prompt")
    assert "LowVRAM Benchmark Prompt v1" in record.command.argv[prompt_index + 1]

    loaded = BenchmarkOrchestrationRecord.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert loaded == record
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "p1.10.0"


def test_orchestrator_returns_parse_failed_with_monitor_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    events: list[str] = []
    bad_execution = RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=True,
        exit_code=0,
        stdout="generated text",
        stderr="no timings",
        duration_seconds=1.0,
    )
    adapter = FakeAdapter(events, bad_execution)
    _patch_success_dependencies(monkeypatch, events)

    record = BenchmarkOrchestrator(adapter=adapter).run(_request(model_path))

    assert record.result.success is False
    assert record.result.error_type == ErrorType.PARSE_FAILED
    assert record.execution == bad_execution
    assert record.ram == _ram_result()
    assert record.vram == _vram_result()
    assert record.performance is None


def test_orchestrator_rejects_missing_model_before_spawn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    adapter = FakeAdapter(events, _execution_success())
    monkeypatch.setattr(orchestrator_module, "collect_system", _hardware)

    record = BenchmarkOrchestrator(adapter=adapter).run(
        _request(tmp_path / "missing.gguf")
    )

    assert record.result.success is False
    assert record.result.error_type == ErrorType.MODEL_NOT_FOUND
    assert events == []
    assert record.command is None
    assert record.execution is None


@pytest.mark.parametrize("flag", ["--prompt", "--prompt=evil", "-p", "--prompt-file"])
def test_orchestrator_rejects_standard_prompt_override(
    monkeypatch,
    tmp_path: Path,
    flag: str,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    events: list[str] = []
    adapter = FakeAdapter(events, _execution_success())
    monkeypatch.setattr(orchestrator_module, "collect_system", _hardware)
    monkeypatch.setattr(
        orchestrator_module,
        "detect_llama_cpp",
        lambda explicit_path, adapter: _detection(),
    )

    record = BenchmarkOrchestrator(adapter=adapter).run(
        _request(model_path, extra_args=[flag])
    )

    assert record.result.success is False
    assert record.result.error_type == ErrorType.UNKNOWN
    assert "Standard Prompt" in (record.result.error_message or "")
    assert events == []


def test_orchestrator_propagates_structured_spawn_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")
    events: list[str] = []

    class FailingAdapter(FakeAdapter):
        def spawn(self, request):
            raise RuntimeSpawnError(
                error_type=ErrorType.RUNTIME_NOT_FOUND,
                message="runtime vanished before benchmark spawn",
                duration_seconds=0.01,
            )

    adapter = FailingAdapter(events, _execution_success())
    monkeypatch.setattr(orchestrator_module, "collect_system", _hardware)
    monkeypatch.setattr(
        orchestrator_module,
        "detect_llama_cpp",
        lambda explicit_path, adapter: _detection(),
    )

    record = BenchmarkOrchestrator(adapter=adapter).run(_request(model_path))

    assert record.result.success is False
    assert record.result.error_type == ErrorType.RUNTIME_NOT_FOUND
    assert "vanished" in (record.result.error_message or "")
