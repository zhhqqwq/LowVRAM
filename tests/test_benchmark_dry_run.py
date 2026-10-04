"""P1-13 Benchmark Dry Run tests."""

from pathlib import Path

import pytest

import lowvram.dry_run as dry_run_module
import lowvram.runtime.preparation as preparation_module
from lowvram.dry_run import build_benchmark_dry_run
from lowvram.models.benchmark import ErrorType
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.preparation import BenchmarkPreparationRequest
from lowvram.models.runtime import RuntimeExecutionRequest, RuntimeExecutionResult


class ProbeOnlyAdapter:
    runtime_name = "llama.cpp"

    def __init__(self, results: list[RuntimeExecutionResult]) -> None:
        self.results = iter(results)
        self.requests: list[RuntimeExecutionRequest] = []

    def build_command(self, request: RuntimeExecutionRequest) -> list[str]:
        return [request.executable, *request.arguments]

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        self.requests.append(request)
        return next(self.results)


def _request(model_path: Path, llama_cli: str | None = None) -> BenchmarkPreparationRequest:
    return BenchmarkPreparationRequest(
        model_path=str(model_path),
        llama_cli=llama_cli,
        context_length=4096,
        threads=8,
        gpu_layers=0,
        batch_size=512,
        temperature=0.0,
        seed=42,
        prompt_version="v1",
    )


def _execution(output: str) -> RuntimeExecutionResult:
    return RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=True,
        exit_code=0,
        stdout=output,
        stderr="",
        duration_seconds=0.01,
    )


def _verified_detection(executable: str = "/opt/llama/llama-cli") -> LlamaCppDetectionResult:
    return LlamaCppDetectionResult(
        found=True,
        executable=executable,
        source="explicit_path",
        candidate_name="llama-cli",
        runnable=True,
        version="b9001",
        version_command=[executable, "--version"],
        verified_eligible=True,
        message="detected",
    )


def test_dry_run_returns_exact_command_without_model_spawn(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    executable = tmp_path / "llama-cli"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    adapter = ProbeOnlyAdapter([_execution("version: b9001\n")])

    result = build_benchmark_dry_run(
        _request(model, str(executable)),
        adapter=adapter,
    )

    assert result.ready is True
    assert result.runtime.name == "llama.cpp"
    assert result.runtime.version == "b9001"
    assert result.model == str(model)
    assert result.configuration.context_length == 4096
    assert result.configuration.threads == 8
    assert result.configuration.gpu_layers == 0
    assert result.configuration.batch_size == 512
    assert result.configuration.temperature == 0.0
    assert result.configuration.seed == 42
    assert result.prompt_version == "v1"
    assert result.command is not None
    assert result.command.argv[:15] == (
        str(executable),
        "--model",
        str(model),
        "--ctx-size",
        "4096",
        "--threads",
        "8",
        "--gpu-layers",
        "0",
        "--batch-size",
        "512",
        "--temp",
        "0.0",
        "--seed",
        "42",
    )
    prompt_index = result.command.argv.index("--prompt")
    assert "LowVRAM Benchmark Prompt v1" in result.command.argv[prompt_index + 1]
    assert [request.arguments for request in adapter.requests] == [["--version"]]


def test_missing_model_blocks_before_runtime_probe(tmp_path: Path) -> None:
    adapter = ProbeOnlyAdapter([_execution("version: b9001\n")])

    result = build_benchmark_dry_run(
        _request(tmp_path / "missing.gguf", "/missing/llama-cli"),
        adapter=adapter,
    )

    assert result.ready is False
    assert result.error_type == ErrorType.MODEL_NOT_FOUND
    assert result.command is None
    assert adapter.requests == []


def test_unrecognized_runtime_version_blocks_dry_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    detection = LlamaCppDetectionResult(
        found=True,
        executable="/opt/llama/llama-cli",
        source="explicit_path",
        candidate_name="llama-cli",
        runnable=True,
        version=None,
        verified_eligible=False,
        message="version output was not recognized",
    )
    monkeypatch.setattr(
        preparation_module,
        "detect_llama_cpp",
        lambda explicit_path, adapter: detection,
    )

    result = build_benchmark_dry_run(_request(model))

    assert result.ready is False
    assert result.runtime.version is None
    assert result.error_type == ErrorType.UNKNOWN
    assert "version" in (result.error_message or "")
    assert result.command is None


@pytest.mark.parametrize("flag", ["--prompt", "--prompt=other", "-p", "--prompt-file"])
def test_shared_preparation_rejects_prompt_override(
    monkeypatch,
    tmp_path: Path,
    flag: str,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    monkeypatch.setattr(
        preparation_module,
        "detect_llama_cpp",
        lambda explicit_path, adapter: _verified_detection(),
    )
    request = _request(model).model_copy(update={"extra_args": (flag,)})

    result = build_benchmark_dry_run(request)

    assert result.ready is False
    assert result.error_type == ErrorType.UNKNOWN
    assert "Standard Prompt" in (result.error_message or "")


def test_dry_run_wrapper_uses_shared_preparation_result(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    request = _request(model)
    prepared = preparation_module.prepare_llama_cpp_benchmark
    calls = 0

    def wrapped(request_arg, *, adapter=None):
        nonlocal calls
        calls += 1
        return prepared(request_arg, adapter=adapter)

    executable = tmp_path / "llama-cli"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    request = request.model_copy(update={"llama_cli": str(executable)})
    adapter = ProbeOnlyAdapter([_execution("version: b9001\n")])
    monkeypatch.setattr(dry_run_module, "prepare_llama_cpp_benchmark", wrapped)

    result = build_benchmark_dry_run(request, adapter=adapter)

    assert calls == 1
    assert result.ready is True
