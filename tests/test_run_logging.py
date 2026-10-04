"""P1-14 run logging tests."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

import lowvram.runtime.logging as logging_module
from lowvram.models.benchmark import BenchmarkResult, ErrorType
from lowvram.models.failure import (
    FailureClassificationResult,
    FailureEvidenceSource,
)
from lowvram.models.model import ModelInfo
from lowvram.models.orchestrator import BenchmarkOrchestrationRecord
from lowvram.models.recipe import Recipe
from lowvram.models.runtime import RuntimeExecutionResult
from lowvram.runtime.logging import RunLoggingError, persist_run_artifacts


def _record(
    *,
    run_id: str = "run-123",
    stdout: str = "stdout 日本語\n$HOME; echo untouched\n",
    stderr: str = "stderr café\n",
    execution: bool = True,
) -> BenchmarkOrchestrationRecord:
    message = "runtime process exited with code 7"
    runtime_result = (
        RuntimeExecutionResult(
            runtime_name="llama.cpp",
            success=False,
            exit_code=7,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=1.0,
            error_type=ErrorType.PROCESS_CRASH,
            error_message=message,
        )
        if execution
        else None
    )
    return BenchmarkOrchestrationRecord(
        run_id=run_id,
        timestamp=datetime(2026, 10, 4, tzinfo=UTC),
        model=ModelInfo(
            id="example",
            name="Example",
            source="https://example.invalid/model",
            architecture="dense",
            parameter_count=1,
            format="gguf",
            quantization="Q4",
        ),
        configuration=Recipe(
            model_id="example",
            runtime="llama.cpp",
            context_length=4096,
            gpu_layers=0,
            threads=8,
            batch_size=512,
        ),
        prompt_version="v1",
        execution=runtime_result,
        failure_classification=FailureClassificationResult(
            error_type=ErrorType.PROCESS_CRASH,
            rule="process_crash.exit",
            message=message,
            evidence_sources=(FailureEvidenceSource.STRUCTURED,),
        ),
        result=BenchmarkResult(
            success=False,
            error_type=ErrorType.PROCESS_CRASH,
            error_message=message,
        ),
    )


def test_persist_run_artifacts_writes_exact_three_files(tmp_path: Path) -> None:
    record = _record()
    runs_root = tmp_path / "runs"

    paths = persist_run_artifacts(record, runs_root)

    assert paths.run_dir == runs_root / record.run_id
    assert sorted(path.name for path in paths.run_dir.iterdir()) == [
        "benchmark.json",
        "stderr.log",
        "stdout.log",
    ]
    loaded = BenchmarkOrchestrationRecord.model_validate_json(
        paths.benchmark_json.read_text(encoding="utf-8")
    )
    assert loaded == record
    assert paths.stdout_log.read_text(encoding="utf-8") == record.execution.stdout
    assert paths.stderr_log.read_text(encoding="utf-8") == record.execution.stderr


def test_logging_preserves_runtime_text_without_shell_transformation(
    tmp_path: Path,
) -> None:
    stdout = "one\n$HOME; rm -rf never-executed\n日本語\n"
    stderr = "two\n$(echo untouched)\n"
    record = _record(stdout=stdout, stderr=stderr)

    paths = persist_run_artifacts(record, tmp_path / "runs")

    assert paths.stdout_log.read_bytes() == stdout.encode("utf-8")
    assert paths.stderr_log.read_bytes() == stderr.encode("utf-8")


def test_pre_execution_failure_still_gets_empty_log_files(tmp_path: Path) -> None:
    record = _record(execution=False)

    paths = persist_run_artifacts(record, tmp_path / "runs")

    assert paths.stdout_log.read_bytes() == b""
    assert paths.stderr_log.read_bytes() == b""


def test_final_run_directory_appears_only_after_all_files_exist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    record = _record()
    runs_root = tmp_path / "runs"
    real_rename = os.rename
    checked = False

    def checked_rename(source, destination):
        nonlocal checked
        source_path = Path(source)
        destination_path = Path(destination)
        assert destination_path == runs_root / record.run_id
        assert not destination_path.exists()
        assert sorted(path.name for path in source_path.iterdir()) == [
            "benchmark.json",
            "stderr.log",
            "stdout.log",
        ]
        checked = True
        real_rename(source, destination)

    monkeypatch.setattr(logging_module.os, "rename", checked_rename)

    persist_run_artifacts(record, runs_root)

    assert checked is True
    assert (runs_root / record.run_id).is_dir()


def test_write_failure_cleans_staging_and_never_commits_final_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    record = _record()
    runs_root = tmp_path / "runs"
    real_write = logging_module._write_utf8_text

    def fail_on_stderr(path: Path, content: str) -> None:
        if path.name == "stderr.log":
            raise OSError("simulated disk failure")
        real_write(path, content)

    monkeypatch.setattr(logging_module, "_write_utf8_text", fail_on_stderr)

    with pytest.raises(RunLoggingError, match="failed to persist run artifacts"):
        persist_run_artifacts(record, runs_root)

    assert not (runs_root / record.run_id).exists()
    assert list(runs_root.iterdir()) == []


def test_existing_run_directory_is_never_overwritten(tmp_path: Path) -> None:
    record = _record()
    runs_root = tmp_path / "runs"
    final_dir = runs_root / record.run_id
    final_dir.mkdir(parents=True)
    sentinel = final_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(RunLoggingError, match="already exists"):
        persist_run_artifacts(record, runs_root)

    assert sentinel.read_text(encoding="utf-8") == "keep"
