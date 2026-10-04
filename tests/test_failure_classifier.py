"""P1-11 failure-classification tests."""

from pathlib import Path

import pytest

from lowvram.models.benchmark import ErrorType
from lowvram.models.failure import (
    FailureClassificationRequest,
    FailureEvidenceSource,
    FailureStage,
)
from lowvram.models.runtime import RuntimeExecutionResult
from lowvram.runtime import classify_failure

FIXTURES = Path(__file__).parent / "fixtures" / "failures"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _crash(*, stdout: str = "", stderr: str = "", exit_code: int = 1) -> RuntimeExecutionResult:
    return RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=False,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=1.0,
        error_type=ErrorType.PROCESS_CRASH,
        error_message=f"runtime process exited with code {exit_code}",
    )


def test_preflight_missing_model_is_model_not_found() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.PREFLIGHT,
            model_path_exists=False,
            message="model file not found",
        )
    )

    assert result.error_type == ErrorType.MODEL_NOT_FOUND
    assert result.rule == "model_not_found.preflight"
    assert result.evidence_sources == (FailureEvidenceSource.STRUCTURED,)


def test_runtime_not_found_structured_status_beats_text() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_START,
            upstream_error_type=ErrorType.RUNTIME_NOT_FOUND,
            message="out of memory text from an unrelated launcher message",
        )
    )

    assert result.error_type == ErrorType.RUNTIME_NOT_FOUND
    assert result.rule == "runtime_not_found.structured"


def test_timeout_has_precedence_over_oom_text() -> None:
    execution = RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=False,
        exit_code=None,
        stdout="",
        stderr=_fixture("oom_cuda.txt"),
        duration_seconds=30,
        error_type=ErrorType.TIMEOUT,
        error_message="timed out",
    )

    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=execution,
        )
    )

    assert result.error_type == ErrorType.TIMEOUT
    assert result.rule == "timeout.structured"


@pytest.mark.parametrize(
    ("fixture_name", "expected_rule", "expected_source"),
    [
        ("oom_cuda.txt", "oom.cuda_error", FailureEvidenceSource.STDERR),
        ("oom_cpu.txt", "oom.cpu_bad_alloc", FailureEvidenceSource.STDERR),
        ("oom_allocation.txt", "oom.allocation_failure", FailureEvidenceSource.STDERR),
    ],
)
def test_explicit_oom_evidence_refines_process_crash(
    fixture_name: str,
    expected_rule: str,
    expected_source: FailureEvidenceSource,
) -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=_crash(stderr=_fixture(fixture_name)),
        )
    )

    assert result.error_type == ErrorType.OUT_OF_MEMORY
    assert result.rule == expected_rule
    assert expected_source in result.evidence_sources
    assert FailureEvidenceSource.STRUCTURED in result.evidence_sources


def test_stdout_is_valid_failure_evidence_source() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=_crash(stdout="fatal: std::bad_alloc\n"),
        )
    )

    assert result.error_type == ErrorType.OUT_OF_MEMORY
    assert FailureEvidenceSource.STDOUT in result.evidence_sources


def test_model_file_disappearing_after_preflight_is_model_not_found() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=_crash(stderr=_fixture("model_missing.txt")),
        )
    )

    assert result.error_type == ErrorType.MODEL_NOT_FOUND
    assert result.rule == "model_not_found.gguf_open"


def test_model_load_failure_is_classified_from_runtime_evidence() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=_crash(stderr=_fixture("model_load_failed.txt")),
        )
    )

    assert result.error_type == ErrorType.MODEL_LOAD_FAILED
    assert result.rule == "model_load_failed.explicit"


def test_oom_has_precedence_over_model_load_failure() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=_crash(stderr=_fixture("oom_during_model_load.txt")),
        )
    )

    assert result.error_type == ErrorType.OUT_OF_MEMORY
    assert result.rule.startswith("oom.")


def test_generic_nonzero_exit_remains_process_crash() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=_crash(stderr=_fixture("generic_crash.txt"), exit_code=7),
        )
    )

    assert result.error_type == ErrorType.PROCESS_CRASH
    assert result.rule == "process_crash.exit"


def test_memory_word_alone_does_not_create_false_oom() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.RUNTIME_EXECUTION,
            execution=_crash(stderr=_fixture("ambiguous_memory.txt")),
        )
    )

    assert result.error_type == ErrorType.PROCESS_CRASH


def test_parse_failure_is_preserved_after_successful_execution() -> None:
    execution = RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=True,
        exit_code=0,
        stdout="generated text",
        stderr="no timing lines",
        duration_seconds=1.0,
    )
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.OUTPUT_PARSE,
            execution=execution,
            upstream_error_type=ErrorType.PARSE_FAILED,
            message="timing output incomplete",
        )
    )

    assert result.error_type == ErrorType.PARSE_FAILED
    assert result.rule == "parse_failed.structured"


def test_unknown_is_the_final_fallback() -> None:
    result = classify_failure(
        FailureClassificationRequest(
            stage=FailureStage.INTERNAL,
            message="monitor returned an unexpected state",
        )
    )

    assert result.error_type == ErrorType.UNKNOWN
    assert result.rule == "unknown.fallback"
