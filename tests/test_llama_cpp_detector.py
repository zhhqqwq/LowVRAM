"""P1-06 llama.cpp detector tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from lowvram.models.benchmark import ErrorType
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.runtime import RuntimeExecutionRequest, RuntimeExecutionResult
from lowvram.runtime import detector


class FakeAdapter:
    runtime_name = "llama.cpp"

    def __init__(self, results: list[RuntimeExecutionResult]) -> None:
        self.results = iter(results)
        self.requests: list[RuntimeExecutionRequest] = []

    def build_command(self, request: RuntimeExecutionRequest) -> list[str]:
        return [request.executable, *request.arguments]

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        self.requests.append(request)
        return next(self.results)


def _execution_result(
    *,
    success: bool,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 0,
    error_type: ErrorType | None = None,
    error_message: str | None = None,
) -> RuntimeExecutionResult:
    return RuntimeExecutionResult(
        runtime_name="llama.cpp",
        success=success,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.01,
        error_type=error_type,
        error_message=error_message,
    )


def test_candidate_names_cover_windows_and_linux() -> None:
    assert detector.llama_cpp_candidate_names("Linux") == ("llama-cli", "main")
    assert detector.llama_cpp_candidate_names("Windows") == (
        "llama-cli.exe",
        "llama-cli",
        "main.exe",
        "main",
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("version: 6382 (deadbeef)\n", "6382"),
        ("llama.cpp version b7421\n", "b7421"),
        ("build: 8123\n", "b8123"),
        ("some text b9001 more text\n", "b9001"),
        ("version: unknown\n", None),
        ("version: development\n", None),
        ("unrelated output\n", None),
    ],
)
def test_parse_llama_cpp_version(output: str, expected: str | None) -> None:
    assert detector.parse_llama_cpp_version(output) == expected


def test_explicit_path_has_priority_and_parses_version(monkeypatch) -> None:
    adapter = FakeAdapter(
        [_execution_result(success=True, stdout="version: 6382 (deadbeef)\n")]
    )
    monkeypatch.setattr(detector, "_is_executable_file", lambda path, system_name=None: True)
    monkeypatch.setattr(
        detector.shutil,
        "which",
        lambda name: pytest.fail("PATH search must not run for explicit_path"),
    )

    result = detector.detect_llama_cpp(
        "/opt/llama/llama-cli",
        adapter=adapter,
        system_name="Linux",
    )

    assert result.found is True
    assert result.source == "explicit_path"
    assert result.executable == "/opt/llama/llama-cli"
    assert result.candidate_name == "llama-cli"
    assert result.runnable is True
    assert result.version == "6382"
    assert result.verified_eligible is True
    assert adapter.requests[0].arguments == ["--version"]


def test_invalid_explicit_path_does_not_fall_back_to_path(monkeypatch) -> None:
    monkeypatch.setattr(detector, "_is_executable_file", lambda path, system_name=None: False)
    monkeypatch.setattr(
        detector.shutil,
        "which",
        lambda name: pytest.fail("invalid explicit path must not silently fall back"),
    )

    result = detector.detect_llama_cpp("/missing/llama-cli", system_name="Linux")

    assert result.found is False
    assert result.probe_error_type == ErrorType.RUNTIME_NOT_FOUND
    assert result.verified_eligible is False


def test_explicit_path_unknown_start_error_is_found_but_not_runnable(monkeypatch) -> None:
    adapter = FakeAdapter(
        [
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.UNKNOWN,
                error_message="too many open files",
            ),
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.UNKNOWN,
                error_message="too many open files",
            ),
        ]
    )
    monkeypatch.setattr(detector, "_is_executable_file", lambda path, system_name=None: True)

    result = detector.detect_llama_cpp(
        "/opt/llama/llama-cli",
        adapter=adapter,
        system_name="Linux",
    )

    assert result.found is True
    assert result.runnable is False
    assert result.version is None
    assert result.verified_eligible is False
    assert result.probe_error_type == ErrorType.UNKNOWN


def test_path_search_prefers_llama_cli_when_verified(monkeypatch) -> None:
    searched: list[str] = []

    def fake_which(name: str) -> str | None:
        searched.append(name)
        if name == "llama-cli":
            return "/usr/local/bin/llama-cli"
        return None

    adapter = FakeAdapter([_execution_result(success=True, stdout="version: 7000\n")])
    monkeypatch.setattr(detector.shutil, "which", fake_which)

    result = detector.detect_llama_cpp(
        adapter=adapter,
        system_name="Linux",
    )

    assert result.found is True
    assert result.executable == "/usr/local/bin/llama-cli"
    assert result.candidate_name == "llama-cli"
    assert searched == ["llama-cli"]


def test_path_search_supports_legacy_main(monkeypatch) -> None:
    def fake_which(name: str) -> str | None:
        if name == "main":
            return "/usr/local/bin/main"
        return None

    adapter = FakeAdapter([_execution_result(success=True, stderr="build: 4321\n")])
    monkeypatch.setattr(detector.shutil, "which", fake_which)

    result = detector.detect_llama_cpp(
        adapter=adapter,
        system_name="Linux",
    )

    assert result.found is True
    assert result.candidate_name == "main"
    assert result.version == "b4321"
    assert result.verified_eligible is True


def test_path_search_continues_after_unrecognized_preferred_candidate(monkeypatch) -> None:
    candidates = {
        "llama-cli": "/usr/local/bin/llama-cli",
        "main": "/usr/local/bin/main",
    }
    monkeypatch.setattr(detector.shutil, "which", lambda name: candidates.get(name))
    adapter = FakeAdapter(
        [
            _execution_result(success=True, stdout="version: unknown\n"),
            _execution_result(success=True, stdout="still unknown\n"),
            _execution_result(success=True, stdout="build: 9001\n"),
        ]
    )

    result = detector.detect_llama_cpp(adapter=adapter, system_name="Linux")

    assert result.executable == "/usr/local/bin/main"
    assert result.candidate_name == "main"
    assert result.version == "b9001"
    assert result.verified_eligible is True


def test_path_search_returns_preferred_runnable_fallback_when_none_verify(
    monkeypatch,
) -> None:
    candidates = {
        "llama-cli": "/usr/local/bin/llama-cli",
        "main": "/usr/local/bin/main",
    }
    monkeypatch.setattr(detector.shutil, "which", lambda name: candidates.get(name))
    adapter = FakeAdapter(
        [
            _execution_result(success=True, stdout="version: unknown\n"),
            _execution_result(success=True, stdout="still unknown\n"),
            _execution_result(success=True, stdout="version: development\n"),
            _execution_result(success=True, stdout="still unknown\n"),
        ]
    )

    result = detector.detect_llama_cpp(adapter=adapter, system_name="Linux")

    assert result.executable == "/usr/local/bin/llama-cli"
    assert result.candidate_name == "llama-cli"
    assert result.runnable is True
    assert result.version is None
    assert result.verified_eligible is False


def test_path_search_prefers_later_runnable_candidate_over_start_failure(monkeypatch) -> None:
    candidates = {
        "llama-cli": "/bad/llama-cli",
        "main": "/good/main",
    }
    monkeypatch.setattr(detector.shutil, "which", lambda name: candidates.get(name))
    adapter = FakeAdapter(
        [
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.UNKNOWN,
                error_message="resource failure",
            ),
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.UNKNOWN,
                error_message="resource failure",
            ),
            _execution_result(success=True, stdout="version: unknown\n"),
            _execution_result(success=True, stdout="still unknown\n"),
        ]
    )

    result = detector.detect_llama_cpp(adapter=adapter, system_name="Linux")

    assert result.executable == "/good/main"
    assert result.runnable is True
    assert result.version is None


def test_windows_duplicate_candidate_resolution_is_probed_once(monkeypatch) -> None:
    searched: list[str] = []

    def fake_which(name: str) -> str | None:
        searched.append(name)
        if name in {"llama-cli.exe", "llama-cli"}:
            return r"C:\Tools\llama-cli.exe"
        return None

    adapter = FakeAdapter(
        [
            _execution_result(success=True, stdout="version: unknown\n"),
            _execution_result(success=True, stdout="still unknown\n"),
        ]
    )
    monkeypatch.setattr(detector.shutil, "which", fake_which)

    result = detector.detect_llama_cpp(adapter=adapter, system_name="Windows")

    assert result.executable == r"C:\Tools\llama-cli.exe"
    assert len(adapter.requests) == 2
    assert "llama-cli.exe" in searched
    assert "llama-cli" in searched


def test_version_probe_falls_back_to_positional_version(monkeypatch) -> None:
    adapter = FakeAdapter(
        [
            _execution_result(success=True, stdout="no parseable version here\n"),
            _execution_result(success=True, stdout="version: b7555\n"),
        ]
    )
    monkeypatch.setattr(detector.shutil, "which", lambda name: "/bin/llama-cli")

    result = detector.detect_llama_cpp(
        adapter=adapter,
        system_name="Linux",
    )

    assert result.version == "b7555"
    assert result.version_command == ["/bin/llama-cli", "version"]
    assert [request.arguments for request in adapter.requests] == [
        ["--version"],
        ["version"],
    ]


def test_timeout_probe_proves_process_was_runnable(monkeypatch) -> None:
    adapter = FakeAdapter(
        [
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.TIMEOUT,
                error_message="timed out",
            ),
            _execution_result(
                success=False,
                exit_code=7,
                error_type=ErrorType.PROCESS_CRASH,
                error_message="code 7",
            ),
        ]
    )
    monkeypatch.setattr(detector.shutil, "which", lambda name: "/bin/llama-cli")

    result = detector.detect_llama_cpp(adapter=adapter, system_name="Linux")

    assert result.found is True
    assert result.runnable is True
    assert result.version is None
    assert result.verified_eligible is False


def test_not_found_result_when_path_has_no_candidates(monkeypatch) -> None:
    monkeypatch.setattr(detector.shutil, "which", lambda name: None)

    result = detector.detect_llama_cpp(system_name="Linux")

    assert result == LlamaCppDetectionResult(
        found=False,
        probe_error_type=ErrorType.RUNTIME_NOT_FOUND,
        message="llama.cpp executable was not found on PATH",
    )


def test_path_candidate_that_cannot_start_is_retained_as_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(
        detector.shutil,
        "which",
        lambda name: "/bad/llama-cli" if name == "llama-cli" else None,
    )
    adapter = FakeAdapter(
        [
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.RUNTIME_NOT_FOUND,
                error_message="cannot start",
            ),
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.RUNTIME_NOT_FOUND,
                error_message="cannot start",
            ),
        ]
    )

    result = detector.detect_llama_cpp(adapter=adapter, system_name="Linux")

    assert result.found is True
    assert result.executable == "/bad/llama-cli"
    assert result.runnable is False
    assert result.probe_error_type == ErrorType.RUNTIME_NOT_FOUND


def test_explicit_executable_check_requires_execute_bit_on_posix(tmp_path: Path) -> None:
    path = tmp_path / "llama-cli"
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o644)

    assert detector._is_executable_file(path, "Linux") is False

    path.chmod(0o755)

    assert detector._is_executable_file(path, "Linux") is True


def test_detection_model_rejects_verified_without_version() -> None:
    with pytest.raises(ValidationError):
        LlamaCppDetectionResult(
            found=True,
            executable="/bin/llama-cli",
            source="path",
            candidate_name="llama-cli",
            runnable=True,
            version=None,
            verified_eligible=True,
        )


def test_detection_model_rejects_version_without_runnable_state() -> None:
    with pytest.raises(ValidationError, match="runnable"):
        LlamaCppDetectionResult(
            found=True,
            executable="/bin/llama-cli",
            source="path",
            candidate_name="llama-cli",
            runnable=False,
            version="b9001",
            version_command=["/bin/llama-cli", "--version"],
            verified_eligible=False,
        )


def test_detection_model_rejects_version_command_without_version() -> None:
    with pytest.raises(ValidationError, match="version_command"):
        LlamaCppDetectionResult(
            found=True,
            executable="/bin/llama-cli",
            source="path",
            candidate_name="llama-cli",
            runnable=True,
            version_command=["/bin/llama-cli", "--version"],
            verified_eligible=False,
        )
