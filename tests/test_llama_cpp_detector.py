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
        adapter=adapter,  # type: ignore[arg-type]
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


def test_path_search_prefers_llama_cli(monkeypatch) -> None:
    searched: list[str] = []

    def fake_which(name: str) -> str | None:
        searched.append(name)
        if name == "llama-cli":
            return "/usr/local/bin/llama-cli"
        return None

    adapter = FakeAdapter([_execution_result(success=True, stdout="version: 7000\n")])
    monkeypatch.setattr(detector.shutil, "which", fake_which)

    result = detector.detect_llama_cpp(
        adapter=adapter,  # type: ignore[arg-type]
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
        adapter=adapter,  # type: ignore[arg-type]
        system_name="Linux",
    )

    assert result.found is True
    assert result.candidate_name == "main"
    assert result.version == "b4321"
    assert result.verified_eligible is True


def test_version_probe_falls_back_to_positional_version(monkeypatch) -> None:
    adapter = FakeAdapter(
        [
            _execution_result(success=True, stdout="no parseable version here\n"),
            _execution_result(success=True, stdout="version: b7555\n"),
        ]
    )
    monkeypatch.setattr(detector.shutil, "which", lambda name: "/bin/llama-cli")

    result = detector.detect_llama_cpp(
        adapter=adapter,  # type: ignore[arg-type]
        system_name="Linux",
    )

    assert result.version == "b7555"
    assert result.version_command == ["/bin/llama-cli", "version"]
    assert [request.arguments for request in adapter.requests] == [
        ["--version"],
        ["version"],
    ]


def test_unrecognized_version_is_found_but_not_verified(monkeypatch) -> None:
    adapter = FakeAdapter(
        [
            _execution_result(success=True, stdout="unknown build text\n"),
            _execution_result(success=True, stdout="still unknown\n"),
        ]
    )
    monkeypatch.setattr(detector.shutil, "which", lambda name: "/bin/llama-cli")

    result = detector.detect_llama_cpp(
        adapter=adapter,  # type: ignore[arg-type]
        system_name="Linux",
    )

    assert result.found is True
    assert result.runnable is True
    assert result.version is None
    assert result.verified_eligible is False
    assert "not recognized" in (result.message or "")


def test_not_found_result_when_path_has_no_candidates(monkeypatch) -> None:
    monkeypatch.setattr(detector.shutil, "which", lambda name: None)

    result = detector.detect_llama_cpp(system_name="Linux")

    assert result == LlamaCppDetectionResult(
        found=False,
        probe_error_type=ErrorType.RUNTIME_NOT_FOUND,
        message="llama.cpp executable was not found on PATH",
    )


def test_path_candidate_that_cannot_start_is_skipped(monkeypatch) -> None:
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
                error_type=ErrorType.RUNTIME_NOT_FOUND,
                error_message="cannot start",
            ),
            _execution_result(
                success=False,
                exit_code=None,
                error_type=ErrorType.RUNTIME_NOT_FOUND,
                error_message="cannot start",
            ),
            _execution_result(success=True, stdout="version: 9000\n"),
        ]
    )

    result = detector.detect_llama_cpp(
        adapter=adapter,  # type: ignore[arg-type]
        system_name="Linux",
    )

    assert result.executable == "/good/main"
    assert result.version == "9000"


def test_explicit_executable_check_requires_execute_bit_on_posix(
    tmp_path: Path,
) -> None:
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
