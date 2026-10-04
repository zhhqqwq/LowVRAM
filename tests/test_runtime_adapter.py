"""P1-05 Runtime Adapter tests."""

import sys

import pytest
from pydantic import ValidationError

from lowvram.models.benchmark import ErrorType
from lowvram.models.runtime import RuntimeExecutionRequest, RuntimeExecutionResult
from lowvram.runtime import LlamaCppRuntimeAdapter


def _python_request(code: str, timeout_seconds: float = 5.0) -> RuntimeExecutionRequest:
    return RuntimeExecutionRequest(
        executable=sys.executable,
        arguments=["-c", code],
        timeout_seconds=timeout_seconds,
    )


def test_llama_cpp_adapter_builds_shell_free_argv() -> None:
    adapter = LlamaCppRuntimeAdapter()
    request = RuntimeExecutionRequest(
        executable="/opt/llama/llama-cli",
        arguments=["-m", "model with spaces.gguf", "--prompt", "hello; echo unsafe"],
    )

    assert adapter.build_command(request) == [
        "/opt/llama/llama-cli",
        "-m",
        "model with spaces.gguf",
        "--prompt",
        "hello; echo unsafe",
    ]


def test_runtime_adapter_captures_stdout_stderr_and_exit_code() -> None:
    adapter = LlamaCppRuntimeAdapter()
    request = _python_request(
        "import sys; print('stdout-line'); print('stderr-line', file=sys.stderr)"
    )

    result = adapter.execute(request)

    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "stdout-line"
    assert result.stderr.strip() == "stderr-line"
    assert result.error_type is None
    assert result.duration_seconds >= 0


def test_runtime_adapter_classifies_nonzero_exit_as_process_crash() -> None:
    adapter = LlamaCppRuntimeAdapter()

    result = adapter.execute(_python_request("import sys; sys.exit(7)"))

    assert result.success is False
    assert result.exit_code == 7
    assert result.error_type == ErrorType.PROCESS_CRASH
    assert "code 7" in (result.error_message or "")


def test_runtime_adapter_classifies_missing_executable() -> None:
    adapter = LlamaCppRuntimeAdapter()
    request = RuntimeExecutionRequest(
        executable="lowvram-this-runtime-does-not-exist-7d1bbab5",
        arguments=[],
    )

    result = adapter.execute(request)

    assert result.success is False
    assert result.exit_code is None
    assert result.error_type == ErrorType.RUNTIME_NOT_FOUND
    assert request.executable in (result.error_message or "")


def test_runtime_adapter_times_out_and_returns_partial_output() -> None:
    adapter = LlamaCppRuntimeAdapter()
    request = _python_request(
        "import time; print('before-timeout', flush=True); time.sleep(5)",
        timeout_seconds=0.1,
    )

    result = adapter.execute(request)

    assert result.success is False
    assert result.exit_code is None
    assert result.error_type == ErrorType.TIMEOUT
    assert "before-timeout" in result.stdout
    assert result.duration_seconds < 5


def test_runtime_adapter_preserves_arguments_without_shell_expansion() -> None:
    adapter = LlamaCppRuntimeAdapter()
    request = RuntimeExecutionRequest(
        executable=sys.executable,
        arguments=["-c", "import sys; print(sys.argv[1])", "$(echo injected)"],
    )

    result = adapter.execute(request)

    assert result.success is True
    assert result.stdout.strip() == "$(echo injected)"


def test_runtime_execution_result_rejects_inconsistent_success() -> None:
    with pytest.raises(ValidationError):
        RuntimeExecutionResult(
            runtime_name="llama.cpp",
            success=True,
            exit_code=1,
            stdout="",
            stderr="",
            duration_seconds=0.1,
        )


def test_runtime_execution_result_requires_crash_exit_code() -> None:
    with pytest.raises(ValidationError):
        RuntimeExecutionResult(
            runtime_name="llama.cpp",
            success=False,
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            error_type=ErrorType.PROCESS_CRASH,
            error_message="crashed",
        )
