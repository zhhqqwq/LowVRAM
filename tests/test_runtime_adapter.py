"""P1-05 Runtime Adapter tests."""

import errno
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

import lowvram.runtime.adapter as runtime_adapter
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


def test_runtime_adapter_passes_argv_to_popen_with_shell_false(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 1234
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            captured["timeout"] = timeout
            return "stdout", "stderr"

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(runtime_adapter.subprocess, "Popen", fake_popen)

    adapter = LlamaCppRuntimeAdapter()
    request = RuntimeExecutionRequest(
        executable=r"C:\Program Files\llama.cpp\llama-cli.exe",
        arguments=["--model", r"D:\AI Models\model.gguf", "--prompt", "a;b"],
        timeout_seconds=7,
    )

    result = adapter.execute(request)

    assert result.success is True
    assert captured["command"] == [
        r"C:\Program Files\llama.cpp\llama-cli.exe",
        "--model",
        r"D:\AI Models\model.gguf",
        "--prompt",
        "a;b",
    ]
    kwargs = captured["kwargs"]
    assert kwargs["shell"] is False
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert captured["timeout"] == 7


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


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOEXEC, "bad executable format"),
    ],
)
def test_runtime_adapter_classifies_unusable_executable_as_runtime_not_found(
    monkeypatch,
    error: OSError,
) -> None:
    monkeypatch.setattr(
        runtime_adapter.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    result = LlamaCppRuntimeAdapter().execute(
        RuntimeExecutionRequest(executable="llama-cli")
    )

    assert result.success is False
    assert result.exit_code is None
    assert result.error_type == ErrorType.RUNTIME_NOT_FOUND


def test_runtime_adapter_does_not_misclassify_unrelated_os_error(monkeypatch) -> None:
    error = OSError(errno.EMFILE, "too many open files")
    monkeypatch.setattr(
        runtime_adapter.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    result = LlamaCppRuntimeAdapter().execute(
        RuntimeExecutionRequest(executable="llama-cli")
    )

    assert result.success is False
    assert result.exit_code is None
    assert result.error_type == ErrorType.UNKNOWN
    assert "too many open files" in (result.error_message or "")


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


def test_timeout_uses_direct_root_kill_if_psutil_tree_cleanup_cannot_reach_root(
    monkeypatch,
) -> None:
    class FakeProcess:
        pid = 4321
        returncode: int | None = None
        killed = False
        communicate_calls = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(cmd=["llama-cli"], timeout=timeout or 0)
            return "partial-output", ""

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    fake_process = FakeProcess()
    monkeypatch.setattr(runtime_adapter.subprocess, "Popen", lambda *args, **kwargs: fake_process)
    monkeypatch.setattr(runtime_adapter, "_terminate_process_tree", lambda pid: None)

    result = LlamaCppRuntimeAdapter().execute(
        RuntimeExecutionRequest(
            executable="llama-cli",
            timeout_seconds=0.01,
        )
    )

    assert fake_process.killed is True
    assert result.error_type == ErrorType.TIMEOUT
    assert result.exit_code is None
    assert result.stdout == "partial-output"


def test_terminate_process_tree_terminates_children_before_root_and_kills_survivors(
    monkeypatch,
) -> None:
    events: list[tuple[str, int]] = []

    class FakePsutilProcess:
        def __init__(self, pid: int, children: list["FakePsutilProcess"] | None = None) -> None:
            self.pid = pid
            self._children = children or []

        def children(self, recursive: bool = False) -> list["FakePsutilProcess"]:
            assert recursive is True
            return list(self._children)

        def terminate(self) -> None:
            events.append(("terminate", self.pid))

        def kill(self) -> None:
            events.append(("kill", self.pid))

    child = FakePsutilProcess(101)
    root = FakePsutilProcess(100, [child])

    monkeypatch.setattr(runtime_adapter.psutil, "Process", lambda pid: root)

    wait_calls = 0

    def fake_wait_procs(
        processes: list[FakePsutilProcess],
        timeout: float,
    ) -> tuple[list[FakePsutilProcess], list[FakePsutilProcess]]:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            assert processes == [child, root]
            assert timeout == 1.0
            return [root], [child]
        assert processes == [child]
        return [child], []

    monkeypatch.setattr(runtime_adapter.psutil, "wait_procs", fake_wait_procs)

    runtime_adapter._terminate_process_tree(100)

    assert events == [
        ("terminate", 101),
        ("terminate", 100),
        ("kill", 101),
    ]
    assert wait_calls == 2


def test_runtime_adapter_preserves_arguments_without_shell_expansion() -> None:
    adapter = LlamaCppRuntimeAdapter()
    request = RuntimeExecutionRequest(
        executable=sys.executable,
        arguments=["-c", "import sys; print(sys.argv[1])", "$(echo injected)"],
    )

    result = adapter.execute(request)

    assert result.success is True
    assert result.stdout.strip() == "$(echo injected)"


@pytest.mark.parametrize(
    ("timeout", "arguments"),
    [
        (0, []),
        (-1, []),
        (1, [123]),
    ],
)
def test_runtime_execution_request_rejects_invalid_inputs(
    timeout: float,
    arguments: list[object],
) -> None:
    with pytest.raises(ValidationError):
        RuntimeExecutionRequest(
            executable="llama-cli",
            arguments=arguments,
            timeout_seconds=timeout,
        )


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


def test_runtime_execution_result_rejects_exit_code_for_timeout() -> None:
    with pytest.raises(ValidationError):
        RuntimeExecutionResult(
            runtime_name="llama.cpp",
            success=False,
            exit_code=-9,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            error_type=ErrorType.TIMEOUT,
            error_message="timed out",
        )

def test_spawn_exposes_pid_before_wait_and_preserves_result() -> None:
    adapter = LlamaCppRuntimeAdapter()
    session = adapter.spawn(
        _python_request(
            "import os, time; print(os.getpid(), flush=True); time.sleep(0.05)"
        )
    )

    assert session.pid > 0
    result = session.wait()

    assert result.success is True
    assert result.exit_code == 0
    assert result.stdout.strip() == str(session.pid)


def test_runtime_process_session_rejects_second_wait() -> None:
    adapter = LlamaCppRuntimeAdapter()
    session = adapter.spawn(_python_request("print('done')"))

    result = session.wait()
    assert result.success is True

    with pytest.raises(runtime_adapter.RuntimeSessionStateError, match="already closed"):
        session.wait()


def test_spawn_error_is_structured_before_session_exists(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_adapter.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileNotFoundError(errno.ENOENT, "missing")
        ),
    )

    with pytest.raises(runtime_adapter.RuntimeSpawnError) as raised:
        LlamaCppRuntimeAdapter().spawn(
            RuntimeExecutionRequest(executable="missing-llama-cli")
        )

    assert raised.value.error_type == ErrorType.RUNTIME_NOT_FOUND
    assert "missing-llama-cli" in raised.value.message


def test_spawn_enomem_is_classified_as_out_of_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        runtime_adapter.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError(errno.ENOMEM, "cannot allocate memory")
        ),
    )

    with pytest.raises(runtime_adapter.RuntimeSpawnError) as raised:
        LlamaCppRuntimeAdapter().spawn(
            RuntimeExecutionRequest(executable="llama-cli")
        )

    assert raised.value.error_type == ErrorType.OUT_OF_MEMORY
    assert "memory exhaustion" in raised.value.message
