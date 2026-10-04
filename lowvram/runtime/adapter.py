"""Runtime adapter abstraction and subprocess implementation."""

import errno
import subprocess
import time
from abc import ABC, abstractmethod

import psutil

from lowvram.models.benchmark import ErrorType
from lowvram.models.runtime import RuntimeExecutionRequest, RuntimeExecutionResult

_TERMINATION_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)
_RUNTIME_UNAVAILABLE_ERRNOS = {
    errno.ENOENT,
    errno.ENOTDIR,
    errno.EACCES,
    errno.ENOEXEC,
}
_RUNTIME_UNAVAILABLE_WINERRORS = {
    2,    # ERROR_FILE_NOT_FOUND
    3,    # ERROR_PATH_NOT_FOUND
    5,    # ERROR_ACCESS_DENIED
    193,  # ERROR_BAD_EXE_FORMAT
}


class RuntimeAdapter(ABC):
    """Abstract interface for executing one external model runtime."""

    @property
    @abstractmethod
    def runtime_name(self) -> str:
        """Return the stable LowVRAM runtime name."""

    @abstractmethod
    def build_command(self, request: RuntimeExecutionRequest) -> list[str]:
        """Build the argv passed to the operating system without a shell."""

    @abstractmethod
    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Execute one runtime process and return a structured outcome."""


class SubprocessRuntimeAdapter(RuntimeAdapter):
    """Base adapter for runtimes exposed as local executables."""

    def build_command(self, request: RuntimeExecutionRequest) -> list[str]:
        """Build a shell-free command from an explicit executable plus arguments."""
        return [request.executable, *request.arguments]

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Run one process, capture output, and normalize common failures."""
        command = self.build_command(request)
        started_at = time.monotonic()

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
        except OSError as exc:
            error_type = _classify_start_error(exc)
            if error_type == ErrorType.RUNTIME_NOT_FOUND:
                error_message = (
                    f"runtime executable unavailable: {request.executable}: {exc}"
                )
            else:
                error_message = f"runtime process could not be started: {exc}"
            return RuntimeExecutionResult(
                runtime_name=self.runtime_name,
                success=False,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - started_at,
                error_type=error_type,
                error_message=error_message,
            )

        try:
            stdout, stderr = process.communicate(timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process.pid)
            _kill_root_fallback(process)
            stdout, stderr = process.communicate()
            return RuntimeExecutionResult(
                runtime_name=self.runtime_name,
                success=False,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=time.monotonic() - started_at,
                error_type=ErrorType.TIMEOUT,
                error_message=(
                    f"runtime exceeded timeout of {request.timeout_seconds:g} seconds"
                ),
            )

        duration = time.monotonic() - started_at
        if process.returncode == 0:
            return RuntimeExecutionResult(
                runtime_name=self.runtime_name,
                success=True,
                exit_code=0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        return RuntimeExecutionResult(
            runtime_name=self.runtime_name,
            success=False,
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            error_type=ErrorType.PROCESS_CRASH,
            error_message=f"runtime process exited with code {process.returncode}",
        )


def _classify_start_error(exc: OSError) -> ErrorType:
    """Separate unavailable executables from unrelated OS-level start failures."""
    winerror = getattr(exc, "winerror", None)
    if (
        isinstance(exc, FileNotFoundError | PermissionError)
        or exc.errno in _RUNTIME_UNAVAILABLE_ERRNOS
        or winerror in _RUNTIME_UNAVAILABLE_WINERRORS
    ):
        return ErrorType.RUNTIME_NOT_FOUND
    return ErrorType.UNKNOWN


def _kill_root_fallback(process: subprocess.Popen[str]) -> None:
    """Ensure the direct child is killed even when psutil tree handling cannot reach it."""
    if process.poll() is not None:
        return
    try:
        process.kill()
    except OSError:
        return


def _terminate_process_tree(root_pid: int) -> None:
    """Terminate a timed-out process and descendants, escalating to kill if needed."""
    try:
        root = psutil.Process(root_pid)
    except _TERMINATION_ERRORS:
        return

    try:
        descendants = root.children(recursive=True)
    except _TERMINATION_ERRORS:
        descendants = []

    processes = [*descendants, root]
    for process in processes:
        try:
            process.terminate()
        except _TERMINATION_ERRORS:
            continue

    _, alive = psutil.wait_procs(processes, timeout=1.0)
    for process in alive:
        try:
            process.kill()
        except _TERMINATION_ERRORS:
            continue

    if alive:
        psutil.wait_procs(alive, timeout=1.0)
