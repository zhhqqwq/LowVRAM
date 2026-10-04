"""Runtime adapter abstraction, spawn session, and subprocess implementation."""

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


class RuntimeSessionStateError(RuntimeError):
    """Raised when a runtime process session is used after it is closed."""


class RuntimeSpawnError(RuntimeError):
    """Structured process-start failure raised before a runtime session exists."""

    def __init__(
        self,
        *,
        error_type: ErrorType,
        message: str,
        duration_seconds: float,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.duration_seconds = duration_seconds


class RuntimeProcessSession:
    """One already-spawned runtime process whose PID is available before wait()."""

    def __init__(
        self,
        *,
        runtime_name: str,
        request: RuntimeExecutionRequest,
        process: subprocess.Popen[str],
        started_at: float,
    ) -> None:
        self._runtime_name = runtime_name
        self._request = request
        self._process = process
        self._started_at = started_at
        self._closed = False

    @property
    def pid(self) -> int:
        """Return the direct runtime process PID while the session is active."""
        return self._process.pid

    def wait(self) -> RuntimeExecutionResult:
        """Wait for process completion and normalize timeout/exit behavior."""
        if self._closed:
            raise RuntimeSessionStateError("runtime process session is already closed")

        try:
            stdout, stderr = self._process.communicate(
                timeout=self._request.timeout_seconds
            )
        except subprocess.TimeoutExpired:
            _terminate_process_tree(self._process.pid)
            _kill_root_fallback(self._process)
            stdout, stderr = self._process.communicate()
            self._closed = True
            return RuntimeExecutionResult(
                runtime_name=self._runtime_name,
                success=False,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=time.monotonic() - self._started_at,
                error_type=ErrorType.TIMEOUT,
                error_message=(
                    f"runtime exceeded timeout of "
                    f"{self._request.timeout_seconds:g} seconds"
                ),
            )

        self._closed = True
        duration = time.monotonic() - self._started_at
        exit_code = self._process.returncode
        if exit_code == 0:
            return RuntimeExecutionResult(
                runtime_name=self._runtime_name,
                success=True,
                exit_code=0,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        if exit_code is None:
            return RuntimeExecutionResult(
                runtime_name=self._runtime_name,
                success=False,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                error_type=ErrorType.UNKNOWN,
                error_message="runtime process ended without a return code",
            )

        return RuntimeExecutionResult(
            runtime_name=self._runtime_name,
            success=False,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            error_type=ErrorType.PROCESS_CRASH,
            error_message=f"runtime process exited with code {exit_code}",
        )

    def cancel(self) -> None:
        """Best-effort terminate and drain an active session without publishing a result."""
        if self._closed:
            return

        _terminate_process_tree(self._process.pid)
        _kill_root_fallback(self._process)
        try:
            self._process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_root_fallback(self._process)
            self._process.communicate()
        self._closed = True


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

    def spawn(self, request: RuntimeExecutionRequest) -> RuntimeProcessSession:
        """Spawn a runtime and expose its PID before waiting for completion."""
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
                message = f"runtime executable unavailable: {request.executable}: {exc}"
            else:
                message = f"runtime process could not be started: {exc}"
            raise RuntimeSpawnError(
                error_type=error_type,
                message=message,
                duration_seconds=time.monotonic() - started_at,
            ) from exc

        return RuntimeProcessSession(
            runtime_name=self.runtime_name,
            request=request,
            process=process,
            started_at=started_at,
        )

    def execute(self, request: RuntimeExecutionRequest) -> RuntimeExecutionResult:
        """Backward-compatible spawn-and-wait execution helper."""
        try:
            session = self.spawn(request)
        except RuntimeSpawnError as exc:
            return RuntimeExecutionResult(
                runtime_name=self.runtime_name,
                success=False,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=exc.duration_seconds,
                error_type=exc.error_type,
                error_message=exc.message,
            )
        return session.wait()


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
