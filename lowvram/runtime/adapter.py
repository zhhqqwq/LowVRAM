"""Runtime adapter abstraction and subprocess implementation."""

from abc import ABC, abstractmethod
import subprocess
import time

import psutil

from lowvram.models.benchmark import ErrorType
from lowvram.models.runtime import RuntimeExecutionRequest, RuntimeExecutionResult

_TERMINATION_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)


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
        except FileNotFoundError:
            return RuntimeExecutionResult(
                runtime_name=self.runtime_name,
                success=False,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - started_at,
                error_type=ErrorType.RUNTIME_NOT_FOUND,
                error_message=f"runtime executable not found: {request.executable}",
            )
        except OSError as exc:
            return RuntimeExecutionResult(
                runtime_name=self.runtime_name,
                success=False,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - started_at,
                error_type=ErrorType.RUNTIME_NOT_FOUND,
                error_message=f"runtime executable could not be started: {exc}",
            )

        try:
            stdout, stderr = process.communicate(timeout=request.timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process.pid)
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
