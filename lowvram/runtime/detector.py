"""P1-06 llama.cpp executable discovery and version probing."""

import os
import platform
import re
import shutil
from pathlib import Path
from typing import Literal

from lowvram.models.benchmark import ErrorType
from lowvram.models.detection import LlamaCppDetectionResult
from lowvram.models.runtime import RuntimeExecutionRequest, RuntimeExecutionResult
from lowvram.runtime.llama_cpp import LlamaCppRuntimeAdapter

_VERSION_PATTERNS = (
    re.compile(
        r"(?im)^\s*(?:llama\.cpp\s+)?version\s*[:=]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9._+-]*)"
    ),
    re.compile(r"(?im)^\s*build(?:\s+number)?\s*[:=]\s*([0-9]+)\b"),
    re.compile(r"(?i)\bb([0-9]{3,})\b"),
)
_VERSION_PROBES = (["--version"], ["version"])


def llama_cpp_candidate_names(system_name: str | None = None) -> tuple[str, ...]:
    """Return preferred modern and legacy executable names for the platform."""
    current = (system_name or platform.system()).lower()
    if current == "windows":
        return ("llama-cli.exe", "llama-cli", "main.exe", "main")
    return ("llama-cli", "main")


def parse_llama_cpp_version(output: str) -> str | None:
    """Parse common llama.cpp version/build forms from probe output."""
    for index, pattern in enumerate(_VERSION_PATTERNS):
        match = pattern.search(output)
        if match is None:
            continue
        value = match.group(1)
        if index in {1, 2}:
            return f"b{value}"
        return value
    return None


def _is_executable_file(path: Path, system_name: str | None = None) -> bool:
    """Check that an explicit path is a runnable file on Windows or POSIX."""
    if not path.is_file():
        return False
    current = (system_name or platform.system()).lower()
    if current == "windows":
        return True
    return os.access(path, os.X_OK)


def _probe_output(result: RuntimeExecutionResult) -> str:
    """Combine captured streams for version parsing without shell processing."""
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def _probe_candidate(
    executable: str,
    source: Literal["explicit_path", "path"],
    candidate_name: str,
    adapter: LlamaCppRuntimeAdapter,
) -> LlamaCppDetectionResult:
    """Probe one discovered executable with --version then legacy version."""
    saw_started_process = False
    last_error_type: ErrorType | None = None
    last_message: str | None = None

    for arguments in _VERSION_PROBES:
        result = adapter.execute(
            RuntimeExecutionRequest(
                executable=executable,
                arguments=arguments,
                timeout_seconds=5.0,
            )
        )

        if result.error_type != ErrorType.RUNTIME_NOT_FOUND:
            saw_started_process = True

        if result.success:
            version = parse_llama_cpp_version(_probe_output(result))
            if version is not None:
                return LlamaCppDetectionResult(
                    found=True,
                    executable=executable,
                    source=source,
                    candidate_name=candidate_name,
                    runnable=True,
                    version=version,
                    version_command=[executable, *arguments],
                    verified_eligible=True,
                    message="llama.cpp executable and version detected",
                )
            last_error_type = None
            last_message = "llama.cpp executable ran but version output was not recognized"
            continue

        last_error_type = result.error_type
        last_message = result.error_message

    return LlamaCppDetectionResult(
        found=True,
        executable=executable,
        source=source,
        candidate_name=candidate_name,
        runnable=saw_started_process,
        version=None,
        version_command=None,
        verified_eligible=False,
        probe_error_type=last_error_type,
        message=last_message or "llama.cpp version could not be detected",
    )


def detect_llama_cpp(
    explicit_path: str | None = None,
    *,
    adapter: LlamaCppRuntimeAdapter | None = None,
    system_name: str | None = None,
) -> LlamaCppDetectionResult:
    """Detect llama.cpp from an explicit path or the current PATH."""
    runtime_adapter = adapter or LlamaCppRuntimeAdapter()

    if explicit_path is not None:
        path = Path(explicit_path)
        if not _is_executable_file(path, system_name):
            return LlamaCppDetectionResult(
                found=False,
                probe_error_type=ErrorType.RUNTIME_NOT_FOUND,
                message=f"llama.cpp executable path is not runnable: {explicit_path}",
            )
        return _probe_candidate(
            executable=str(path),
            source="explicit_path",
            candidate_name=path.name,
            adapter=runtime_adapter,
        )

    for candidate_name in llama_cpp_candidate_names(system_name):
        executable = shutil.which(candidate_name)
        if executable is None:
            continue

        result = _probe_candidate(
            executable=executable,
            source="path",
            candidate_name=candidate_name,
            adapter=runtime_adapter,
        )

        if result.runnable or result.version is not None:
            return result

    return LlamaCppDetectionResult(
        found=False,
        probe_error_type=ErrorType.RUNTIME_NOT_FOUND,
        message="llama.cpp executable was not found on PATH",
    )
