"""Runtime adapter, detector, and command-builder public API."""

from lowvram.runtime.adapter import RuntimeAdapter, SubprocessRuntimeAdapter
from lowvram.runtime.command_builder import (
    build_llama_cpp_command,
    to_runtime_execution_request,
)
from lowvram.runtime.detector import (
    detect_llama_cpp,
    llama_cpp_candidate_names,
    parse_llama_cpp_version,
)
from lowvram.runtime.llama_cpp import LlamaCppRuntimeAdapter

__all__ = [
    "LlamaCppRuntimeAdapter",
    "RuntimeAdapter",
    "SubprocessRuntimeAdapter",
    "build_llama_cpp_command",
    "detect_llama_cpp",
    "llama_cpp_candidate_names",
    "parse_llama_cpp_version",
    "to_runtime_execution_request",
]
