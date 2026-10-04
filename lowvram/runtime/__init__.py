"""Runtime adapter, detector, command-builder, parser, and orchestrator API."""

from lowvram.runtime.adapter import (
    RuntimeAdapter,
    RuntimeProcessSession,
    RuntimeSessionStateError,
    RuntimeSpawnError,
    SubprocessRuntimeAdapter,
)
from lowvram.runtime.command_builder import (
    build_llama_cpp_command,
    to_runtime_execution_request,
)
from lowvram.runtime.detector import (
    detect_llama_cpp,
    llama_cpp_candidate_names,
    parse_llama_cpp_version,
)
from lowvram.runtime.failure_classifier import classify_failure
from lowvram.runtime.llama_cpp import LlamaCppRuntimeAdapter
from lowvram.runtime.orchestrator import (
    BenchmarkOrchestrator,
    save_benchmark_orchestration_record,
)
from lowvram.runtime.output_parser import parse_llama_cpp_output
from lowvram.runtime.preparation import prepare_llama_cpp_benchmark

__all__ = [
    "BenchmarkOrchestrator",
    "LlamaCppRuntimeAdapter",
    "RuntimeAdapter",
    "RuntimeProcessSession",
    "RuntimeSessionStateError",
    "RuntimeSpawnError",
    "SubprocessRuntimeAdapter",
    "build_llama_cpp_command",
    "classify_failure",
    "detect_llama_cpp",
    "llama_cpp_candidate_names",
    "parse_llama_cpp_output",
    "parse_llama_cpp_version",
    "prepare_llama_cpp_benchmark",
    "save_benchmark_orchestration_record",
    "to_runtime_execution_request",
]
