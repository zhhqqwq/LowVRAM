"""Runtime adapter and detector public API."""

from lowvram.runtime.adapter import RuntimeAdapter, SubprocessRuntimeAdapter
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
    "detect_llama_cpp",
    "llama_cpp_candidate_names",
    "parse_llama_cpp_version",
]
