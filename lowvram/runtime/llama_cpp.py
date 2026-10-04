"""Initial llama.cpp runtime adapter."""

from lowvram.runtime.adapter import SubprocessRuntimeAdapter


class LlamaCppRuntimeAdapter(SubprocessRuntimeAdapter):
    """Execute an explicitly supplied llama.cpp executable and argv."""

    @property
    def runtime_name(self) -> str:
        """Return the stable runtime identifier used by LowVRAM."""
        return "llama.cpp"
