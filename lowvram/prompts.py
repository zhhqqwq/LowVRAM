"""Versioned Standard Prompt loading and integrity verification."""

import hashlib
import sys
from pathlib import Path

from lowvram.models.prompt import BenchmarkPrompt

DEFAULT_PROMPT_VERSION = "v1"
_PROMPT_SHA256 = {
    "v1": "83192002c73012391d251db70fb58dac2eb8ada79e9b56a8d92ac0f4e76eb75d",
}


class BenchmarkPromptError(RuntimeError):
    """Base class for Standard Prompt loading errors."""


class UnsupportedPromptVersionError(BenchmarkPromptError):
    """Raised when a caller requests an unregistered prompt version."""


class PromptNotFoundError(BenchmarkPromptError):
    """Raised when the registered prompt file cannot be located."""


class PromptIntegrityError(BenchmarkPromptError):
    """Raised when prompt bytes no longer match the registered version digest."""


def _default_prompt_directories() -> tuple[Path, ...]:
    """Return source-tree and installed-data prompt locations."""
    source_tree = Path(__file__).resolve().parent.parent / "benchmark_prompts"
    installed_data = Path(sys.prefix) / "share" / "lowvram" / "benchmark_prompts"
    return (source_tree, installed_data)


def _resolve_prompt_path(version: str, prompt_dir: Path | None) -> Path:
    """Resolve a supported prompt version without accepting silent fallbacks."""
    filename = f"{version}.txt"
    if prompt_dir is not None:
        path = prompt_dir / filename
        if path.is_file():
            return path
        raise PromptNotFoundError(f"benchmark prompt file not found for version {version}")

    for directory in _default_prompt_directories():
        path = directory / filename
        if path.is_file():
            return path

    raise PromptNotFoundError(f"benchmark prompt file not found for version {version}")


def load_benchmark_prompt(
    version: str = DEFAULT_PROMPT_VERSION,
    *,
    prompt_dir: Path | None = None,
) -> BenchmarkPrompt:
    """Load one registered benchmark prompt and verify its exact SHA-256 digest."""
    expected_digest = _PROMPT_SHA256.get(version)
    if expected_digest is None:
        raise UnsupportedPromptVersionError(f"unsupported benchmark prompt version: {version}")

    path = _resolve_prompt_path(version, prompt_dir)
    content = path.read_text(encoding="utf-8")
    actual_digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if actual_digest != expected_digest:
        raise PromptIntegrityError(
            f"benchmark prompt integrity check failed for version {version}"
        )

    return BenchmarkPrompt(
        prompt_version=version,
        content=content,
        sha256=actual_digest,
    )
