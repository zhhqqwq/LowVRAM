"""Versioned Standard Prompt loading and integrity verification."""

import hashlib
from importlib.resources import files
from pathlib import Path
from typing import cast

from lowvram.models.prompt import BenchmarkPrompt, PromptVersion

DEFAULT_PROMPT_VERSION: PromptVersion = "v1"
_PROMPT_PACKAGE = "lowvram.benchmark_prompts"
_PROMPT_SHA256: dict[PromptVersion, str] = {
    "v1": "83192002c73012391d251db70fb58dac2eb8ada79e9b56a8d92ac0f4e76eb75d",
}


class BenchmarkPromptError(RuntimeError):
    """Base class for Standard Prompt loading errors."""


class UnsupportedPromptVersionError(BenchmarkPromptError):
    """Raised when a caller requests an unregistered prompt version."""


class PromptNotFoundError(BenchmarkPromptError):
    """Raised when the registered prompt resource cannot be located."""


class PromptIntegrityError(BenchmarkPromptError):
    """Raised when prompt bytes no longer match the registered version digest."""


def _read_prompt_bytes(version: PromptVersion, prompt_dir: Path | None) -> bytes:
    """Read exact prompt bytes from an override directory or package resources."""
    filename = f"{version}.txt"

    if prompt_dir is not None:
        path = prompt_dir / filename
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise PromptNotFoundError(
                f"benchmark prompt file not found for version {version}"
            ) from exc

    resource = files(_PROMPT_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise PromptNotFoundError(
            f"packaged benchmark prompt not found for version {version}"
        )
    try:
        return resource.read_bytes()
    except FileNotFoundError as exc:
        raise PromptNotFoundError(
            f"packaged benchmark prompt not found for version {version}"
        ) from exc


def load_benchmark_prompt(
    version: str = DEFAULT_PROMPT_VERSION,
    *,
    prompt_dir: Path | None = None,
) -> BenchmarkPrompt:
    """Load one registered benchmark prompt and verify exact packaged bytes."""
    if version not in _PROMPT_SHA256:
        raise UnsupportedPromptVersionError(
            f"unsupported benchmark prompt version: {version}"
        )

    prompt_version = cast(PromptVersion, version)
    expected_digest = _PROMPT_SHA256[prompt_version]
    raw = _read_prompt_bytes(prompt_version, prompt_dir)
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != expected_digest:
        raise PromptIntegrityError(
            f"benchmark prompt integrity check failed for version {version}"
        )

    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptIntegrityError(
            f"benchmark prompt is not valid UTF-8 for version {version}"
        ) from exc

    return BenchmarkPrompt(
        prompt_version=prompt_version,
        content=content,
        sha256=actual_digest,
    )
