"""P1-08 Standard Prompt tests."""

import hashlib
from importlib.resources import files
from pathlib import Path

import pytest
from pydantic import ValidationError

from lowvram.models.prompt import BenchmarkPrompt
from lowvram.prompts import (
    DEFAULT_PROMPT_VERSION,
    PromptIntegrityError,
    PromptNotFoundError,
    UnsupportedPromptVersionError,
    load_benchmark_prompt,
)

EXPECTED_V1_SHA256 = "83192002c73012391d251db70fb58dac2eb8ada79e9b56a8d92ac0f4e76eb75d"
ROOT_PROMPT = Path(__file__).parent.parent / "benchmark_prompts" / "v1.txt"


def test_v1_prompt_loads_from_package_resource_with_pinned_version_and_hash() -> None:
    prompt = load_benchmark_prompt()

    assert DEFAULT_PROMPT_VERSION == "v1"
    assert prompt.prompt_version == "v1"
    assert prompt.sha256 == EXPECTED_V1_SHA256
    assert prompt.content.startswith("LowVRAM Benchmark Prompt v1\n")
    assert len(prompt.content) > 2000


def test_root_review_copy_matches_packaged_resource_exactly() -> None:
    root_bytes = ROOT_PROMPT.read_bytes()
    packaged_bytes = files("lowvram.benchmark_prompts").joinpath("v1.txt").read_bytes()

    assert root_bytes == packaged_bytes
    assert hashlib.sha256(root_bytes).hexdigest() == EXPECTED_V1_SHA256


def test_v1_prompt_is_stable_across_repeated_loads() -> None:
    first = load_benchmark_prompt()
    second = load_benchmark_prompt()

    assert first == second
    assert first.content == second.content


def test_v1_prompt_is_self_contained_and_offline() -> None:
    prompt = load_benchmark_prompt()

    assert "http://" not in prompt.content
    assert "https://" not in prompt.content
    assert "Use only the information in this prompt." in prompt.content
    assert "personal data" in prompt.content
    assert "The readings are synthetic" in prompt.content


def test_prompt_model_is_immutable() -> None:
    prompt = load_benchmark_prompt()

    with pytest.raises(ValidationError):
        prompt.content = "mutated"


def test_missing_prompt_file_raises_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(PromptNotFoundError, match="version v1"):
        load_benchmark_prompt(prompt_dir=tmp_path)


def test_modified_prompt_file_fails_integrity_check(tmp_path: Path) -> None:
    modified_path = tmp_path / "v1.txt"
    modified_path.write_bytes(ROOT_PROMPT.read_bytes() + b"tampered\n")

    with pytest.raises(PromptIntegrityError, match="version v1"):
        load_benchmark_prompt(prompt_dir=tmp_path)


def test_line_ending_change_is_detected_as_integrity_drift(tmp_path: Path) -> None:
    original = ROOT_PROMPT.read_bytes()
    modified_path = tmp_path / "v1.txt"
    modified_path.write_bytes(original.replace(b"\n", b"\r\n"))

    with pytest.raises(PromptIntegrityError, match="version v1"):
        load_benchmark_prompt(prompt_dir=tmp_path)


def test_unknown_prompt_version_is_rejected_before_file_lookup(tmp_path: Path) -> None:
    (tmp_path / "v2.txt").write_text("unregistered", encoding="utf-8")

    with pytest.raises(UnsupportedPromptVersionError, match="v2"):
        load_benchmark_prompt("v2", prompt_dir=tmp_path)


def test_prompt_model_rejects_mismatched_digest() -> None:
    with pytest.raises(ValidationError, match="sha256 does not match"):
        BenchmarkPrompt(
            prompt_version="v1",
            content="different content",
            sha256=EXPECTED_V1_SHA256,
        )
