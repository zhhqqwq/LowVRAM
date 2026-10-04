"""P1-07 llama.cpp Command Builder tests."""

import math

import pytest
from pydantic import ValidationError

from lowvram.models.command import LlamaCppCommand, LlamaCppCommandRequest
from lowvram.runtime import (
    LlamaCppRuntimeAdapter,
    build_llama_cpp_command,
    to_runtime_execution_request,
)


def _request(**overrides: object) -> LlamaCppCommandRequest:
    values: dict[str, object] = {
        "executable": "/opt/llama.cpp/llama-cli",
        "model_path": "/models/example model.gguf",
        "context_length": 4096,
        "threads": 8,
        "gpu_layers": 32,
        "batch_size": 512,
        "temperature": 0.8,
        "seed": 42,
        "extra_args": (),
    }
    values.update(overrides)
    return LlamaCppCommandRequest.model_validate(values)


def test_builder_maps_required_parameters_in_deterministic_order() -> None:
    command = build_llama_cpp_command(_request())

    assert command.argv == (
        "/opt/llama.cpp/llama-cli",
        "--model",
        "/models/example model.gguf",
        "--ctx-size",
        "4096",
        "--threads",
        "8",
        "--gpu-layers",
        "32",
        "--batch-size",
        "512",
        "--temp",
        "0.8",
        "--seed",
        "42",
    )
    assert command.arguments == command.argv[1:]


def test_builder_is_deterministic_for_identical_input() -> None:
    request = _request(extra_args=("--no-display-prompt", "--log-disable"))

    first = build_llama_cpp_command(request)
    second = build_llama_cpp_command(request)

    assert first == second
    assert first.argv == second.argv


def test_builder_preserves_model_and_executable_paths_as_single_argv_items() -> None:
    request = _request(
        executable=r"C:\Program Files\llama.cpp\llama-cli.exe",
        model_path=r"D:\AI Models\Qwen 3.gguf",
    )

    command = build_llama_cpp_command(request)

    assert command.argv[0] == r"C:\Program Files\llama.cpp\llama-cli.exe"
    assert command.argv[2] == r"D:\AI Models\Qwen 3.gguf"


def test_builder_preserves_extra_args_without_shell_splitting() -> None:
    request = _request(
        extra_args=(
            "--prompt",
            "hello; echo not-a-shell",
            "--custom-value=alpha beta",
        )
    )

    command = build_llama_cpp_command(request)

    assert command.argv[-3:] == (
        "--prompt",
        "hello; echo not-a-shell",
        "--custom-value=alpha beta",
    )


def test_builder_appends_extra_args_after_managed_arguments() -> None:
    command = build_llama_cpp_command(
        _request(extra_args=("--top-k", "20", "--top-p", "0.9"))
    )

    assert command.argv[-4:] == ("--top-k", "20", "--top-p", "0.9")


def test_unrelated_long_option_with_managed_prefix_is_not_false_positive() -> None:
    request = _request(extra_args=("--threads-batch", "4"))

    command = build_llama_cpp_command(request)

    assert command.argv[-2:] == ("--threads-batch", "4")


@pytest.mark.parametrize(
    "conflicting_arg",
    [
        "--model",
        "--model=other.gguf",
        "-m",
        "--ctx-size",
        "-c",
        "--threads=16",
        "-t",
        "--gpu-layers",
        "--n-gpu-layers=99",
        "-ngl",
        "--batch-size",
        "-b",
        "--temp=1.0",
        "--temperature",
        "--seed",
        "-s",
    ],
)
def test_request_rejects_extra_args_that_override_managed_flags(
    conflicting_arg: str,
) -> None:
    with pytest.raises(ValidationError, match="cannot override managed"):
        _request(extra_args=(conflicting_arg,))


def test_request_cannot_be_mutated_to_bypass_managed_flag_validation() -> None:
    request = _request(extra_args=("--top-k", "20"))

    with pytest.raises(ValidationError, match="cannot override managed"):
        request.extra_args += ("--model=other.gguf",)


def test_cpu_only_gpu_layers_zero_is_valid() -> None:
    command = build_llama_cpp_command(_request(gpu_layers=0))

    index = command.argv.index("--gpu-layers")
    assert command.argv[index + 1] == "0"


def test_negative_gpu_layers_are_invalid() -> None:
    with pytest.raises(ValidationError):
        _request(gpu_layers=-1)


@pytest.mark.parametrize("temperature", [math.inf, -math.inf, math.nan])
def test_non_finite_temperature_is_invalid(temperature: float) -> None:
    with pytest.raises(ValidationError):
        _request(temperature=temperature)


def test_zero_temperature_and_negative_seed_are_preserved() -> None:
    command = build_llama_cpp_command(_request(temperature=0.0, seed=-1))

    temperature_index = command.argv.index("--temp")
    seed_index = command.argv.index("--seed")
    assert command.argv[temperature_index + 1] == "0.0"
    assert command.argv[seed_index + 1] == "-1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("executable", "llama\x00-cli"),
        ("model_path", "model\x00.gguf"),
        ("extra_args", ("--prompt", "bad\x00value")),
    ],
)
def test_request_rejects_nul_bytes(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="NUL"):
        _request(**{field: value})


def test_command_record_is_immutable_after_validation() -> None:
    command = build_llama_cpp_command(_request())

    with pytest.raises(ValidationError):
        command.arguments += ("--top-k", "20")


def test_command_model_rejects_inconsistent_full_argv() -> None:
    with pytest.raises(ValidationError, match="argv must equal"):
        LlamaCppCommand(
            executable="llama-cli",
            arguments=("--model", "model.gguf"),
            argv=("llama-cli", "--model", "different.gguf"),
        )


def test_runtime_request_bridge_preserves_exact_command() -> None:
    command = build_llama_cpp_command(
        _request(
            executable=r"C:\Program Files\llama.cpp\llama-cli.exe",
            model_path=r"D:\AI Models\model.gguf",
            extra_args=("--top-k", "20"),
        )
    )

    request = to_runtime_execution_request(command, timeout_seconds=45)
    adapter = LlamaCppRuntimeAdapter()

    assert tuple(adapter.build_command(request)) == command.argv
    assert request.timeout_seconds == 45
    assert request.executable == command.executable
    assert tuple(request.arguments) == command.arguments
