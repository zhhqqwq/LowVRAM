"""P1-09 llama.cpp Output Parser tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from lowvram.models.benchmark import ErrorType
from lowvram.models.output_parser import (
    LlamaCppOutputParseResult,
    LlamaCppTimingMetrics,
)
from lowvram.runtime.output_parser import parse_llama_cpp_output

FIXTURES = Path(__file__).parent / "fixtures" / "llama_cpp_timings"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_classic_llama_print_timings_from_stderr() -> None:
    result = parse_llama_cpp_output(
        stdout="generated model text\n",
        stderr=_fixture("valid_classic.txt"),
    )

    assert result.success is True
    assert result.error_type is None
    assert result.metrics is not None
    assert result.metrics.load_time_seconds == pytest.approx(0.57915)
    assert result.metrics.prompt_eval_time_seconds == pytest.approx(0.65563)
    assert result.metrics.prompt_tokens_per_second == pytest.approx(15.25)
    assert result.metrics.eval_time_seconds == pytest.approx(2.18097)
    assert result.metrics.generation_tokens_per_second == pytest.approx(12.38)


def test_parse_current_llama_perf_context_prefix() -> None:
    result = parse_llama_cpp_output(
        stdout=_fixture("valid_perf_context.txt"),
        stderr="",
    )

    assert result.success is True
    assert result.metrics is not None
    assert result.metrics.load_time_seconds is None
    assert result.metrics.prompt_eval_time_seconds == pytest.approx(3.84367)
    assert result.metrics.prompt_tokens_per_second == 51.25
    assert result.metrics.eval_time_seconds == pytest.approx(1.68613)
    assert result.metrics.generation_tokens_per_second == 18.39


def test_parse_generation_eval_label_variant() -> None:
    result = parse_llama_cpp_output(
        stdout="",
        stderr=(
            "INFO prompt eval time = 15.73 ms / 12 tokens "
            "(1.31 ms per token, 762.68 tokens per second)\n"
            "INFO generation eval time = 2465.50 ms / 512 runs "
            "(4.82 ms per token, 207.67 tokens per second)\n"
        ),
    )

    assert result.success is True
    assert result.metrics is not None
    assert result.metrics.generation_tokens_per_second == 207.67


def test_stdout_and_stderr_are_both_inputs() -> None:
    result = parse_llama_cpp_output(
        stdout=(
            "llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens "
            "(10.00 ms per token, 100.00 tokens per second)\n"
        ),
        stderr=(
            "llama_perf_context_print: eval time = 200.00 ms / 20 tokens "
            "(10.00 ms per token, 100.00 tokens per second)\n"
        ),
    )

    assert result.success is True
    assert result.metrics is not None
    assert result.metrics.prompt_eval_time_seconds == 0.1
    assert result.metrics.eval_time_seconds == 0.2


def test_duplicate_timing_blocks_choose_last_complete_block() -> None:
    result = parse_llama_cpp_output(
        stdout="",
        stderr=_fixture("valid_duplicate_blocks.txt"),
    )

    assert result.success is True
    assert result.metrics == LlamaCppTimingMetrics(
        load_time_seconds=None,
        prompt_eval_time_seconds=0.8,
        prompt_tokens_per_second=50.0,
        eval_time_seconds=2.4,
        generation_tokens_per_second=20.0,
    )


def test_trailing_partial_block_fails_instead_of_using_stale_metrics() -> None:
    result = parse_llama_cpp_output(
        stdout="",
        stderr=_fixture("invalid_trailing_partial.txt"),
    )

    assert result.success is False
    assert result.metrics is None
    assert result.error_type == ErrorType.PARSE_FAILED


@pytest.mark.parametrize(
    "name",
    [
        "invalid_prompt_only.txt",
        "invalid_eval_only.txt",
        "invalid_missing_tps.txt",
        "invalid_nonfinite.txt",
        "invalid_zero.txt",
    ],
)
def test_partial_or_malformed_output_returns_parse_failed(name: str) -> None:
    result = parse_llama_cpp_output(stdout="", stderr=_fixture(name))

    assert result.success is False
    assert result.metrics is None
    assert result.error_type == ErrorType.PARSE_FAILED
    assert result.error_message


def test_parser_does_not_derive_tps_when_output_omits_it() -> None:
    result = parse_llama_cpp_output(
        stdout="",
        stderr=(
            "llama_print_timings: prompt eval time = 1000.00 ms / 10 tokens "
            "(100.00 ms per token)\n"
            "llama_print_timings: eval time = 2000.00 ms / 20 runs "
            "(100.00 ms per token)\n"
        ),
    )

    assert result.success is False
    assert result.error_type == ErrorType.PARSE_FAILED


def test_unrelated_eval_words_are_ignored() -> None:
    result = parse_llama_cpp_output(
        stdout="model output mentions evaluation time but no runtime metrics",
        stderr="no timing block here",
    )

    assert result.success is False
    assert result.error_type == ErrorType.PARSE_FAILED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("load_time_seconds", 0.0),
        ("prompt_eval_time_seconds", 0.0),
        ("prompt_tokens_per_second", float("inf")),
        ("eval_time_seconds", -1.0),
        ("generation_tokens_per_second", float("nan")),
    ],
)
def test_timing_model_requires_positive_finite_values(field: str, value: float) -> None:
    values = {
        "load_time_seconds": 0.5,
        "prompt_eval_time_seconds": 1.0,
        "prompt_tokens_per_second": 2.0,
        "eval_time_seconds": 3.0,
        "generation_tokens_per_second": 4.0,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        LlamaCppTimingMetrics.model_validate(values)


def test_parse_result_model_rejects_failure_without_parse_failed() -> None:
    with pytest.raises(ValidationError, match="parse_failed"):
        LlamaCppOutputParseResult(
            success=False,
            error_type=ErrorType.UNKNOWN,
            error_message="bad output",
        )


def test_parse_result_model_rejects_success_without_metrics() -> None:
    with pytest.raises(ValidationError, match="requires metrics"):
        LlamaCppOutputParseResult(success=True)


def test_parser_reads_direct_load_time_without_deriving_it() -> None:
    result = parse_llama_cpp_output(
        stdout="",
        stderr=(
            "llama_perf_context_print: load time = 250.00 ms\n"
            "llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens "
            "(10.00 ms per token, 100.00 tokens per second)\n"
            "llama_perf_context_print: eval time = 200.00 ms / 20 runs "
            "(10.00 ms per token, 100.00 tokens per second)\n"
        ),
    )
    assert result.success is True
    assert result.metrics is not None
    assert result.metrics.load_time_seconds == 0.25


def test_malformed_load_time_does_not_fabricate_value() -> None:
    result = parse_llama_cpp_output(
        stdout="",
        stderr=(
            "llama_perf_context_print: load time = nan ms\n"
            "llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens "
            "(10.00 ms per token, 100.00 tokens per second)\n"
            "llama_perf_context_print: eval time = 200.00 ms / 20 runs "
            "(10.00 ms per token, 100.00 tokens per second)\n"
        ),
    )
    assert result.success is True
    assert result.metrics is not None
    assert result.metrics.load_time_seconds is None
